#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Needle-in-a-haystack retrieval check for a Flint OpenAI-compatible server.

Builds a synthetic haystack (no dataset download), sizes each prompt with the
server's ``/tokenize`` endpoint so the chat prompt lands on the requested
token count, hides one access code at each requested depth and asks the model
to return it. Every prompt starts with a unique session line, so no request
reuses another one's prefix cache unless ``--share-prefix`` is given.

Exit status: 0 when at least ``--min-pass`` of the probes pass (all by
default), 1 otherwise, 2 when the server cannot be reached.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
import time
import urllib.error
import urllib.request
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import regex as re

ADJECTIVES = [
    "amber", "brisk", "cobalt", "dusty", "eager", "faded", "gentle", "hollow",
    "icy", "jagged", "keen", "lucid", "mellow", "narrow", "olive", "pale",
    "quiet", "rusty", "silver", "tidy", "umber", "vivid", "wooden", "young",
]  # fmt: skip
NOUNS = [
    "lantern", "barge", "orchard", "ledger", "furnace", "compass", "harbor",
    "quarry", "satchel", "turbine", "village", "wagon", "beacon", "cellar",
    "drum", "engine", "ferry", "garden", "hammer", "kettle", "ladder", "mill",
]  # fmt: skip
VERBS = [
    "inspected", "repaired", "counted", "painted", "loaded", "measured",
    "cleaned", "moved", "sorted", "weighed", "checked", "labelled",
]  # fmt: skip
PLACES = [
    "Northgate", "Riverside", "Old Quay", "Hillcrest", "Saltmarsh", "Eastfield",
    "Copper Row", "Mill Lane", "Westbrook", "Fairhaven", "Stonebridge",
]  # fmt: skip
VAULTS = ["Heron", "Juniper", "Marlin", "Obsidian", "Quartz", "Sable", "Tamarack"]

INTRO = (
    "Below is a long maintenance log. Exactly one entry in it states the "
    "access code of a vault. Read the whole log, then answer the question "
    "that follows it."
)


@dataclass
class Probe:
    target_tokens: int
    depth: float
    prompt_tokens: int | None = None
    cached_tokens: int | None = None
    expected: str = ""
    answer: str = ""
    finish_reason: str | None = None
    seconds: float = 0.0
    passed: bool = False
    note: str = ""


def _api_key() -> str:
    return os.environ.get("OPENAI_API_KEY") or os.environ.get("VLLM_API_KEY") or "EMPTY"


def _request(url: str, payload: dict[str, Any] | None, timeout: float) -> Any:
    data = None if payload is None else json.dumps(payload).encode()
    request = urllib.request.Request(
        url,
        data=data,
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {_api_key()}",
        },
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return json.loads(response.read())


def parse_length(text: str) -> int:
    """Parse '32768', '32k' (x1024) or '250000'."""
    text = text.strip().lower()
    if text.endswith("k"):
        return int(float(text[:-1]) * 1024)
    return int(text)


class Haystack:
    """Deterministic filler sentences; numbers stay below six digits."""

    def __init__(self, seed: int) -> None:
        self.rng = random.Random(seed)
        self.sentences: list[str] = []

    def take(self, count: int) -> list[str]:
        while len(self.sentences) < count:
            i = len(self.sentences)
            r = self.rng
            self.sentences.append(
                f"Entry {i}: at {r.choice(PLACES)} the {r.choice(ADJECTIVES)} "
                f"{r.choice(NOUNS)} was {r.choice(VERBS)} by crew "
                f"{r.randint(1, 99)} after {r.randint(2, 900)} minutes, next to "
                f"the {r.choice(ADJECTIVES)} {r.choice(NOUNS)}."
            )
        return self.sentences[:count]


class NeedleCheck:
    def __init__(self, args: argparse.Namespace) -> None:
        self.args = args
        root = args.base_url.rstrip("/")
        self.root = root[: -len("/v1")] if root.endswith("/v1") else root
        self.haystack = Haystack(args.seed)
        self.rng = random.Random(args.seed + 1)
        self.max_model_len: int | None = None

    def tokens_per_sentence(self) -> float:
        sample = self.haystack.take(400)
        response = _request(
            f"{self.root}/tokenize",
            {
                "model": self.args.model,
                "prompt": " ".join(sample),
                "add_special_tokens": False,
            },
            120,
        )
        self.max_model_len = response.get("max_model_len")
        return response["count"] / len(sample)

    def messages(
        self, salt: str, count: int, depth: float, needle: str, question: str
    ) -> list[dict[str, str]]:
        sentences = list(self.haystack.take(count))
        sentences.insert(int(round(depth * count)), needle)
        body = "\n".join(sentences)
        content = f"{salt}{INTRO}\n\n{body}\n\n{question}"
        return [{"role": "user", "content": content}]

    def count(self, messages: list[dict[str, str]]) -> int:
        response = _request(
            f"{self.root}/tokenize",
            {
                "model": self.args.model,
                "messages": messages,
                "add_generation_prompt": True,
            },
            300,
        )
        return int(response["count"])

    def fit(
        self, target: int, per_sentence: float, depth: float, parts: tuple[str, ...]
    ) -> tuple[list[dict[str, str]], int]:
        """Find the sentence count whose chat prompt is closest to target."""
        salt, needle, question = parts
        tolerance = max(32, int(target * 0.002))
        count = max(1, int(target / per_sentence))
        best: tuple[int, int, list[dict[str, str]]] | None = None
        for _ in range(8):
            messages = self.messages(salt, count, depth, needle, question)
            tokens = self.count(messages)
            if best is None or abs(tokens - target) < abs(best[0] - target):
                best = (tokens, count, messages)
            diff = target - tokens
            if abs(diff) <= tolerance:
                break
            step = int(diff / per_sentence)
            count = max(1, count + (step if step else (1 if diff > 0 else -1)))
        assert best is not None
        return best[2], best[0]

    def run_probe(self, target: int, depth: float, per_sentence: float) -> Probe:
        probe = Probe(target_tokens=target, depth=depth)
        vault = f"{self.rng.choice(VAULTS)}-{self.rng.randint(10, 99)}"
        code = str(self.rng.randint(10_000_000, 99_999_999))
        probe.expected = code
        needle = f"Entry note: the access code for the vault named {vault} is {code}."
        question = (
            f"Question: what is the access code for the vault named {vault}? "
            "Reply with the code only."
        )
        salt = (
            ""
            if self.args.share_prefix
            else f"Session {self.rng.getrandbits(48):012x}. "
        )
        limit = target
        if self.max_model_len:
            limit = min(target, self.max_model_len - self.args.max_tokens)
        messages, tokens = self.fit(
            limit, per_sentence, depth, (salt, needle, question)
        )
        payload: dict[str, Any] = {
            "model": self.args.model,
            "messages": messages,
            "max_tokens": self.args.max_tokens,
        }
        if self.args.temperature is not None:
            payload["temperature"] = self.args.temperature
        if self.args.seed is not None:
            payload["seed"] = self.args.seed
        start = time.monotonic()
        try:
            response = _request(
                f"{self.root}/v1/chat/completions", payload, self.args.timeout
            )
        except (urllib.error.URLError, OSError) as exc:
            probe.note = f"request failed: {exc}"
            probe.prompt_tokens = tokens
            return probe
        probe.seconds = round(time.monotonic() - start, 2)
        choice = response["choices"][0]
        message = choice["message"]
        usage = response.get("usage") or {}
        details = usage.get("prompt_tokens_details") or {}
        probe.prompt_tokens = usage.get("prompt_tokens", tokens)
        probe.cached_tokens = details.get("cached_tokens")
        probe.finish_reason = choice.get("finish_reason")
        content = message.get("content") or ""
        reasoning = message.get("reasoning") or message.get("reasoning_content") or ""
        digits = [d.replace(",", "") for d in re.findall(r"\d[\d,]*", content)]
        probe.answer = content.strip()[:80]
        probe.passed = code in digits
        if not probe.passed and code in reasoning:
            probe.note = "code only in reasoning"
            if probe.finish_reason == "length":
                probe.note += " (ran out of max_tokens)"
        return probe

    def run(self) -> int:
        try:
            _request(f"{self.root}/v1/models", None, 30)
        except (urllib.error.URLError, OSError) as exc:
            print(f"server at {self.root} is not reachable: {exc}", file=sys.stderr)
            return 2
        per_sentence = self.tokens_per_sentence()
        lengths = [parse_length(x) for x in self.args.lengths.split(",") if x]
        depths = [float(x) for x in self.args.depths.split(",") if x]
        print(
            f"{per_sentence:.2f} tokens per filler sentence, "
            f"max_model_len={self.max_model_len}",
            flush=True,
        )
        probes: list[Probe] = []
        for target in lengths:
            for depth in depths:
                probe = self.run_probe(target, depth, per_sentence)
                probes.append(probe)
                print(
                    f"{'PASS' if probe.passed else 'FAIL'} target={target} "
                    f"depth={depth:.2f} prompt={probe.prompt_tokens} "
                    f"cached={probe.cached_tokens} {probe.seconds:.1f} s "
                    f"answer={probe.answer!r} {probe.note}",
                    flush=True,
                )
        passed = sum(p.passed for p in probes)
        needed = len(probes) if self.args.min_pass is None else self.args.min_pass
        ok = passed >= needed
        if self.args.report:
            report = {
                "passed": ok,
                "score": f"{passed}/{len(probes)}",
                "probes": [asdict(p) for p in probes],
            }
            Path(self.args.report).write_text(json.dumps(report, indent=2) + "\n")
        if self.args.markdown:
            lines = [
                "| Target tokens | Prompt tokens | Depth | Result | Seconds | Note |",
                "| ---: | ---: | ---: | --- | ---: | --- |",
            ]
            for p in probes:
                lines.append(
                    f"| {p.target_tokens} | {p.prompt_tokens} | {p.depth:.2f} | "
                    f"{'PASS' if p.passed else 'FAIL'} | {p.seconds:.1f} | "
                    f"{p.note.replace('|', '/')} |"
                )
            Path(self.args.markdown).write_text("\n".join(lines) + "\n")
        print(f"needle: {passed}/{len(probes)} passed (need {needed})")
        return 0 if ok else 1


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--base-url", default="http://127.0.0.1:8000")
    parser.add_argument("--model", default="flint")
    parser.add_argument(
        "--lengths",
        default="32768,131072,250000",
        help="comma-separated prompt sizes in tokens ('32k' means 32768)",
    )
    parser.add_argument("--depths", default="0.5", help="comma-separated, 0..1")
    parser.add_argument("--max-tokens", type=int, default=2048)
    parser.add_argument(
        "--temperature",
        type=float,
        default=None,
        help="default: the server's generation config",
    )
    parser.add_argument("--seed", type=int, default=20261028)
    parser.add_argument("--timeout", type=float, default=3600.0)
    parser.add_argument("--min-pass", type=int, default=None)
    parser.add_argument(
        "--share-prefix",
        action="store_true",
        help="drop the per-request session line so probes share prefix cache",
    )
    parser.add_argument("--report", help="write a JSON report here")
    parser.add_argument("--markdown", help="write a markdown result table here")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    return NeedleCheck(parse_args(argv)).run()


if __name__ == "__main__":
    sys.exit(main())
