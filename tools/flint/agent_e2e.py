#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""End-to-end coding-agent check for a Flint OpenAI-compatible server.

Runs a small multi-turn agent with three workspace tools (``list_dir``,
``read_file`` and ``run_python``) against a temporary workspace and checks
the serving features a coding agent depends on:

* tool calls come back as parsed ``tool_calls`` with valid JSON arguments,
  and no raw tool or think markup leaks into ``content``;
* reasoning comes back in its own field when the model reasons, and the
  reasoning re-sent in the history reaches the prompt (compared against the
  server's ``/tokenize`` count with and without it);
* ``usage.prompt_tokens_details.cached_tokens`` is positive on later turns
  (prefix caching plus ``--enable-prompt-tokens-details``);
* ``tool_choice="none"`` returns text and no tool calls;
* a streamed request assembles a complete tool call from its deltas.

``run_python`` executes model-written code in a subprocess with the
temporary workspace as working directory, an isolated interpreter and a
timeout. It is a convenience boundary for a local model, not a security
sandbox.

Exit status: 0 when every check passes (warnings allowed), 1 when a check
fails, 2 when the server cannot be reached.
"""

from __future__ import annotations

import argparse
import copy
import json
import os
import random
import subprocess
import sys
import tempfile
import time
import urllib.error
import urllib.request
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import regex as re

try:
    from openai import OpenAI
except ImportError:  # pragma: no cover - depends on the environment
    print("agent_e2e.py needs the openai package (use the repo .venv)", file=sys.stderr)
    raise

RAW_MARKUP = ("<think>", "</think>", "<tool_call>", "</tool_call>", "<arg_key>")
ANSWER_RE = re.compile(r"ANSWER:\s*\$?\s*(-?[\d,]+)")

TOOLS: list[dict[str, Any]] = [
    {
        "type": "function",
        "function": {
            "name": "list_dir",
            "description": "List the entries of a directory in the workspace. "
            "Directories end with '/'.",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": "Directory relative to the workspace "
                        "root; use '.' for the root.",
                    }
                },
                "required": ["path"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "read_file",
            "description": "Read a UTF-8 text file in the workspace.",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": "File path relative to the workspace root.",
                    }
                },
                "required": ["path"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "run_python",
            "description": "Run a short Python 3 program with the workspace "
            "root as working directory and return its stdout and stderr.",
            "parameters": {
                "type": "object",
                "properties": {
                    "code": {
                        "type": "string",
                        "description": "Complete Python source code to execute.",
                    }
                },
                "required": ["code"],
            },
        },
    },
]
TOOL_REQUIRED = {
    tool["function"]["name"]: tool["function"]["parameters"]["required"]
    for tool in TOOLS
}

GUIDELINES = [
    "Read the relevant files before changing or summarizing them.",
    "Prefer small, reviewable steps and explain each step briefly.",
    "Never invent file contents; inspect them with the tools instead.",
    "Use run_python for arithmetic on data rather than mental math.",
    "Treat files under archive directories as historical unless told otherwise.",
    "Keep tool arguments minimal and use paths relative to the workspace root.",
    "Report the exact numbers you computed, without rounding.",
    "If a tool returns an error, read the message and adjust the next call.",
    "Do not modify files unless the task asks for it.",
    "Quote file names exactly as they appear in directory listings.",
    "Check CSV headers before indexing columns by position.",
    "State assumptions explicitly when the data is ambiguous.",
]


@dataclass
class Check:
    name: str
    status: str  # PASS, WARN or FAIL
    detail: str


@dataclass
class Turn:
    index: int
    prompt_tokens: int | None
    cached_tokens: int | None
    completion_tokens: int | None
    finish_reason: str | None
    reasoning_chars: int
    content_chars: int
    tool_calls: list[str] = field(default_factory=list)
    seconds: float = 0.0


def _root_url(url: str) -> str:
    url = url.rstrip("/")
    return url[: -len("/v1")] if url.endswith("/v1") else url


def _api_key() -> str:
    return os.environ.get("OPENAI_API_KEY") or os.environ.get("VLLM_API_KEY") or "EMPTY"


def _post_json(url: str, payload: dict[str, Any], timeout: float) -> dict[str, Any]:
    request = urllib.request.Request(
        url,
        data=json.dumps(payload).encode(),
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {_api_key()}",
        },
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return json.loads(response.read())


def _extra(obj: Any, key: str) -> Any:
    value = getattr(obj, key, None)
    if value is None:
        extra = getattr(obj, "model_extra", None) or {}
        value = extra.get(key)
    return value


def _reasoning_of(obj: Any) -> tuple[str | None, str | None]:
    """Return (reasoning text, response field name) of a message or delta."""
    for key in ("reasoning_content", "reasoning"):
        value = _extra(obj, key)
        if isinstance(value, str) and value.strip():
            return value, key
    return None, None


def _raw_markup(text: str | None) -> list[str]:
    return [tag for tag in RAW_MARKUP if text and tag in text]


class Workspace:
    """A temporary project directory the tools are confined to."""

    def __init__(self, seed: int, python_timeout: float) -> None:
        self._tmp = tempfile.TemporaryDirectory(prefix="flint-agent-")
        self.root = Path(self._tmp.name).resolve()
        self.python_timeout = python_timeout
        rng = random.Random(seed)
        items = ["bolt", "nut", "washer", "gear", "spring", "bearing", "valve"]
        rows = [(name, rng.randint(3, 60), rng.randint(2, 99)) for name in items]
        self.expected_total = sum(qty * price for _, qty, price in rows)
        stale = [(name, rng.randint(100, 900), rng.randint(2, 99)) for name in items]
        (self.root / "data" / "archive").mkdir(parents=True)
        (self.root / "README.md").write_text(
            "# Parts inventory\n\n"
            "The current inventory export is the CSV file directly under "
            "data/. Older exports live in data/archive/ and must be ignored.\n"
            "Columns: item, qty, unit_price (whole dollars).\n"
        )
        self._write_csv(self.root / "data" / "inventory_current.csv", rows)
        self._write_csv(self.root / "data" / "archive" / "inventory_2019.csv", stale)
        (self.root / "data" / "NOTES.txt").write_text(
            "Prices were last reviewed by the purchasing team.\n"
        )

    @staticmethod
    def _write_csv(path: Path, rows: list[tuple[str, int, int]]) -> None:
        lines = ["item,qty,unit_price"] + [f"{n},{q},{p}" for n, q, p in rows]
        path.write_text("\n".join(lines) + "\n")

    def close(self) -> None:
        self._tmp.cleanup()

    def _resolve(self, path: str) -> Path:
        target = (self.root / path).resolve()
        if target != self.root and self.root not in target.parents:
            raise ValueError(f"path escapes the workspace: {path}")
        return target

    def call(self, name: str, args: dict[str, Any]) -> str:
        try:
            if name == "list_dir":
                target = self._resolve(str(args.get("path", ".")))
                entries = sorted(
                    p.name + ("/" if p.is_dir() else "") for p in target.iterdir()
                )
                return "\n".join(entries) or "(empty)"
            if name == "read_file":
                target = self._resolve(str(args["path"]))
                return target.read_text(errors="replace")[:16000]
            if name == "run_python":
                return self._run_python(str(args["code"]))
            return f"error: unknown tool {name}"
        except Exception as exc:  # the model sees tool errors, as in a real agent
            return f"error: {type(exc).__name__}: {exc}"

    def _run_python(self, code: str) -> str:
        env = {
            "PATH": os.environ.get("PATH", "/usr/bin:/bin"),
            "HOME": str(self.root),
            "PYTHONDONTWRITEBYTECODE": "1",
        }
        try:
            proc = subprocess.run(
                [sys.executable, "-I", "-c", code],
                cwd=self.root,
                env=env,
                capture_output=True,
                text=True,
                timeout=self.python_timeout,
            )
        except subprocess.TimeoutExpired:
            return f"error: timed out after {self.python_timeout:.0f} s"
        out = f"exit code {proc.returncode}\nstdout:\n{proc.stdout}"
        if proc.stderr:
            out += f"\nstderr:\n{proc.stderr}"
        return out[:8000]


class AgentCheck:
    def __init__(self, args: argparse.Namespace) -> None:
        self.args = args
        self.root = _root_url(args.base_url)
        self.client = OpenAI(
            base_url=f"{self.root}/v1", api_key=_api_key(), timeout=args.timeout
        )
        self.workspace = Workspace(args.seed, args.python_timeout)
        self.checks: list[Check] = []
        self.turns: list[Turn] = []
        self.transcript: list[dict[str, Any]] = []
        self.reasoning_fields: set[str] = set()

    # ------------------------------------------------------------------ utils
    def add(self, name: str, status: str, detail: str) -> None:
        self.checks.append(Check(name, status, detail))
        print(f"[{status}] {name}: {detail}", flush=True)

    def sampling(self) -> dict[str, Any]:
        kwargs: dict[str, Any] = {"max_tokens": self.args.max_tokens}
        if self.args.temperature is not None:
            kwargs["temperature"] = self.args.temperature
        if self.args.seed is not None:
            kwargs["seed"] = self.args.seed
        return kwargs

    def system_prompt(self) -> str:
        lines = [
            "You are a careful coding agent working in a project workspace. "
            "Use the provided tools to inspect files and run Python; do not "
            "guess file contents. When the task is done, reply with a short "
            "explanation followed by a final line 'ANSWER: <integer>'.",
            "",
            "Workspace guidelines:",
        ]
        rounds = max(1, self.args.system_pad_rounds)
        for round_index in range(rounds):
            for rule_index, rule in enumerate(GUIDELINES):
                lines.append(f"{round_index + 1}.{rule_index + 1} {rule}")
        return "\n".join(lines)

    def tokenize_count(self, messages: list[dict[str, Any]]) -> int:
        payload = {
            "model": self.args.model,
            "messages": messages,
            "tools": TOOLS,
            "add_generation_prompt": True,
        }
        return int(_post_json(f"{self.root}/tokenize", payload, 120)["count"])

    @staticmethod
    def for_tokenize(
        messages: list[dict[str, Any]], keep_reasoning: bool
    ) -> list[dict[str, Any]]:
        """Copy history with reasoning under the field /tokenize reads."""
        out = copy.deepcopy(messages)
        for message in out:
            value = message.pop("reasoning_content", None)
            value = message.pop("reasoning", value)
            if keep_reasoning and value:
                message["reasoning"] = value
        return out

    # ------------------------------------------------------------- agent loop
    def run_agent(self) -> None:
        messages: list[dict[str, Any]] = [
            {"role": "system", "content": self.system_prompt()},
            {
                "role": "user",
                "content": "Find the current inventory CSV in the workspace "
                "and compute the total inventory value, i.e. the sum of "
                "qty * unit_price over all of its rows. Explore with list_dir "
                "and read_file, and use run_python for the arithmetic.",
            },
        ]
        bad_calls: list[str] = []
        markup: list[str] = []
        finish_mismatch = 0
        total_calls = 0
        final_content: str | None = None
        # Index of the first request whose history re-sends reasoning.
        reasoning_probe: int | None = None
        tool_names: set[str] = set()

        for index in range(self.args.max_turns):
            sent = copy.deepcopy(messages)
            start = time.monotonic()
            response = self.client.chat.completions.create(
                model=self.args.model,
                messages=sent,
                tools=TOOLS,
                tool_choice="auto",
                **self.sampling(),
            )
            elapsed = time.monotonic() - start
            choice = response.choices[0]
            message = choice.message
            reasoning, reasoning_field = _reasoning_of(message)
            if reasoning_field:
                self.reasoning_fields.add(reasoning_field)
            content = message.content or ""
            markup += [f"turn {index}: {tag}" for tag in _raw_markup(content)]

            usage = response.usage
            details = getattr(usage, "prompt_tokens_details", None) if usage else None
            turn = Turn(
                index=index,
                prompt_tokens=usage.prompt_tokens if usage else None,
                cached_tokens=getattr(details, "cached_tokens", None),
                completion_tokens=usage.completion_tokens if usage else None,
                finish_reason=choice.finish_reason,
                reasoning_chars=len(reasoning or ""),
                content_chars=len(content),
                seconds=round(elapsed, 3),
            )
            self.turns.append(turn)
            if reasoning_probe == index:
                # ``sent`` is exactly the history of this request.
                self.check_reasoning_resent(turn, sent)

            assistant: dict[str, Any] = {"role": "assistant", "content": content}
            if reasoning:
                assistant[self.args.history_reasoning_key] = reasoning
            calls = list(message.tool_calls or [])
            if calls:
                assistant["tool_calls"] = [
                    {
                        "id": call.id,
                        "type": "function",
                        "function": {
                            "name": call.function.name,
                            "arguments": call.function.arguments,
                        },
                    }
                    for call in calls
                ]
            messages.append(assistant)
            self.transcript.append(
                {
                    "turn": index,
                    "reasoning": reasoning,
                    "content": content,
                    "tool_calls": assistant.get("tool_calls", []),
                    "finish_reason": choice.finish_reason,
                }
            )
            print(
                f"turn {index}: {len(calls)} tool call(s), finish="
                f"{choice.finish_reason}, prompt={turn.prompt_tokens}, "
                f"cached={turn.cached_tokens}, reasoning={turn.reasoning_chars} "
                f"chars, {elapsed:.1f} s",
                flush=True,
            )
            if not calls:
                final_content = content
                break
            if choice.finish_reason != "tool_calls":
                finish_mismatch += 1
            if reasoning and reasoning_probe is None:
                # The next request re-sends this reasoning in its history.
                reasoning_probe = index + 1

            seen_ids: set[str] = set()
            for call in calls:
                total_calls += 1
                name = call.function.name
                tool_names.add(name)
                turn.tool_calls.append(name)
                problem = None
                try:
                    parsed = json.loads(call.function.arguments or "{}")
                except json.JSONDecodeError as exc:
                    parsed, problem = {}, f"arguments are not JSON ({exc})"
                if problem is None and not isinstance(parsed, dict):
                    problem = "arguments are not a JSON object"
                if problem is None and name not in TOOL_REQUIRED:
                    problem = "unknown tool name"
                if problem is None:
                    missing = [k for k in TOOL_REQUIRED[name] if k not in parsed]
                    if missing:
                        problem = f"missing required arguments {missing}"
                if not call.id or call.id in seen_ids:
                    problem = problem or "missing or duplicate tool_call id"
                seen_ids.add(call.id)
                if problem:
                    bad_calls.append(f"turn {index} {name}: {problem}")
                    result = f"error: {problem}"
                else:
                    result = self.workspace.call(name, parsed)
                messages.append(
                    {"role": "tool", "tool_call_id": call.id, "content": result}
                )
                self.transcript.append(
                    {"turn": index, "tool": name, "args": parsed, "result": result}
                )

        if total_calls == 0:
            self.add("tool_calls_parse", "FAIL", "the model never called a tool")
        elif bad_calls:
            self.add("tool_calls_parse", "FAIL", "; ".join(bad_calls))
        else:
            self.add(
                "tool_calls_parse",
                "PASS",
                f"{total_calls} call(s) parsed, tools used: {sorted(tool_names)}",
            )
        if finish_mismatch:
            self.add(
                "finish_reason_tool_calls",
                "WARN",
                f"{finish_mismatch} turn(s) returned tool calls without "
                "finish_reason='tool_calls'",
            )
        self.add(
            "no_raw_markup",
            "FAIL" if markup else "PASS",
            "; ".join(markup) or "content carries no think/tool markup",
        )
        self.check_reasoning_seen()
        self.check_cache_hits()
        self.check_answer(final_content)

    # ----------------------------------------------------------------- checks
    def check_reasoning_seen(self) -> None:
        reasoned = [t.index for t in self.turns if t.reasoning_chars]
        if reasoned:
            self.add(
                "reasoning_returned",
                "PASS",
                f"reasoning on turns {reasoned} in field(s) "
                f"{sorted(self.reasoning_fields)}",
            )
        else:
            status = "FAIL" if self.args.require_reasoning else "WARN"
            self.add("reasoning_returned", status, "the model did not reason")
        if reasoned and not any(
            c.name == "reasoning_resent" for c in self.checks
        ):  # the only reasoning turn was the last one
            self.add(
                "reasoning_resent",
                "WARN",
                "no request followed a reasoning turn; history not checked",
            )

    def check_reasoning_resent(self, turn: Turn, history: list[dict[str, Any]]) -> None:
        try:
            with_reasoning = self.tokenize_count(self.for_tokenize(history, True))
            without = self.tokenize_count(self.for_tokenize(history, False))
        except (urllib.error.URLError, KeyError, ValueError) as exc:
            self.add("reasoning_resent", "FAIL", f"/tokenize failed: {exc}")
            return
        actual = turn.prompt_tokens
        detail = (
            f"turn {turn.index} prompt_tokens={actual}; /tokenize with "
            f"reasoning={with_reasoning}, without={without} "
            f"(history key {self.args.history_reasoning_key!r})"
        )
        if with_reasoning <= without:
            self.add("reasoning_resent", "FAIL", "template drops reasoning: " + detail)
        elif actual is None:
            self.add("reasoning_resent", "FAIL", "no usage in response: " + detail)
        elif abs(actual - with_reasoning) < abs(actual - without):
            self.add("reasoning_resent", "PASS", detail)
        else:
            self.add("reasoning_resent", "FAIL", "server dropped it: " + detail)

    def check_cache_hits(self) -> None:
        later = self.turns[1:]
        if not later:
            self.add("prefix_cache_hits", "FAIL", "only one turn; nothing to compare")
            return
        if any(t.cached_tokens is None for t in later):
            self.add(
                "prefix_cache_hits",
                "FAIL",
                "usage.prompt_tokens_details.cached_tokens missing (serve with "
                "--enable-prefix-caching --enable-prompt-tokens-details)",
            )
            return
        misses = [t.index for t in later if not t.cached_tokens]
        detail = ", ".join(
            f"turn {t.index}: {t.cached_tokens}/{t.prompt_tokens}" for t in later
        )
        self.add("prefix_cache_hits", "FAIL" if misses else "PASS", detail)

    def check_answer(self, content: str | None) -> None:
        expected = self.workspace.expected_total
        if content is None:
            self.add(
                "final_answer",
                "FAIL",
                f"no final answer within {self.args.max_turns} turns",
            )
            return
        matches = ANSWER_RE.findall(content)
        got = int(matches[-1].replace(",", "")) if matches else None
        status = "PASS" if got == expected else "FAIL"
        self.add("final_answer", status, f"expected {expected}, got {got}")

    def run_tool_choice_none(self) -> None:
        messages = [
            {"role": "system", "content": self.system_prompt()},
            {
                "role": "user",
                "content": "Which files are in the workspace root? If you "
                "cannot use tools right now, say so in one sentence.",
            },
        ]
        response = self.client.chat.completions.create(
            model=self.args.model,
            messages=messages,
            tools=TOOLS,
            tool_choice="none",
            **self.sampling(),
        )
        message = response.choices[0].message
        content = message.content or ""
        problems = []
        if message.tool_calls:
            problems.append(f"{len(message.tool_calls)} tool call(s) returned")
        if not content.strip():
            problems.append("empty content")
        problems += [f"raw {tag}" for tag in _raw_markup(content)]
        self.add(
            "tool_choice_none",
            "FAIL" if problems else "PASS",
            "; ".join(problems) or f"text reply ({len(content)} chars), no tool calls",
        )

    def run_streaming(self) -> None:
        messages = [
            {"role": "system", "content": self.system_prompt()},
            {
                "role": "user",
                "content": "Call list_dir on the workspace root and wait for "
                "the result.",
            },
        ]
        stream = self.client.chat.completions.create(
            model=self.args.model,
            messages=messages,
            tools=TOOLS,
            tool_choice="auto",
            stream=True,
            stream_options={"include_usage": True},
            **self.sampling(),
        )
        calls: dict[int, dict[str, str]] = {}
        content, reasoning, finish = "", "", None
        for chunk in stream:
            if not chunk.choices:
                continue
            choice = chunk.choices[0]
            finish = choice.finish_reason or finish
            delta = choice.delta
            content += delta.content or ""
            for key in ("reasoning_content", "reasoning"):
                piece = _extra(delta, key)
                if isinstance(piece, str):
                    reasoning += piece
                    break
            for part in delta.tool_calls or []:
                entry = calls.setdefault(
                    part.index, {"id": "", "name": "", "arguments": ""}
                )
                entry["id"] = part.id or entry["id"]
                if part.function is not None:
                    entry["name"] += part.function.name or ""
                    entry["arguments"] += part.function.arguments or ""
        problems = [f"raw {tag}" for tag in _raw_markup(content)]
        listed = [c for c in calls.values() if c["name"] == "list_dir"]
        if not listed:
            problems.append(
                f"no streamed list_dir call (calls: {list(calls.values())})"
            )
        for call in listed:
            try:
                if not isinstance(json.loads(call["arguments"] or "{}"), dict):
                    problems.append("streamed arguments are not an object")
            except json.JSONDecodeError:
                problems.append(f"streamed arguments not JSON: {call['arguments']!r}")
        if listed and finish != "tool_calls":
            problems.append(f"finish_reason={finish}")
        self.add(
            "streaming_tool_call",
            "FAIL" if problems else "PASS",
            "; ".join(problems)
            or f"{len(calls)} call(s) assembled from deltas, reasoning "
            f"{len(reasoning)} chars",
        )

    # ------------------------------------------------------------------- main
    def run(self) -> int:
        stages = [self.run_agent, self.run_tool_choice_none]
        if self.args.stream_check:
            stages.append(self.run_streaming)
        try:
            for stage in stages:
                try:
                    stage()
                except Exception as exc:  # report and continue with the others
                    self.add(stage.__name__, "FAIL", f"{type(exc).__name__}: {exc}")
        finally:
            self.workspace.close()
        failed = [c.name for c in self.checks if c.status == "FAIL"]
        report = {
            "base_url": self.root,
            "model": self.args.model,
            "expected_total": self.workspace.expected_total,
            "checks": [asdict(c) for c in self.checks],
            "turns": [asdict(t) for t in self.turns],
            "transcript": self.transcript,
            "passed": not failed,
        }
        if self.args.report:
            Path(self.args.report).write_text(json.dumps(report, indent=2) + "\n")
        if self.args.markdown:
            lines = ["| Check | Status | Detail |", "| --- | --- | --- |"]
            for c in self.checks:
                detail = c.detail.replace("|", "/").replace("\n", " ")
                lines.append(f"| {c.name} | {c.status} | {detail} |")
            Path(self.args.markdown).write_text("\n".join(lines) + "\n")
        print("agent e2e:", "FAIL " + ", ".join(failed) if failed else "PASS")
        return 1 if failed else 0


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--base-url", default="http://127.0.0.1:8000")
    parser.add_argument("--model", default="flint")
    parser.add_argument("--max-turns", type=int, default=10)
    parser.add_argument("--max-tokens", type=int, default=4096)
    parser.add_argument(
        "--temperature",
        type=float,
        default=None,
        help="default: the server's generation config",
    )
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--timeout", type=float, default=900.0)
    parser.add_argument("--python-timeout", type=float, default=20.0)
    parser.add_argument(
        "--system-pad-rounds",
        type=int,
        default=4,
        help="repeat the guideline list this many times so the shared prefix "
        "spans several KV blocks",
    )
    parser.add_argument(
        "--history-reasoning-key",
        choices=("reasoning_content", "reasoning"),
        default="reasoning_content",
        help="field used to re-send assistant reasoning in the history",
    )
    parser.add_argument(
        "--require-reasoning",
        action="store_true",
        help="fail (instead of warn) when the model never reasons",
    )
    parser.add_argument("--no-stream-check", dest="stream_check", action="store_false")
    parser.add_argument("--report", help="write a JSON report here")
    parser.add_argument("--markdown", help="write a markdown check table here")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    root = _root_url(args.base_url)
    try:
        urllib.request.urlopen(f"{root}/health", timeout=10).close()
    except (urllib.error.URLError, OSError) as exc:
        print(f"server at {root} is not healthy: {exc}", file=sys.stderr)
        return 2
    return AgentCheck(args).run()


if __name__ == "__main__":
    sys.exit(main())
