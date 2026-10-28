# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Reasoning parser for the GLM-4.7 ``<think>``/``<tool_call>`` output format.

It follows the upstream ``glm47`` parser, and also supports chat templates
whose generation prompt stops at ``<|assistant|>``:

* If the generation prompt opens ``<think>`` (GLM-4.7), the output starts
  inside the reasoning block. Text up to ``</think>`` is reasoning.
  ``enable_thinking=False`` or ``thinking=False`` turns extraction off,
  because the template then closes the block in the prompt.
* If the generation prompt stops at ``<|assistant|>``, the model opens
  ``<think>`` itself when it chooses to reason. Output with no ``<think>``
  is the answer and is returned as content. The output alone shows whether
  the model reasoned, so the thinking switches are not needed.

With both layouts:

* ``</think>`` and ``<tool_call>`` end reasoning. ``<tool_call>`` and the
  rest of the output become content for the tool parser.
* A stray ``</think>`` is dropped, and trailing whitespace is trimmed from
  the reasoning.
* ``is_reasoning_end`` ignores markers from earlier turns of the prompt.

If the tokenizer has no chat template, the parser treats the output as
starting inside reasoning, like upstream.
"""

from collections.abc import Container, Iterable, Sequence
from typing import TYPE_CHECKING

from vllm.entrypoints.openai.engine.protocol import DeltaMessage
from vllm.logger import init_logger
from vllm.reasoning.abs_reasoning_parsers import ReasoningParser

if TYPE_CHECKING:
    from vllm.entrypoints.openai.chat_completion.protocol import ChatCompletionRequest
    from vllm.entrypoints.openai.responses.protocol import ResponsesRequest
    from vllm.tokenizers import TokenizerLike

logger = init_logger(__name__)

THINK_START = "<think>"
THINK_END = "</think>"
TOOL_CALL_START = "<tool_call>"

# Special tokens that open a turn in a rendered prompt. Reasoning markers
# before the last of them belong to an earlier turn.
TURN_BOUNDARY_TOKENS = (
    "<|system|>",
    "<|developer|>",
    "<|user|>",
    "<|assistant|>",
    "<|observation|>",
)


def generation_prompt_opens_think(chat_template: object) -> bool | None:
    """Return whether a chat template's generation prompt leaves ``<think>``
    open, or ``None`` if this cannot be determined.

    It looks at the template text after the last ``add_generation_prompt``.
    An empty ``<think></think>`` block (thinking disabled) does not count.
    """
    if isinstance(chat_template, dict):
        chat_template = chat_template.get("default")
    if not isinstance(chat_template, str):
        return None
    index = chat_template.rfind("add_generation_prompt")
    if index < 0:
        return None
    tail = chat_template[index:].replace(THINK_START + THINK_END, "")
    return THINK_START in tail


class Glm47ReasoningParser(ReasoningParser):
    """Reasoning parser for GLM-4.7-style ``<think>`` output.

    See the module docstring for the two prompt layouts it supports.
    """

    def __init__(self, tokenizer: "TokenizerLike", *args, **kwargs):
        super().__init__(tokenizer, *args, **kwargs)
        if not self.model_tokenizer:
            raise ValueError(
                "The model tokenizer must be passed to the ReasoningParser "
                "constructor during construction."
            )

        vocab = self.vocab
        think_start_id = vocab.get(THINK_START)
        think_end_id = vocab.get(THINK_END)
        if think_start_id is None or think_end_id is None:
            raise RuntimeError(
                f"{self.__class__.__name__} could not locate the think "
                "start/end tokens in the tokenizer!"
            )
        self._think_start_id: int = think_start_id
        self._think_end_id: int = think_end_id
        self._tool_call_id: int | None = vocab.get(TOOL_CALL_START)
        self._end_ids = frozenset(
            i for i in (think_end_id, self._tool_call_id) if i is not None
        )
        self._boundary_ids = frozenset(
            vocab[token] for token in TURN_BOUNDARY_TOKENS if token in vocab
        )
        self._markers: tuple[str, ...] = (THINK_START, THINK_END)
        if self._tool_call_id is not None:
            self._markers += (TOOL_CALL_START,)

        prompt_opens_think = generation_prompt_opens_think(
            getattr(tokenizer, "chat_template", None)
        )
        # The model opens <think> itself only when the template is known not
        # to open it; otherwise keep the upstream behaviour.
        self.model_opens_think = prompt_opens_think is False

        chat_kwargs = kwargs.get("chat_template_kwargs") or {}
        thinking = chat_kwargs.get("thinking")
        enable_thinking = chat_kwargs.get("enable_thinking")
        self.thinking_enabled = (
            self.model_opens_think
            or (thinking is None and enable_thinking is None)
            or bool(thinking)
            or bool(enable_thinking)
        )
        # Whether the output starts inside the reasoning block.
        self._starts_in_reasoning = self.thinking_enabled and not self.model_opens_think

        logger.info_once(
            "glm47 reasoning parser: the %s opens <think>.",
            "model" if self.model_opens_think else "chat template",
        )

        # Streaming state, reset when a new stream starts.
        self._reset_streaming_state()

    def _reset_streaming_state(self) -> None:
        self._in_reasoning = self._starts_in_reasoning
        self._reasoning_done = False
        self._held_whitespace = ""

    @property
    def reasoning_start_str(self) -> str:
        return THINK_START

    @property
    def reasoning_end_str(self) -> str:
        return THINK_END

    def _parse(
        self,
        text: str,
        in_reasoning: bool,
        reasoning_done: bool,
        markers: Container[str],
    ) -> tuple[str, str, bool, bool]:
        """Split ``text`` into (reasoning, content, in_reasoning, done).

        Only the markers in ``markers`` are recognised. Once reasoning is
        done, the rest of the text is content.
        """
        reasoning: list[str] = []
        content: list[str] = []
        pos = 0
        while not reasoning_done and pos < len(text):
            index, marker = len(text), ""
            for candidate in self._markers:
                found = text.find(candidate, pos) if candidate in markers else -1
                if 0 <= found < index:
                    index, marker = found, candidate
            (reasoning if in_reasoning else content).append(text[pos:index])
            pos = index + len(marker)
            if marker == THINK_START:
                in_reasoning = True
            elif marker:
                # </think> is markup; <tool_call> is content for the tool parser.
                in_reasoning, reasoning_done = False, True
                if marker == TOOL_CALL_START:
                    content.append(marker)
        if reasoning_done:
            content.append(text[pos:])
        return "".join(reasoning), "".join(content), in_reasoning, reasoning_done

    def _turn_start(self, input_ids: Sequence[int]) -> int:
        for i in range(len(input_ids) - 1, -1, -1):
            if input_ids[i] in self._boundary_ids:
                return i + 1
        return 0

    def is_reasoning_end(self, input_ids: Sequence[int]) -> bool:
        if not self.thinking_enabled:
            return True
        # Only the current turn counts. A prompt that ends at a turn
        # boundary has not ended reasoning yet.
        for i in range(len(input_ids) - 1, -1, -1):
            token_id = input_ids[i]
            if token_id in self._end_ids:
                return True
            if token_id == self._think_start_id or token_id in self._boundary_ids:
                return False
        return False

    def is_reasoning_end_streaming(
        self, input_ids: Sequence[int], delta_ids: Iterable[int]
    ) -> bool:
        if not self.thinking_enabled:
            return True
        return any(token_id in self._end_ids for token_id in delta_ids)

    def extract_content_ids(self, input_ids: list[int]) -> list[int]:
        if not self.thinking_enabled:
            return input_ids
        start = self._turn_start(input_ids)
        in_reasoning = self._starts_in_reasoning
        for i in range(start, len(input_ids)):
            token_id = input_ids[i]
            if token_id == self._think_end_id:
                return input_ids[i + 1 :]
            if token_id == self._tool_call_id:
                return input_ids[i:]
            if token_id == self._think_start_id:
                in_reasoning = True
        return [] if in_reasoning else input_ids[start:]

    def count_reasoning_tokens(self, token_ids: Sequence[int]) -> int:
        if not self.thinking_enabled:
            return 0
        count = 0
        in_reasoning = self._starts_in_reasoning
        for token_id in token_ids:
            if token_id in self._end_ids:
                break
            if token_id == self._think_start_id:
                in_reasoning = True
            elif in_reasoning:
                count += 1
        return count

    def extract_reasoning(
        self, model_output: str, request: "ChatCompletionRequest | ResponsesRequest"
    ) -> tuple[str | None, str | None]:
        if not self.thinking_enabled:
            return None, model_output
        reasoning, content, _, _ = self._parse(
            model_output, self._starts_in_reasoning, False, self._markers
        )
        return reasoning.rstrip() or None, content or None

    def extract_reasoning_streaming(
        self,
        previous_text: str,
        current_text: str,
        delta_text: str,
        previous_token_ids: Sequence[int],
        current_token_ids: Sequence[int],
        delta_token_ids: Sequence[int],
    ) -> DeltaMessage | None:
        if not previous_text and not previous_token_ids:
            self._reset_streaming_state()
        if not self.thinking_enabled or self._reasoning_done:
            return DeltaMessage(content=delta_text) if delta_text else None

        # A marker counts only if its token was generated; the same text
        # spelled with ordinary tokens is plain text.
        markers: Container[str] = self._markers
        if delta_token_ids:
            markers = {
                marker
                for marker, token_id in (
                    (THINK_START, self._think_start_id),
                    (THINK_END, self._think_end_id),
                    (TOOL_CALL_START, self._tool_call_id),
                )
                if token_id is not None and token_id in delta_token_ids
            }
        reasoning, content, self._in_reasoning, self._reasoning_done = self._parse(
            delta_text, self._in_reasoning, self._reasoning_done, markers
        )

        # Hold trailing whitespace back until more reasoning follows, so the
        # streamed reasoning matches the trimmed non-streaming result.
        if reasoning:
            reasoning = self._held_whitespace + reasoning
            trimmed = reasoning.rstrip()
            self._held_whitespace = reasoning[len(trimmed) :]
            reasoning = trimmed
        if self._reasoning_done:
            self._held_whitespace = ""

        if not reasoning and not content:
            return None
        return DeltaMessage(reasoning=reasoning or None, content=content or None)
