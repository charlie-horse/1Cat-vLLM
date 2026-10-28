# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tests for the ``glm47`` reasoning parser.

The tokenizer is a byte-level BPE built in memory with the turn and
reasoning tokens of the GLM-4.7 prompt format, so no network is needed.
"""

import functools
import itertools
import json

import pytest
from tokenizers import Tokenizer, decoders, models, pre_tokenizers
from transformers import AddedToken, PreTrainedTokenizerFast

from tests.reasoning.utils import run_reasoning_extraction
from vllm.entrypoints.openai.chat_completion.protocol import ChatCompletionRequest
from vllm.parser.abstract_parser import _WrappedParser
from vllm.reasoning import ReasoningParser, ReasoningParserManager
from vllm.reasoning.deepseek_v3_reasoning_parser import (
    DeepSeekV3ReasoningWithThinkingParser,
)
from vllm.reasoning.glm47_reasoning_parser import (
    Glm47ReasoningParser,
    generation_prompt_opens_think,
)
from vllm.tool_parsers.glm47_moe_tool_parser import Glm47MoeModelToolParser

TURN_TOKENS = [
    "<|system|>",
    "<|user|>",
    "<|assistant|>",
    "<|observation|>",
    "<|developer|>",
]
MARKUP_TOKENS = [
    "<think>",
    "</think>",
    "<tool_call>",
    "</tool_call>",
    "<arg_key>",
    "</arg_key>",
    "<arg_value>",
    "</arg_value>",
    "<tool_response>",
    "</tool_response>",
]

_HISTORY = (
    "{%- for m in messages -%}"
    "{%- if m.role == 'user' -%}<|user|>{{ m.content }}"
    "{%- elif m.role == 'assistant' -%}<|assistant|>"
    "{%- if m.reasoning_content %}<think>{{ m.reasoning_content }}</think>"
    "{% endif -%}{{ m.content or '' }}"
    "{%- elif m.role == 'tool' -%}"
    "<|observation|><tool_response>{{ m.content }}</tool_response>"
    "{%- endif -%}"
    "{%- endfor -%}"
)
# The model opens <think> itself: the generation prompt stops at the turn.
MODEL_OPENS_THINK_TEMPLATE = (
    _HISTORY + "{%- if add_generation_prompt -%}<|assistant|>{%- endif -%}"
)
# GLM-4.7 style: the generation prompt opens <think>.
PROMPT_OPENS_THINK_TEMPLATE = _HISTORY + (
    "{%- if add_generation_prompt -%}<|assistant|>"
    "{{- '</think>' if (enable_thinking is defined and not enable_thinking)"
    " else '<think>' -}}{%- endif -%}"
)
# GLM-4.5 style: only an empty block, when thinking is disabled.
EMPTY_THINK_TEMPLATE = _HISTORY + (
    "{%- if add_generation_prompt -%}<|assistant|>"
    "{{- '\\n<think></think>' if (enable_thinking is defined and not"
    " enable_thinking) else '' -}}{%- endif -%}"
)

CALL = (
    "<tool_call>get_weather<arg_key>city</arg_key><arg_value>Paris</arg_value>"
    "<arg_key>days</arg_key><arg_value>3</arg_value></tool_call>"
)
CALL_ARGS = {"city": "Paris", "days": 3}
NO_ARG_CALL = "<tool_call>list_files</tool_call>"
TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "get_weather",
            "parameters": {
                "type": "object",
                "properties": {
                    "city": {"type": "string"},
                    "days": {"type": "integer"},
                },
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "list_files",
            "parameters": {"type": "object", "properties": {}},
        },
    },
]


@functools.cache
def _tokenizer(chat_template: str | None) -> PreTrainedTokenizerFast:
    alphabet = sorted(pre_tokenizers.ByteLevel.alphabet())
    backend = Tokenizer(
        models.BPE(vocab={c: i for i, c in enumerate(alphabet)}, merges=[])
    )
    backend.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    backend.decoder = decoders.ByteLevel()
    tokenizer = PreTrainedTokenizerFast(tokenizer_object=backend)
    tokenizer.add_special_tokens(
        {
            "additional_special_tokens": [
                AddedToken(t, special=True, normalized=False) for t in TURN_TOKENS
            ]
        }
    )
    tokenizer.add_tokens(
        [AddedToken(t, special=False, normalized=False) for t in MARKUP_TOKENS]
    )
    tokenizer.chat_template = chat_template
    return tokenizer


def _ids(tokenizer, text: str) -> list[int]:
    return tokenizer.encode(text, add_special_tokens=False)


def _parser(template: str | None = MODEL_OPENS_THINK_TEMPLATE, **kwargs):
    tokenizer = _tokenizer(template)
    parser_cls = ReasoningParserManager.get_reasoning_parser("glm47")
    return parser_cls(tokenizer, **kwargs)


def _extract(parser: ReasoningParser, output: str, streaming: bool):
    tokenizer = parser.model_tokenizer
    tokens = [
        tokenizer.convert_tokens_to_string([token])
        for token in tokenizer.tokenize(output)
    ]
    return run_reasoning_extraction(parser, tokens, streaming=streaming)


def test_registered():
    assert ReasoningParserManager.get_reasoning_parser("glm47") is (
        Glm47ReasoningParser
    )
    # glm45 keeps its own parser.
    assert ReasoningParserManager.get_reasoning_parser("glm45") is (
        DeepSeekV3ReasoningWithThinkingParser
    )
    parser = _parser()
    assert parser.reasoning_start_str == "<think>"
    assert parser.reasoning_end_str == "</think>"


@pytest.mark.parametrize(
    "template, expected",
    [
        pytest.param(MODEL_OPENS_THINK_TEMPLATE, False, id="model_opens"),
        pytest.param(PROMPT_OPENS_THINK_TEMPLATE, True, id="prompt_opens"),
        pytest.param(EMPTY_THINK_TEMPLATE, False, id="empty_block"),
        pytest.param({"default": PROMPT_OPENS_THINK_TEMPLATE}, True, id="dict"),
        pytest.param(_HISTORY, None, id="no_generation_prompt"),
        pytest.param(None, None, id="no_template"),
    ],
)
def test_generation_prompt_opens_think(template, expected):
    assert generation_prompt_opens_think(template) is expected


# The model opens <think> itself, so output without it is the answer.
# Each case: output, reasoning, content, is_reasoning_end(output).
MODEL_OPENS_CASES = {
    "with_think": (
        "<think>This is a reasoning section</think>This is the rest",
        "This is a reasoning section",
        "This is the rest",
        True,
    ),
    "without_think": ("This is the rest", None, "This is the rest", False),
    "complete_reasoning": (
        "<think>This is a reasoning section</think>",
        "This is a reasoning section",
        None,
        True,
    ),
    "multiline": (
        "<think>This is a reasoning\nsection</think>This is the rest\nThat",
        "This is a reasoning\nsection",
        "This is the rest\nThat",
        True,
    ),
    "empty_think": ("<think></think>The answer is 4.", None, "The answer is 4.", True),
    "trailing_whitespace": (
        "<think>Reason.\n\n</think>\n\nAnswer.",
        "Reason.",
        "\n\nAnswer.",
        True,
    ),
    "only_open_tag": (
        "<think>This is a reasoning section",
        "This is a reasoning section",
        None,
        False,
    ),
    "newline_before_think": (
        "\n<think>Reason.</think>Answer.",
        "Reason.",
        "\nAnswer.",
        True,
    ),
    "stray_think_end": ("Answer.</think> More.", None, "Answer. More.", True),
    "think_then_tool_call": (
        "<think>Need the weather.</think>" + CALL,
        "Need the weather.",
        CALL,
        True,
    ),
    "tool_call_ends_reasoning": (
        "<think>Need the weather." + CALL,
        "Need the weather.",
        CALL,
        True,
    ),
    "tool_call_without_think": (CALL, None, CALL, True),
    "text_then_tool_call": (
        "Let me check." + CALL,
        None,
        "Let me check." + CALL,
        True,
    ),
}


@pytest.mark.parametrize("streaming", [False, True], ids=["full", "stream"])
@pytest.mark.parametrize("case", MODEL_OPENS_CASES.keys())
def test_model_opens_think(case: str, streaming: bool):
    output, reasoning, content, is_end = MODEL_OPENS_CASES[case]
    parser = _parser()
    assert parser.model_opens_think

    assert _extract(parser, output, streaming) == (reasoning, content)
    assert parser.is_reasoning_end(_ids(parser.model_tokenizer, output)) is is_end


def test_thinking_switch_ignored_when_model_opens_think():
    # The template has no switch, so the model's own <think> still counts.
    for kwargs in ({"enable_thinking": False}, {"thinking": False}):
        parser = _parser(chat_template_kwargs=kwargs)
        assert parser.thinking_enabled
        for streaming in (False, True):
            assert _extract(parser, "<think>R</think>A", streaming) == ("R", "A")


def test_streaming_state_resets_for_new_stream():
    parser = _parser()
    assert _extract(parser, "<think>R</think>A", streaming=True) == ("R", "A")
    assert _extract(parser, "<think>S</think>B", streaming=True) == ("S", "B")
    assert _extract(parser, "Plain", streaming=True) == (None, "Plain")


@pytest.mark.parametrize(
    "prompt, is_end",
    [
        pytest.param("<|user|>What is 2+2?<|assistant|>", False, id="new_turn"),
        pytest.param(
            "<|user|>Q<|assistant|><think>Earlier.</think>A.<|user|>Q2<|assistant|>",
            False,
            id="after_reasoning_turn",
        ),
        pytest.param(
            "<|system|>Use <tool_call>{name}<arg_key>k</arg_key><arg_value>v"
            "</arg_value></tool_call>\n<|user|>Q<|assistant|>",
            False,
            id="tool_format_in_system_prompt",
        ),
        pytest.param(
            "<|user|>Q<|assistant|><think>R</think>" + CALL + "<|observation|>"
            "<tool_response>sunny</tool_response><|assistant|>",
            False,
            id="after_tool_response",
        ),
        pytest.param("<|user|>Q<|assistant|><think>Still", False, id="open_think"),
        pytest.param(
            "<|user|>Q<|assistant|><think>R</think>Partial",
            True,
            id="continue_after_reasoning",
        ),
        pytest.param(
            "<|user|>Q<|assistant|>Partial<tool_call>get_",
            True,
            id="continue_in_tool_call",
        ),
    ],
)
@pytest.mark.parametrize(
    "template",
    [MODEL_OPENS_THINK_TEMPLATE, PROMPT_OPENS_THINK_TEMPLATE],
    ids=["model_opens", "prompt_opens"],
)
def test_is_reasoning_end_full_prompt(template: str, prompt: str, is_end: bool):
    parser = _parser(template)
    assert parser.is_reasoning_end(_ids(parser.model_tokenizer, prompt)) is is_end


def test_is_reasoning_end_streaming():
    parser = _parser()
    vocab = parser.vocab
    text_id = vocab["a"]
    prefix = [vocab["<think>"], text_id]
    for marker, expected in (
        ("</think>", True),
        ("<tool_call>", True),
        ("<think>", False),
    ):
        delta = [text_id, vocab[marker]]
        assert parser.is_reasoning_end_streaming(prefix + delta, delta) is expected
        # The structured-output manager passes an iterator.
        delta_iter = itertools.islice(prefix + delta, len(prefix), None)
        assert parser.is_reasoning_end_streaming(prefix + delta, delta_iter) is (
            expected
        )
    assert not parser.is_reasoning_end_streaming(prefix, [text_id])


def test_extract_content_ids():
    parser = _parser()
    v = parser.vocab
    think, end, call = v["<think>"], v["</think>"], v["<tool_call>"]
    call_end, a, b = v["</tool_call>"], v["a"], v["b"]
    assert parser.extract_content_ids([think, a, end, b, call]) == [b, call]
    assert parser.extract_content_ids([think, a, call, b]) == [call, b]
    assert parser.extract_content_ids([a, call, b]) == [call, b]
    assert parser.extract_content_ids([a, b]) == [a, b]
    assert parser.extract_content_ids([think, a]) == []
    # A tool call from an earlier turn is not the end of this turn's reasoning.
    earlier = [call, a, call_end, v["<|observation|>"], b, v["<|assistant|>"]]
    assert parser.extract_content_ids(earlier + [think, a]) == []


@pytest.mark.parametrize(
    "output, count",
    [
        ("<think>abc</think>de", 3),
        ("abc", 0),
        ("<think>ab<tool_call>f", 2),
        ("<think>abc", 3),
    ],
)
def test_count_reasoning_tokens(output: str, count: int):
    parser = _parser()
    ids = _ids(parser.model_tokenizer, output)
    assert parser.count_reasoning_tokens(ids) == count


# GLM-4.7 style: the output starts inside the reasoning block.
PROMPT_OPENS_CASES = {
    "without_open_think": (
        "This is a reasoning section</think>This is the rest",
        "This is a reasoning section",
        "This is the rest",
        True,
    ),
    "with_think": (
        "<think>This is a reasoning section</think>This is the rest",
        "This is a reasoning section",
        "This is the rest",
        True,
    ),
    "without_think": ("This is the rest", "This is the rest", None, False),
    "tool_call_ends_reasoning": (
        "Need the weather." + CALL,
        "Need the weather.",
        CALL,
        True,
    ),
}


@pytest.mark.parametrize("streaming", [False, True], ids=["full", "stream"])
@pytest.mark.parametrize("case", PROMPT_OPENS_CASES.keys())
@pytest.mark.parametrize(
    "template",
    [PROMPT_OPENS_THINK_TEMPLATE, None],
    ids=["prompt_opens", "no_template"],
)
def test_prompt_opens_think(template, case: str, streaming: bool):
    output, reasoning, content, is_end = PROMPT_OPENS_CASES[case]
    parser = _parser(template)
    assert not parser.model_opens_think

    assert _extract(parser, output, streaming) == (reasoning, content)
    assert parser.is_reasoning_end(_ids(parser.model_tokenizer, output)) is is_end


@pytest.mark.parametrize("kwargs", [{"enable_thinking": False}, {"thinking": False}])
def test_prompt_opens_think_honours_thinking_switch(kwargs: dict):
    parser = _parser(PROMPT_OPENS_THINK_TEMPLATE, chat_template_kwargs=kwargs)
    assert not parser.thinking_enabled
    output = "Simple question.</think>4"
    for streaming in (False, True):
        assert _extract(parser, output, streaming) == (None, output)
    assert parser.is_reasoning_end(_ids(parser.model_tokenizer, "<|assistant|>"))


# End to end with the glm47 tool parser, as the chat completion server runs
# them. Each case: output, reasoning, content, [(tool name, arguments)].
END_TO_END_CASES = {
    "think_then_tool_call": (
        "<think>Need the weather.</think>" + CALL,
        "Need the weather.",
        None,
        [("get_weather", CALL_ARGS)],
    ),
    "tool_call_ends_reasoning": (
        "<think>Need the weather." + CALL,
        "Need the weather.",
        None,
        [("get_weather", CALL_ARGS)],
    ),
    "tool_call_without_think": (CALL, None, None, [("get_weather", CALL_ARGS)]),
    "text_then_tool_call": (
        "Let me check." + CALL,
        None,
        "Let me check.",
        [("get_weather", CALL_ARGS)],
    ),
    "no_arg_then_tool_call": (
        "<think>Look first.</think>" + NO_ARG_CALL + CALL,
        "Look first.",
        None,
        [("list_files", {}), ("get_weather", CALL_ARGS)],
    ),
    "answer_without_think": ("The answer is 4.", None, "The answer is 4.", []),
    "answer_after_think": (
        "<think>Easy.</think>The answer is 4.",
        "Easy.",
        "The answer is 4.",
        [],
    ),
}


def _request() -> ChatCompletionRequest:
    return ChatCompletionRequest(
        model="test-model",
        messages=[{"role": "user", "content": "Weather in Paris?"}],
        tools=TOOLS,
    )


def _run_streaming(tokenizer, request, output: str):
    parser_cls = type(
        "Glm47TestParser",
        (_WrappedParser,),
        {
            "reasoning_parser_cls": Glm47ReasoningParser,
            "tool_parser_cls": Glm47MoeModelToolParser,
        },
    )
    parser = parser_cls(tokenizer, request.tools)
    prompt_ids = _ids(tokenizer, "<|user|>Weather in Paris?<|assistant|>")
    ids = _ids(tokenizer, output)
    reasoning, content = "", ""
    calls: dict[int, list[str]] = {}
    previous = ""
    for i, token_id in enumerate(ids):
        current = tokenizer.decode(ids[: i + 1])
        delta = parser.parse_delta(
            current[len(previous) :], [token_id], request, prompt_token_ids=prompt_ids
        )
        previous = current
        if delta is None:
            continue
        reasoning += delta.reasoning or ""
        content += delta.content or ""
        for call in delta.tool_calls or []:
            name_args = calls.setdefault(call.index, ["", ""])
            name_args[0] += call.function.name or ""
            name_args[1] += call.function.arguments or ""
    tool_calls = [(name, json.loads(args)) for name, args in calls.values()]
    return reasoning or None, content or None, tool_calls


def _run_full(tokenizer, request, output: str):
    reasoning, content = Glm47ReasoningParser(tokenizer).extract_reasoning(
        output, request
    )
    tool_parser = Glm47MoeModelToolParser(tokenizer, request.tools)
    info = tool_parser.extract_tool_calls(content or "", request)
    tool_calls = [
        (call.function.name, json.loads(call.function.arguments))
        for call in info.tool_calls
    ]
    return reasoning, info.content or None, tool_calls


@pytest.mark.parametrize("streaming", [False, True], ids=["full", "stream"])
@pytest.mark.parametrize("case", END_TO_END_CASES.keys())
def test_with_tool_parser(case: str, streaming: bool):
    output, reasoning, content, tool_calls = END_TO_END_CASES[case]
    tokenizer = _tokenizer(MODEL_OPENS_THINK_TEMPLATE)
    run = _run_streaming if streaming else _run_full
    assert run(tokenizer, _request(), output) == (reasoning, content, tool_calls)
