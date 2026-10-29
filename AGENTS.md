# Agent Instructions

This is a personal fork of 1Cat-vLLM for NVIDIA V100 (SM70) GPUs.

## Python environment

- Always use `uv` and `.venv/bin/python`. Never use system `python3` or bare
  `pip`/`pip install`.
- Create the environment with `uv venv --python 3.12`, install with
  `uv pip install ...`, and run tests with `.venv/bin/python -m pytest ...`.

## Code style

- Python lines are limited to 88 characters.

## Commits

- Add attribution trailers to every AI-assisted commit, for example
  `Co-authored-by: Claude`.

## Fork scope

- Target SM70/V100. Other GPUs are not supported or tested here.
- Keep the routes of models you are not working on unchanged.
- Every new fast path gets an environment kill switch: a `VLLM_SM70_*` flag
  registered in `vllm/envs.py` that restores the previous route.
