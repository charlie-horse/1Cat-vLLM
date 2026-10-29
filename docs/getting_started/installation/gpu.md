---
toc_depth: 3
---

# GPU

This fork of vLLM builds for NVIDIA CUDA GPUs and targets V100 (SM70).

--8<-- "docs/getting_started/installation/gpu.cuda.inc.md:installation"

## Requirements

- OS: Linux
- Python: 3.10 -- 3.13

!!! note
    vLLM does not support Windows natively. To run vLLM on Windows, you can use the Windows Subsystem for Linux (WSL) with a compatible Linux distribution, or use some community-maintained forks, e.g. [https://github.com/SystemPanic/vllm-windows](https://github.com/SystemPanic/vllm-windows).

--8<-- "docs/getting_started/installation/gpu.cuda.inc.md:requirements"

## Set up using Python

### Create a new Python environment

--8<-- "docs/getting_started/installation/python_env_setup.inc.md"

--8<-- "docs/getting_started/installation/gpu.cuda.inc.md:set-up-using-python"

### Pre-built wheels {#pre-built-wheels}

--8<-- "docs/getting_started/installation/gpu.cuda.inc.md:pre-built-wheels"

### Build wheel from source

--8<-- "docs/getting_started/installation/gpu.cuda.inc.md:build-wheel-from-source"

## Set up using Docker

### Pre-built images

--8<-- [start:pre-built-images]

--8<-- "docs/getting_started/installation/gpu.cuda.inc.md:pre-built-images"

--8<-- [end:pre-built-images]

### Build image from source

--8<-- [start:build-image-from-source]

--8<-- "docs/getting_started/installation/gpu.cuda.inc.md:build-image-from-source"

--8<-- [end:build-image-from-source]

## Supported features

--8<-- "docs/getting_started/installation/gpu.cuda.inc.md:supported-features"
