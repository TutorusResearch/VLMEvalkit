# 🧮 Multimodal Korean Mathematical Reasoning Benchmark

[한국어](/README.md) | English

> **Looking for the original VLMEvalKit documentation?** See [VLMEvalKit Documentation](/docs/en/KoreanMathBenchmark.md)

[![][github-contributors-shield]][github-contributors-link] • [![][github-forks-shield]][github-forks-link] • [![][github-stars-shield]][github-stars-link] • [![][github-issues-shield]][github-issues-link] • [![][github-license-shield]][github-license-link]

## Overview

This repository hosts a comprehensive benchmark for evaluating **multimodal large language models (MLLMs)** on **Korean mathematical reasoning tasks**. Built on top of [VLMEvalKit](https://github.com/open-compass/VLMEvalKit), this benchmark provides a rigorous evaluation framework for assessing model capabilities in understanding and solving mathematical problems in Korean.

## Features

- **Korean Language Focus**: Dedicated evaluation for Korean mathematical reasoning
- **Multimodal Assessment**: Tests models on both text-based and visual mathematical problems
- **Comprehensive Coverage**: Includes various mathematical topics and difficulty levels
- **Standardized Evaluation**: Consistent evaluation protocols across different models
- **Built on VLMEvalKit**: Leverages the robust infrastructure of VLMEvalKit

## Getting Started

> **Note**: This repository is built on top of [VLMEvalKit](https://github.com/open-compass/VLMEvalKit.git). Please refer to the original repository for more detailed documentation.

### Installation

#### 1. Install PyTorch

First, ensure PyTorch is installed. Visit [PyTorch official website](https://pytorch.org/get-started/locally/) for installation instructions specific to your system.

```bash
# Example: Install PyTorch with CUDA support
pip install torch torchvision torchaudio
```

#### 2. Install Dependencies

Install the required dependencies using pip:

```bash
pip install -e .
```

This will install all dependencies listed in `requirements.txt`.

#### 3. Install Flash Attention

Install flash-attention for optimized inference:

```bash
pip install flash-attn --no-build-isolation
```

## Acknowledgments

This benchmark is built on top of [VLMEvalKit](https://github.com/open-compass/VLMEvalKit). We thank the VLMEvalKit team for providing an excellent foundation for multimodal evaluation.

For the original VLMEvalKit documentation, please visit [VLMEvalKit Documentation](/docs/en/KoreanMathBenchmark.md).

---

[github-contributors-link]: https://github.com/open-compass/VLMEvalKit/graphs/contributors
[github-contributors-shield]: https://img.shields.io/github/contributors/open-compass/VLMEvalKit?color=c4f042&labelColor=black&style=flat-square
[github-forks-link]: https://github.com/open-compass/VLMEvalKit/network/members
[github-forks-shield]: https://img.shields.io/github/forks/open-compass/VLMEvalKit?color=8ae8ff&labelColor=black&style=flat-square
[github-issues-link]: https://github.com/open-compass/VLMEvalKit/issues
[github-issues-shield]: https://img.shields.io/github/issues/open-compass/VLMEvalKit?color=ff80eb&labelColor=black&style=flat-square
[github-license-link]: https://github.com/open-compass/VLMEvalKit/blob/main/LICENSE
[github-license-shield]: https://img.shields.io/github/license/open-compass/VLMEvalKit?color=white&labelColor=black&style=flat-square
[github-stars-link]: https://github.com/open-compass/VLMEvalKit/stargazers
[github-stars-shield]: https://img.shields.io/github/stars/open-compass/VLMEvalKit?color=ffcb47&labelColor=black&style=flat-square
