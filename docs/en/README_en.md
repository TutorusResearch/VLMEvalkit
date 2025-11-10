# 🧮 Multimodal Korean Math Reasoning Benchmark

[한국어](/README.md) | English

[![HuggingFace](https://img.shields.io/badge/🤗%20HuggingFace-Tutoruslabs-yellow)](https://huggingface.co/Tutoruslabs) [![Paper](https://img.shields.io/badge/📄%20Paper-arXiv-red)]()

## Overview

This repository provides a comprehensive benchmark for evaluating **Multimodal Large Language Models (MLLMs)** on **Korean mathematical reasoning tasks**. Built upon [VLMEvalKit](https://github.com/open-compass/VLMEvalKit), it offers a rigorous evaluation framework to assess models' ability to understand and solve Korean mathematical problems.

## Dataset Composition

### 1. KMM-VisMath [![HuggingFace](https://img.shields.io/badge/🤗%20HuggingFace-Dataset-yellow)](https://huggingface.co/datasets/Tutoruslabs/KMMVisMath)
- **Composition**: Korean questions and step-by-step reasoning processes that require utilizing key information (prices, quantities, measurements, etc.) presented in images to solve mathematical problems.
- **Data Volume**: 167 complex reasoning samples, 1,888 simple reasoning samples
- **Features**: Complex reasoning problems that require accurate recognition of visual information in images and performing mathematical operations requested in the questions
- **Purpose**: A benchmark dataset for comprehensively evaluating vision-language models' visual information extraction capabilities, Korean language understanding, and mathematical reasoning abilities

### 2. Korean ChartQA [![HuggingFace](https://img.shields.io/badge/🤗%20HuggingFace-Dataset-yellow)](https://huggingface.co/datasets/Tutoruslabs/ChartQA_KOR)
- **Composition**: A multimodal dataset consisting of chart and graph images (bar, line, pie charts, etc.) with Korean questions and answers based on them.
- **Data Volume**: 2,000 question/answer pairs (1,000 chart images)
- **Features**: English chart images from the original ChartQA translated into Korean, with all text elements including labels, legends, axes, and titles provided in Korean
- **Purpose**: A dataset for evaluating vision-language models' Korean chart interpretation abilities, visual data analysis capabilities, and graph-based question-answering skills

### 3. Elementary Math Multimodal Data [![HuggingFace](https://img.shields.io/badge/🤗%20HuggingFace-Dataset-yellow)](https://huggingface.co/datasets/Tutoruslabs/ELEMENTARY_MATH)
- **Composition**: A benchmark dataset consisting of images requiring elementary-level mathematical thinking such as spatial perception, pattern recognition, and logical reasoning, along with Korean questions and answers
- **Data Volume**: 448 question/answer pairs
- **Features**: Problems requiring visual reasoning rather than simple calculations (e.g., inferring the number of blocks from multiple perspectives, shape rotation and transformation), evaluating image-based spatial cognition and logical thinking abilities
- **Purpose**: A benchmark dataset for comprehensively evaluating vision-language models' elementary-level mathematical thinking, visual reasoning abilities, and spatial perception capabilities

## Getting Started

> **Note**: This repository is built upon [VLMEvalKit](https://github.com/open-compass/VLMEvalKit.git). Please refer to the original repository for more detailed documentation.

### 1. Installation

#### PyTorch Installation

First, ensure PyTorch is installed. Refer to the [PyTorch official website](https://pytorch.org/get-started/locally/) for installation instructions suitable for your system.

```bash
# Example: Install PyTorch with CUDA support
pip install torch torchvision torchaudio
```

#### 2. Install Dependencies

Install required dependencies using pip:

```bash
pip install -e .
```

#### 3. Install Flash Attention

Install flash-attention for optimized inference:

```bash
pip install flash-attn --no-build-isolation
```

### 2. Dataset Preparation

#### 1. Request Dataset Access
- KMM-VisMath [![HuggingFace](https://img.shields.io/badge/🤗%20HuggingFace-Dataset-yellow)](https://huggingface.co/datasets/Tutoruslabs/KMMVisMath)
- ChartQA_KOR [![HuggingFace](https://img.shields.io/badge/🤗%20HuggingFace-Dataset-yellow)](https://huggingface.co/datasets/Tutoruslabs/ChartQA_KOR)
- ELEMENTARY_MATH [![HuggingFace](https://img.shields.io/badge/🤗%20HuggingFace-Dataset-yellow)](https://huggingface.co/datasets/Tutoruslabs/ELEMENTARY_MATH)

#### 2. Download Evaluation Datasets
Download VLMEvalKit evaluation datasets

```bash
sh prepare_dataset.sh
```

### 3. Inference and Evaluation

#### 1. Open-source Vision-Language Model Inference and Evaluation

```bash
export LMUData="./playground"
torchrun --nproc-per-node=2 run.py \
--data KMMVisMath \ # KMMVisMath, ELEMENTARY_MATH
--model <MODEL> \ # ex. Qwen2-VL-7B, InternVL3 (ref. ./vlmeval/config.py)
--mode all \
--verbose \
--work-dir ./outputs/chartqa_kor
```

#### 2. Proprietary Vision-Language Model Inference and Evaluation

- Configure environment variables (.env) at `$VLMEvalkit/.env`
```
# The .env file, place it under $VLMEvalKit
# API Keys of Proprietary VLMs
# QwenVL APIs
DASHSCOPE_API_KEY=
# Gemini w. Google Cloud Backends
GOOGLE_API_KEY=
# OpenAI API
OPENAI_API_KEY=
OPENAI_API_BASE=
# StepAI API
STEPAI_API_KEY=
# REKA API
REKA_API_KEY=
# GLMV API
GLMV_API_KEY=
# CongRong API
CW_API_BASE=
CW_API_KEY=
# SenseNova API
SENSENOVA_API_KEY=
# Hunyuan-Vision API
HUNYUAN_SECRET_KEY=
HUNYUAN_SECRET_ID=
# LMDeploy API
LMDEPLOY_API_BASE=
# You can also set a proxy for calling api models during the evaluation stage
EVAL_PROXY=
```

- Proprietary model evaluation script
```bash
export LMUData="./playground"
python run.py \
--data KMMVisMath \ # ChartQA_KOR, ELEMENTARY_MATH
--model <MODEL> \ # ex. GeminiFlash2-5 (ref. ./vlmeval/config.py)
--mode all \
--verbose \
--work-dir ./outputs/chartqa_kor
```

## Acknowledgements

본 데이터셋은 2023년도 정부(과학기술정보통신부)의 재원으로 정보통신기획평가원의 지원을 받아 수행된 연구임
**(과제번호: RS-2023-00216011, 사람처럼 개념적으로 이해/추론이 가능한 복합인공지능 원천기술 연구)**

This work was supported by Institute of Information & Communications Technology Planning & Evaluation (IITP) grant funded by the Korea government (MSIT)
**(RS-2023-00216011, Development of artificial complex intelligence for conceptually understanding and inferring like human)**

## Citation

- VLMEvalkit
```
@inproceedings{duan2024vlmevalkit,
  title={Vlmevalkit: An open-source toolkit for evaluating large multi-modality models},
  author={Duan, Haodong and Yang, Junming and Qiao, Yuxuan and Fang, Xinyu and Chen, Lin and Liu, Yuan and Dong, Xiaoyi and Zang, Yuhang and Zhang, Pan and Wang, Jiaqi and others},
  booktitle={Proceedings of the 32nd ACM International Conference on Multimedia},
  pages={11198--11201},
  year={2024}
}
```
