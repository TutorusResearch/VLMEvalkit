# 🧮 멀티모달 한국어 수학 추론 벤치마크

한국어 | [English](/docs/en/README_en.md)

## 개요

이 저장소는 **한국어 수학 추론 작업**에 대한 **멀티모달 대규모 언어 모델(MLLMs)**을 평가하기 위한 종합적인 벤치마크를 제공합니다. [VLMEvalKit](https://github.com/open-compass/VLMEvalKit)을 기반으로 구축되었으며, 한국어 수학 문제를 이해하고 해결하는 모델의 능력을 평가하기 위한 엄격한 평가 프레임워크를 제공합니다.

## 데이터셋 구성

### 1. KMM-VisMath [[huggingface](https://huggingface.co/datasets/Tutoruslabs/KMMVisMath)]
- **구성**: 수학 문제 해결에 필요한 핵심 정보(가격, 수량, 측정값 등)가 이미지로 제시되고, 이를 활용하여 풀어야 하는 한국어 질의문 및 단계별 추론 과정으로 구성.
- **데이터 수량**: 복합추론 데이터 167개, 단순추론 데이터 1,888개
- **특징**: 이미지 내 시각정보를 정확히 인식하고 질의문에서 요구하는 수학적 연산을 수행해야 답변 가능한 복합 추론 문제
- **목적**: 시각-언어 모델의 시각 정보 추출 능력, 한국어 이해 능력, 수학적 추론 능력을 통합적으로 평가하기 위한 벤치마크 데이터셋

### 2. 한국어 ChartQA [[huggingface](https://huggingface.co/datasets/Tutoruslabs/ChartQA_KOR)]
- **구성**: 차트 및 그래프 이미지(막대, 선, 파이 차트 등)와 이를 기반으로 한 한국어 질의문 및 답변으로 구성된 멀티모달 데이터셋. 
- **데이터 수량**: 질의/답변 데이터 2,000개 (차트 이미지 1,000개)
- **특징**: 원본 ChartQA의 영어 차트 이미지를 한국어로 번역하여, 차트 내 레이블, 범례, 축, 제목 등 모든 텍스트 요소가 한국어로 제공됨
- **목적**: 시각-언어 모델의 한국어 차트 해석 능력, 시각 데이터 분석 능력, 그래프 기반 질의응답 능력을 평가하기 위한 데이터셋

### 3. 초등 수학 멀티모달 데이터 [[huggingface](https://huggingface.co/datasets/Tutoruslabs/ELEMENTARY_MATH)]
- **구성**: 공간 지각, 패턴 인식, 논리적 사고 등 초등학생 수준의 수학적 사고력을 요구하는 이미지와 한국어 질의문 및 답변으로 구성된 벤치마크 데이터셋
- **데이터 수량**: 질의/답변 데이터 448개 
- **특징**: 단순 계산이 아닌 시각적 추론이 필요한 문제(ex. 여러 방향에서 본 블록 개수 추론, 도형 회전 및 변환 등)로, 이미지 기반 공간 인지 및 논리적 사고 능력 평가
- **목적**: 시각-언어 모델의 초등 수준 수학적 사고력, 시각적 추론 능력, 공간 지각 능력을 종합적으로 평가하기 위한 벤치마크 데이터셋

## 시작하기

> **참고**: 이 저장소는 [VLMEvalKit](https://github.com/open-compass/VLMEvalKit.git)을 기반으로 구축되었습니다. 더 자세한 문서는 원본 저장소를 참고하세요.

### 1. 설치

#### PyTorch 설치

먼저 PyTorch가 설치되어 있는지 확인하세요. 시스템에 맞는 설치 방법은 [PyTorch 공식 웹사이트](https://pytorch.org/get-started/locally/)를 참고하세요.

```bash
# 예시: CUDA 지원과 함께 PyTorch 설치
pip install torch torchvision torchaudio
```

#### 2. 의존성 설치

pip를 사용하여 필요한 의존성을 설치합니다:

```bash
pip install -e .
```

#### 3. Flash Attention 설치

최적화된 추론을 위해 flash-attention을 설치합니다:

```bash
pip install flash-attn --no-build-isolation
```

### 2. 데이터셋 준비

#### 1. 데이터셋 권한 요청
- KMM-VisMath [[huggingface](https://huggingface.co/datasets/Tutoruslabs/KMMVisMath)]
- ChartQA_KOR [[huggingface](https://huggingface.co/datasets/Tutoruslabs/ChartQA_KOR)]
- ELEMENTARY_MATH [[huggingface](https://huggingface.co/datasets/Tutoruslabs/ELEMENTARY_MATH)]

#### 2. 평가용 데이터셋 다운로드
VLMEvalKit 평가용 데이터셋 다운로드

```bash
sh prepare_dataset.sh
```

### 3. 추론 및 평가

#### 1. 오픈소스 시각-언어 모델 추론 및 평가

```bash
export LMUData="./playground"
torchrun --nproc-per-node=2 run.py \
--data KMMVisMath \ # KMMVisMath, ELEMENTARY_MATH
--model <MODEL> \ # ex. Qwen2-VL-7B, InternVL3 (ref. ./vlmeval/config.py)
--mode all \ 
--verbose \
--work-dir ./outputs/chartqa_kor
```

#### 2. 상용 시각-언어 모델 추론 및 평가

- 환경변수(.env) 설정 `$VLMEvalkit/.env`
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

- 상용모델 평가 스크립트
```bash
export LMUData="./playground"
python run.py \
--data KMMVisMath \ # ChartQA_KOR, ELEMENTARY_MATH
--model <MODEL> \ # ex. GeminiFlash2-5 (ref. ./vlmeval/config.py)
--mode all \ 
--verbose \
--work-dir ./outputs/chartqa_kor
```
