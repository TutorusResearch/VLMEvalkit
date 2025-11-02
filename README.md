# 🧮 멀티모달 한국어 수학 추론 벤치마크

한국어 | [English](/docs/en/README_en.md)

## 개요

이 저장소는 **한국어 수학 추론 작업**에 대한 **멀티모달 대규모 언어 모델(MLLMs)**을 평가하기 위한 종합적인 벤치마크를 제공합니다. [VLMEvalKit](https://github.com/open-compass/VLMEvalKit)을 기반으로 구축되었으며, 한국어 수학 문제를 이해하고 해결하는 모델의 능력을 평가하기 위한 엄격한 평가 프레임워크를 제공합니다.

## 주요 특징

- **한국어 중심**: 한국어 수학 추론에 특화된 평가
- **멀티모달 평가**: 텍스트 기반 및 시각적 수학 문제 모두 테스트
- **포괄적인 범위**: 다양한 수학 주제와 난이도 포함
- **표준화된 평가**: 다양한 모델에 대한 일관된 평가 프로토콜
- **VLMEvalKit 기반**: VLMEvalKit의 강력한 인프라 활용

## 시작하기

> **참고**: 이 저장소는 [VLMEvalKit](https://github.com/open-compass/VLMEvalKit.git)을 기반으로 구축되었습니다. 더 자세한 문서는 원본 저장소를 참고하세요.

### 설치

#### 1. PyTorch 설치

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

이 명령어는 `requirements.txt`에 나열된 모든 의존성을 설치합니다.

#### 3. Flash Attention 설치

최적화된 추론을 위해 flash-attention을 설치합니다:

```bash
pip install flash-attn --no-build-isolation
```
