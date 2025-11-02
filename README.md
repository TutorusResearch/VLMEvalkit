# 🧮 멀티모달 한국어 수학 추론 벤치마크

한국어 | [English](/docs/en/README_en.md)

> **원본 VLMEvalKit 문서를 찾으시나요?** [VLMEvalKit 문서](/docs/en/KoreanMathBenchmark.md)를 참고하세요

[![][github-contributors-shield]][github-contributors-link] • [![][github-forks-shield]][github-forks-link] • [![][github-stars-shield]][github-stars-link] • [![][github-issues-shield]][github-issues-link] • [![][github-license-shield]][github-license-link]

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

## 감사의 말

이 벤치마크는 [VLMEvalKit](https://github.com/open-compass/VLMEvalKit)을 기반으로 구축되었습니다. 멀티모달 평가를 위한 훌륭한 기반을 제공해주신 VLMEvalKit 팀에 감사드립니다.

원본 VLMEvalKit 문서는 [VLMEvalKit 문서](/docs/en/KoreanMathBenchmark.md)에서 확인하실 수 있습니다.

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
