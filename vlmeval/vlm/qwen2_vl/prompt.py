from __future__ import annotations
from textwrap import dedent


class Qwen2VLPromptMixin:
    """
    Mixin class for Qwen2VLChat to build custom prompt for different datasets.

    Requires the following methods to be implemented in the subclass:
        - dump_image(line, dataset: str) -> str | list[str]

    Implements the following methods:
        - use_custom_prompt(dataset: str) -> bool
        - build_prompt(line, dataset: str) -> list[dict[str, str]]
    """

    def __init__(self, *args, use_custom_prompt: bool = True, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self._use_custom_prompt = use_custom_prompt

    def set_dump_image(self, dump_image_func):
        self.dump_image_func = dump_image_func

    def dump_image(self, line, dataset):
        return self.dump_image_func(line)

    def use_custom_prompt(self, dataset: str) -> bool:
        from vlmeval.dataset import DATASET_TYPE
        dataset_type = DATASET_TYPE(dataset, default=None)

        if not self._use_custom_prompt:
            return False
        if dataset in {'MMMU_DEV_VAL', 'MMMU_TEST'}:
            return True
        if dataset_type == 'MCQ':
            if dataset is not None and 'LEGO' in dataset:
                return False
            return True
        if dataset_type == 'Y/N' and dataset in {'HallusionBench', 'POPE'}:  # MME has it's own prompt
            return True
        if dataset_type == 'VQA' and dataset not in {'MMVet', 'ChartQAPro', 'ChartQAPro_CoT', 'ChartQAPro_PoT', 'ChartMuseum'}:  # noqa: E501
            return True
        return False

    def build_prompt(self, line, dataset: str) -> list[dict[str, str]]:
        from vlmeval.dataset import DATASET_TYPE

        if dataset in {'MMMU_DEV_VAL', 'MMMU_TEST'}:
            return self._build_mmmu_prompt(line, dataset) 
        if dataset in {'KMMVisMath'}:
            return self._build_kmmvismath_kor_prompt(line, dataset)
        if dataset in {'ChartQA_KOR'}:
            return self._build_chartqa_kor_prompt(line, dataset)
        if dataset in {'ELEMENTARY_MATH_KOR'}:
            return self._build_elementary_math_kor_prompt(line, dataset)
        dataset_type = DATASET_TYPE(dataset, default=None)
        if dataset_type == 'MCQ':
            return self._build_mcq_prompt(line, dataset)
        if dataset_type == 'Y/N':
            return self._build_yorn_prompt(line, dataset)
        if dataset_type == 'VQA':
            return self._build_vqa_prompt(line, dataset)
        raise ValueError(f'Unsupported dataset: {dataset}')

    def _build_mmmu_prompt(self, line, dataset: str) -> list[dict[str, str]]:
        """change the prompt for MMMU dataset: keep all images at beginning."""

        import string

        import pandas as pd

        tgt_path = self.dump_image(line, dataset)
        question = line['question']
        options = {cand: line[cand] for cand in string.ascii_uppercase if cand in line and not pd.isna(line[cand])}
        options_prompt = 'Options:\n'
        for key, item in options.items():
            options_prompt += f'{key}. {item}\n'
        hint = line['hint'] if ('hint' in line and not pd.isna(line['hint'])) else None
        prompt = ''
        if hint is not None:
            prompt += f'Hint: {hint}\n'
        prompt += f'Question: {question}\n'
        if len(options):
            prompt += options_prompt
            prompt += 'Please select the correct answer from the options above. \n'
        prompt = prompt.rstrip()
        msgs = []
        if isinstance(tgt_path, list):
            msgs.extend([dict(type='image', value=p) for p in tgt_path])
        else:
            msgs = [dict(type='image', value=tgt_path)]
        msgs.append(dict(type='text', value=prompt))
        return msgs

    def _build_mcq_prompt(self, line, dataset: str) -> list[dict[str, str]]:
        """change the prompt for MCQ dataset: use chinese prompt if the question contains chinese characters."""
        MCQ_CN_PROMPT = '请直接回答选项字母。'
        MCQ_EN_PROMPT = 'Please select the correct answer from the options above.'

        import string

        import pandas as pd

        def cn_string(s):
            import re

            if re.search('[\u4e00-\u9fff]', s):
                return True
            return False

        tgt_path = self.dump_image(line, dataset)
        question = line['question']
        options = {cand: line[cand] for cand in string.ascii_uppercase if cand in line and not pd.isna(line[cand])}
        options_prompt = 'Options:\n'
        for key, item in options.items():
            options_prompt += f'{key}. {item}\n'
        hint = line['hint'] if ('hint' in line and not pd.isna(line['hint'])) else None
        prompt = ''
        if hint is not None:
            prompt += f'Hint: {hint}\n'
        prompt += f'Question: {question}\n'
        if len(options):
            prompt += options_prompt
            prompt += MCQ_CN_PROMPT if cn_string(prompt) else MCQ_EN_PROMPT
        prompt = prompt.rstrip()
        msgs = []
        if isinstance(tgt_path, list):
            msgs.extend([dict(type='image', value=p) for p in tgt_path])
        else:
            msgs = [dict(type='image', value=tgt_path)]
        msgs.append(dict(type='text', value=prompt))
        return msgs

    def _build_yorn_prompt(self, line, dataset: str) -> list[dict[str, str]]:
        """change the prompt for YORN dataset:"""
        YORN_PROMPT = ' Please answer yes or no.'

        tgt_path = self.dump_image(line, dataset)
        question = line['question']
        msgs = []
        if isinstance(tgt_path, list):
            msgs.extend([dict(type='image', value=p) for p in tgt_path])
        else:
            msgs = [dict(type='image', value=tgt_path)]
        msgs.append(dict(type='text', value=question))
        assert msgs[-1]['type'] == 'text'
        msgs[-1]['value'] += YORN_PROMPT
        return msgs

    def _build_vqa_prompt(self, line, dataset: str) -> list[dict[str, str]]:
        """change the prompt for VQA dataset:"""
        VQA_PROMPT = '\nPlease try to answer the question with short words or phrases if possible.'

        tgt_path = self.dump_image(line, dataset)
        question = line['question']
        msgs = []
        if isinstance(tgt_path, list):
            msgs.extend([dict(type='image', value=p) for p in tgt_path])
        else:
            msgs = [dict(type='image', value=tgt_path)]
        msgs.append(dict(type='text', value=question))
        assert msgs[-1]['type'] == 'text'
        msgs[-1]['value'] += VQA_PROMPT
        return msgs

    def _build_chartqa_kor_prompt(self, line, dataset: str) -> list[dict[str, str]]:
        """change the prompt for VQA dataset:"""
        CHARTQA_KOR_PROMPT = dedent("""
            차트 이미지를 분석하여 다음 질문에 답변하세요.

            **답변 규칙**:

            1. **형식**: 짧은 단어나 구로만 답변
               - 추가 설명 없이 답만 제공
               - 불필요한 공백이나 특수문자 제외

            2. **숫자 답변**:
               - 질문에 "A:B, A/B로 계산"이 포함된 경우:
                 * 소수점 둘째자리까지만 표기하고 셋째자리 이하는 버림 처리
                 * 예: 29/24 = 1.208... → 1.20
                 * 예: 11/11 = 1.0 → 1 (정수인 경우 정수로 표기)
               - 일반 숫자는 차트에 표시된 그대로 답변
               - 단위를 포함하지 않음

            3. **예/아니오 답변**:
               - "네" 또는 "아니오"로만 답변

            4. **텍스트 답변** (색상, 국가명, 항목명, 시간 표현 등):
               - 차트에 표시된 정확한 명칭을 그대로 사용
               - 간결하게 핵심만 표현

            **주의사항**:
            - 차트의 축, 범례, 레이블을 정확히 확인
            - 계산이 필요한 경우 정확하게 계산
            - 추론이 아닌 차트에 표시된 데이터를 기반으로 답변

            질문: {question}
        """).strip()
        tgt_path = self.dump_image(line, dataset)
        question = line['question']
        msgs = []
        if isinstance(tgt_path, list):
            msgs.extend([dict(type='image', value=p) for p in tgt_path])
        else:
            msgs = [dict(type='image', value=tgt_path)]
        msgs.append(dict(type='text', value=question))
        assert msgs[-1]['type'] == 'text'
        msgs[-1]['value'] = CHARTQA_KOR_PROMPT.format_map({'question':msgs[-1]['value']})
        print(f"build_chartqa_kor_prompt: {msgs[-1]['value']}")
        return msgs
    

    def _build_kmmvismath_kor_prompt(self,line, dataset:str) -> list[dict[str, str]]:
        KMM_VisMath_PROMPT="\n질문에 가능한 짧은 단어나 숫자 또는 구로 답하세요."
        tgt_path = self.dump_image(line, dataset)
        question = line['question'] + KMM_VisMath_PROMPT # djshin 2025-11-07
        print(f"build_kmmvismath_kor_prompt: {question}")
        msgs = []
        if isinstance(tgt_path, list):
            msgs.extend([dict(type='image', value=p) for p in tgt_path])
        else:
            msgs = [dict(type='image', value=tgt_path)]
        msgs.append(dict(type='text', value=question))
        assert msgs[-1]['type'] == 'text'
        
        return msgs
    
    def _build_elementary_math_kor_prompt(self, line, dataset: str) -> list[dict[str, str]]:
        """change the prompt for VQA dataset:"""
        ELEMENTARY_MATH_KOR_PROMPT = dedent("""
            다음은 한국 초등학교 수학 문제입니다. 이미지를 보고 문제를 풀어주세요.

            ## 답변 형식 지침
            1. **숫자 답변**: 반드시 단위를 포함하여 표기 (예: "24cm", "3개", "15°")
            2. **선택형 답변**: 보기 기호를 그대로 사용 (예: "(가)", "ㄱ, ㄷ", "2")
            3. **문자/텍스트 답변**: 정확한 용어 사용 (예: "삼각형", "월요일", "빨간색")
            4. **복합 답변**: 문제에서 요구하는 형식 그대로 작성 (예: "가: 5cm, 나: 60°")
            5. **수식 표현**: LaTeX 형식이 필요한 경우 정확히 표기 (예: "$3.14$", "$\\frac{{1}}{{2}}$", "$200.96\\: \\mathrm{{cm^2}}$")

            ## 주의사항
            - 이미지를 꼼꼼히 관찰하고 문제에서 요구하는 정보를 정확히 파악하세요
            - 단위, 기호, 괄호 등을 빠뜨리지 말고 정확히 포함하세요
            - 여러 답이 필요한 경우 문제가 요구하는 순서와 구분자를 따르세요
            - 계산이 필요한 경우 정확히 수행하세요
            - **문제를 다시 쓰지 말고, "최종답변: " 형식으로만 답변하세요**
            - **풀이 과정, 설명, 문제 재서술 등 일체 금지**

            ---

            문제: {question}

            최종답변:
        """).strip()
        tgt_path = self.dump_image(line, dataset)
        question = line['question']
        msgs = []
        if isinstance(tgt_path, list):
            msgs.extend([dict(type='image', value=p) for p in tgt_path])
        else:
            msgs = [dict(type='image', value=tgt_path)]
        msgs.append(dict(type='text', value=question))
        assert msgs[-1]['type'] == 'text'
        msgs[-1]['value'] = ELEMENTARY_MATH_KOR_PROMPT.format_map({'question':msgs[-1]['value']})
        print(f"build_elementary_math_kor_prompt: {msgs[-1]['value']}")
        return msgs