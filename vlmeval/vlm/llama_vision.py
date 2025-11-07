import torch
from PIL import Image
import os.path as osp
import sys
from textwrap import dedent
from .base import BaseModel
from ..smp import *
from ..dataset import DATASET_TYPE

def build_chartqa_kor_prompt(question):
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
    prompt = CHARTQA_KOR_PROMPT.format_map({'question': question})
    return prompt


def build_elementary_math_kor_prompt(question):
    ELEMENTARY_MATH_PROMPT = dedent("""
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
    prompt = ELEMENTARY_MATH_PROMPT.format_map({'question': question})
    return prompt

class llama_vision(BaseModel):

    INSTALL_REQ = False
    INTERLEAVE = True

    def __init__(self, model_path='meta-llama/Llama-3.2-11B-Vision-Instruct', **kwargs):
        try:
            from transformers import MllamaForConditionalGeneration, AutoProcessor
        except Exception as e:
            logging.critical('Please install transformers>=4.45.0 before using llama_vision.')
            raise e

        self.model = MllamaForConditionalGeneration.from_pretrained(
            model_path,
            torch_dtype=torch.bfloat16,
            device_map="auto",
        ).eval()

        self.device = 'cuda'
        self.processor = AutoProcessor.from_pretrained(model_path)
        if 'Instruct' in model_path or 'cot' in model_path or 'CoT' in model_path:
            kwargs_default = dict(do_sample=True, temperature=0.6, top_p=0.9)
        else:
            kwargs_default = dict(do_sample=False, max_new_tokens=2048, temperature=0.0, top_p=None, num_beams=1)
        kwargs.update(kwargs_default)
        print(f'Following kwargs received: {kwargs}, will use as generation config. ')
        self.kwargs = kwargs
        self.model_name = model_path

    def use_custom_prompt(self, dataset):
        if dataset is None:
            return False
        if listinstr(['AI2D', 'MMMU', 'MathVista', 'ChartQA', 'DocVQA','KMMVisMath', 'ChartQA_KOR','ELEMENTARY'], dataset):
            # For Certain dataset we use custom prompt
            return True
        else:
            return False

    def build_prompt(self, line, dataset=None):
        assert self.use_custom_prompt(dataset)
        assert dataset is None or isinstance(dataset, str)
        tgt_path = self.dump_image(line, dataset)
        question = line['question']
        options = {
            cand: line[cand]
            for cand in string.ascii_uppercase
            if cand in line and not pd.isna(line[cand])
        }
        if listinstr(['AI2D'], dataset):
            self.kwargs['max_new_tokens'] = 400
            for key, item in options.items():
                question += f'\n{key}. {item}'
            if '11B' in self.model_name:
                prompt = (
                    f'Look at the scientific diagram carefully and answer the following question: {question}\n'
                    f'Think step by step and finally respond to the question '
                    f"with only the correct option number as \"FINAL ANSWER\"."
                    f"<cot_start>Let's think step by step."
                )
            elif '90B' in self.model_name:
                prompt = (
                    f'Look at the scientific diagram carefully and answer the following question: {question}\n'
                    f'Respond only with the correct option digit.'
                )
        elif listinstr(['MMMU'], dataset):
            self.kwargs['max_new_tokens'] = 2048
            options = '\n'.join([f'{key}. {item}' for key, item in options.items()])
            prompt = (
                f'Look at the image carefully and solve the following question step-by-step. '
                f'Question: {question} Options: {options} Indicate the correct answer at the end.'
            )
            for i in range(len(tgt_path)):
                prompt = prompt.replace(f'<image {i + 1}>', '')
        elif listinstr(['MathVista'], dataset):
            self.kwargs['max_new_tokens'] = 2048
            prompt = f'{question}'
        elif listinstr(['ChartQA'], dataset) and not listinstr(['ChartQA_KOR'], dataset):
            self.kwargs['max_new_tokens'] = 512
            if '11B' in self.model_name:
                prompt = (
                    f'You are provided a chart image and will be asked a question. '
                    f'You have to think through your answer and provide a step-by-step solution. '
                    f'Once you have the solution, write the final answer in at most a few words at the end '
                    f"with the phrase \"FINAL ANSWER:\". "
                    f"The question is: {question}<cot_start>Let's think step by step."
                )
            elif '90B' in self.model_name:
                prompt = (
                    f'You are provided a chart image and will be asked a question. '
                    f'Follow these steps carefully:\n '
                    f'Step 1: Analyze the question to understand what specific data or information is being asked for. '
                    f'Focus on whether the question is asking for a specific number or category '
                    f'from the chart image.\n '
                    f'Step 2: Identify any numbers, categories, or groups mentioned in the question '
                    f'and take note of them. Focus on detecting and matching them directly to the image. \n'
                    f'Step 3: Study the image carefully and find the relevant data corresponding to the categories '
                    f'or numbers mentioned. Avoid unnecessary assumptions or calculations; '
                    f'simply read the correct data from the image.\n '
                    f'Step 4: Develop a clear plan to solve the question by locating the right data. '
                    f'Focus only on the specific category or group that matches the question. \n'
                    f'Step 5: Use step-by-step reasoning to ensure you are referencing the correct numbers '
                    f'or data points from the image, avoiding unnecessary extra steps or interpretations.\n '
                    f"Step 6: Provide the final answer, starting with \"FINAL ANSWER:\" "
                    f'and using as few words as possible, '
                    f'simply stating the number or data point requested. \n\n '
                    f"The question is: {question}<cot_start>Let's think step by step."
                )
        elif listinstr(['DocVQA'], dataset):
            self.kwargs['max_new_tokens'] = 512
            prompt = (
                f'Read the text in the image carefully and answer the question '
                f'with the text as seen exactly in the image. '
                f'For yes/no questions, just respond Yes or No. '
                f'If the answer is numeric, just respond with the number and nothing else. '
                f'If the answer has multiple words, just respond with the words and absolutely nothing else. '
                f'Never respond in a sentence or a phrase.\n Question: {question}'
            )
        elif listinstr(['KMMVisMath'], dataset):
            self.kwargs['max_new_tokens'] = 512
            prompt = question +'\n질문에 가능한 짧은 단어나 숫자 또는 구로 답하세요.'
        elif listinstr(['ChartQA_KOR'],dataset):
            prompt = build_chartqa_kor_prompt(question)
        elif listinstr(['ELEMENTARY_MATH_KOR'], dataset):
            prompt = build_elementary_math_kor_prompt(question)
        else:
            raise NotImplementedError(f'Dataset {dataset}) not supported.')

        message = [dict(type='text', value=prompt)]
        message.extend([dict(type='image', value=s) for s in tgt_path])
        return message

    def generate_inner(self, message, dataset=None):
        payload, images = [], []
        for msg in message:
            if msg['type'] == 'text':
                payload.append({'type': 'text', 'text': msg['value']})
            else:
                payload.append({'type': 'image'})
                images.append(Image.open(msg['value']))
        messages = [{'role': 'user', 'content': payload}]
        input_text = self.processor.apply_chat_template(messages, add_generation_prompt=True)
        inputs = self.processor(images, input_text, return_tensors='pt').to(self.device)
        if not self.use_custom_prompt(dataset):
            if dataset is not None and DATASET_TYPE(dataset) in ['MCQ', 'Y/N']:
                self.kwargs['max_new_tokens'] = 128
            else:
                self.kwargs['max_new_tokens'] = 512
        if "cot" in self.model_name or "CoT" in self.model_name:
            self.kwargs['max_new_tokens'] = 2048
        output = self.model.generate(**inputs, **self.kwargs)
        return self.processor.decode(output[0][inputs['input_ids'].shape[1]:]).replace('<|eot_id|>', '')
