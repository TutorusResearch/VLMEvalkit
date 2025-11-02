torchrun --nproc-per-node=2 run.py \
    --data GSM8K_KOR \
    --model Gemma3-12B \
    --mode all \
    --work-dir /workspace/project/kor_math/github_product/VLMEvalKit/inference_results/chartqa \
    --verbose 

torchrun --nproc-per-node=2 run.py \
    --data ChartQA_KOR \
    --model Gemma3-12B \
    --mode all \
    --work-dir /workspace/project/kor_math/github_product/VLMEvalKit/inference_results/chartqa \
    --verbose 

torchrun --nproc-per-node=2 run.py \
    --data ELEMENTARY_MATH_KOR \
    --model Gemma3-12B \
    --mode all \
    --work-dir /workspace/project/kor_math/github_product/VLMEvalKit/inference_results/chartqa \
    --verbose 
