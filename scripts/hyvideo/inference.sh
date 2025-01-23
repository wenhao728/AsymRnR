CUDA_VISIBLE_DEVICES=0 python scripts/hyvideo/inference.py \
    --model_name hyvideo \
    --pretrained_model_path tencent/HunyuanVideo \
    --revision refs/pr/18 \
    --text_prompt_file configs/prompts.txt \
    --output_dir results/baseline \
    --log_level info \
    --seed 42 \
    --enable_cpu_offload \
    --log_latency


CUDA_VISIBLE_DEVICES=0 python scripts/hyvideo/inference.py \
    --model_name hyvideo \
    --pretrained_model_path tencent/HunyuanVideo \
    --revision refs/pr/18 \
    --text_prompt_file configs/prompts.txt \
    --output_dir results/arnr-base \
    --log_level info \
    --seed 42 \
    --enable_cpu_offload \
    --enable_rnr \
    --schedule_file schedulers/euclidean_hyvideo_cache3.safetensors \
    --rnr_config_file configs/hyvideo_cache3_0.yaml \
    --log_latency


CUDA_VISIBLE_DEVICES=0 python scripts/hyvideo/inference.py \
    --model_name hyvideo \
    --pretrained_model_path tencent/HunyuanVideo \
    --revision refs/pr/18 \
    --text_prompt_file configs/prompts.txt \
    --output_dir results/arnr-fast \
    --log_level info \
    --seed 42 \
    --enable_cpu_offload \
    --enable_rnr \
    --schedule_file schedulers/euclidean_hyvideo_cache3.safetensors \
    --rnr_config_file configs/hyvideo_cache3_1.yaml \
    --log_latency