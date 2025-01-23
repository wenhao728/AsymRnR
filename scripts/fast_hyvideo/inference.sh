CUDA_VISIBLE_DEVICES=0 python scripts/fast_hyvideo/inference.py \
    --model_name fast-hyvideo \
    --pretrained_model_path FastVideo/FastHunyuan-diffusers \
    --text_prompt_file configs/prompts.txt \
    --output_dir results/baseline \
    --log_level info \
    --seed 42 \
    --enable_cpu_offload \
    --log_latency


CUDA_VISIBLE_DEVICES=0 python scripts/fast_hyvideo/inference.py \
    --model_name fast-hyvideo \
    --pretrained_model_path FastVideo/FastHunyuan-diffusers \
    --text_prompt_file configs/prompts.txt \
    --output_dir results/arnr-base \
    --log_level info \
    --seed 42 \
    --enable_cpu_offload \
    --enable_rnr \
    --schedule_file schedulers/euclidean_fast-hyvideo_cache1.safetensors \
    --rnr_config_file configs/fast-hyvideo_cache1_0.yaml \
    --log_latency


CUDA_VISIBLE_DEVICES=0 python scripts/fast_hyvideo/inference.py \
    --model_name fast-hyvideo \
    --pretrained_model_path FastVideo/FastHunyuan-diffusers \
    --text_prompt_file configs/prompts.txt \
    --output_dir results/arnr-fast \
    --log_level info \
    --seed 42 \
    --enable_cpu_offload \
    --enable_rnr \
    --schedule_file schedulers/euclidean_fast-hyvideo_cache1.safetensors \
    --rnr_config_file configs/fast-hyvideo_cache1_1.yaml \
    --log_latency