CUDA_VISIBLE_DEVICES=0 python scripts/mochi/inference.py \
    --model_name mochi \
    --pretrained_model_path genmo/mochi-1-preview \
    --text_prompt_file configs/prompts.txt \
    --output_dir results/baseline \
    --log_level info \
    --seed 42 \
    --enable_cpu_offload \
    --log_latency


CUDA_VISIBLE_DEVICES=0 python scripts/mochi/inference.py \
    --model_name mochi \
    --pretrained_model_path genmo/mochi-1-preview \
    --text_prompt_file configs/prompts.txt \
    --output_dir results/arnr-base \
    --log_level info \
    --seed 42 \
    --enable_cpu_offload \
    --enable_rnr \
    --schedule_file schedulers/euclidean_mochi_cache3.safetensors \
    --rnr_config_file configs/mochi_cache3_0.yaml


CUDA_VISIBLE_DEVICES=0 python scripts/mochi/inference.py \
    --model_name mochi \
    --pretrained_model_path genmo/mochi-1-preview \
    --text_prompt_file configs/prompts.txt \
    --output_dir results/arnr-fast \
    --log_level info \
    --seed 42 \
    --enable_cpu_offload \
    --enable_rnr \
    --schedule_file schedulers/euclidean_mochi_cache3.safetensors \
    --rnr_config_file configs/mochi_cache3_1.yaml