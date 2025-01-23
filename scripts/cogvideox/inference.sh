CUDA_VISIBLE_DEVICES=0 python scripts/cogvideox/inference.py \
    --model_name cogvideox-2b \
    --pretrained_model_path THUDM/CogVideoX-2b \
    --text_prompt_file configs/prompts.txt \
    --output_dir results/baseline \
    --log_level info \
    --seed 42 \
    --log_latency


CUDA_VISIBLE_DEVICES=0 python scripts/cogvideox/inference.py \
    --model_name cogvideox-2b \
    --pretrained_model_path THUDM/CogVideoX-2b \
    --text_prompt_file configs/prompts.txt \
    --output_dir results/arnr-base \
    --log_level info \
    --seed 42 \
    --log_latency \
    --enable_rnr \
    --schedule_file schedulers/euclidean_cogvideox-2b_cache5.safetensors \
    --rnr_config_file configs/cogvideox-2b_cache5_0.yaml


CUDA_VISIBLE_DEVICES=0 python scripts/cogvideox/inference.py \
    --model_name cogvideox-2b \
    --pretrained_model_path THUDM/CogVideoX-2b \
    --text_prompt_file configs/prompts.txt \
    --output_dir results/arnr-fast \
    --log_level info \
    --seed 42 \
    --log_latency \
    --enable_rnr \
    --schedule_file schedulers/euclidean_cogvideox-2b_cache5.safetensors \
    --rnr_config_file configs/cogvideox-2b_cache5_1.yaml