<h1 align="center">
  AsymRnR: Video Diffusion Transformers Acceleration with Asymmetric Reduction and Restoration
</h1>

> [!TIP]
> **TL;DR** A training-free method to accelerate video DiTs without compromising output quality.


<!-- ## 🎨 (WIP) Gallery -->


## 🔧 Setup
Install Pytorch, we have tested the code with PyTorch 2.5.0 and CUDA 12.4. But it should work with other versions as well. You can install PyTorch using the following command:
```
python -m pip install torch==2.5.0 torchvision==0.20.0 --index-url https://download.pytorch.org/whl/cu124
```

Install the dependencies:
```
python -m pip install -r requirements.txt
```

Install our optimized Euclidean distance operator for better performance:
```
python -m pip install .
```


## 🚀 Quick Start
We use the genaral scripts to demonstrate the usage of our method. You can find the detailed scripts for each model in the `scripts` folder:
- CogVideoX: [scripts/cogvideox/inference.sh](scripts/cogvideox/inference.sh)
- Mochi-1: [scripts/mochi/inference.sh](scripts/mochi/inference.sh)
- HunyuanVideo: [scripts/hyvideo/inference.sh](scripts/hyvideo/inference.sh)
- FastVideo-Hunyuan: [scripts/fast_hyvideo/inference.sh](scripts/fast_hyvideo/inference.sh)

Run the baseline model sampling without acceleration:
```bash
CUDA_VISIBLE_DEVICES=0 python scripts/<model_name>/inference.py \
    --model_name <model_name> \
    --pretrained_model_path <model_name_on_hf> \
    --text_prompt_file configs/prompts.txt \
    --output_dir results/baseline \
    --log_level info \
    --seed 42 \
    --log_latency
```
> - You can edit the `configs/prompts.txt` or the `--text_prompt_file` option to change the text prompt.
> - The `log_latency` option is enabled to log the latency of DiTs.
> - See the `scripts/<model_name>/inference.py` for more detailed explanations of the arguments.


Run the video DiTs with AsymRnR for acceleration:
```diff
CUDA_VISIBLE_DEVICES=0 python scripts/<model_name>/inference.py \
    --model_name <model_name> \
    --pretrained_model_path <model_name_on_hf> \
    --text_prompt_file configs/prompts.txt \
    --output_dir results/arnr-base \
    --log_level info \
    --seed 42 \
+    --enable_rnr \
+    --schedule_file schedulers/<scheduler>.safetensors \
+    --rnr_config_file configs/<reduction_configuration_file>.yaml \
    --log_latency
```
> - The `enable_rnr` option is used to enable AsymRnR.
> - The `schedule_file` option is used to specify the schedule file which saves the similarity distribution of the baseline model. See the Section 3.4 in the paper for more details.
> - The `rnr_config_file` option is used to scale the acceleration. See the `configs` folder and the Appendix in the paper for more details.


## 🐛 Common Issues

- **... square_dist is not available ...**
    - Please make sure you have installed the optimized Euclidean distance operator by running `python -m pip install .` in the root directory of the repository.

- **ImportError: libc10.so: cannot open shared object file: No such file or directory**
    - `libc10.so` is made available by PyTorch. Please `import torch` before `import square_dist`.

- **... libstdc++.so.6: version GLIBCXX_x.x.xx not found ..**
    - This error is due to the incompatibility of the GCC version. The simplest solution is to `libstdcxx-ng` by `conda install -c conda-forge libstdcxx-ng`.

## 🚧 Todo
- [x] *2025-01-26* Add more visualization results in the Supplementary Material
- [x] *2025-01-23* Code released


## :hearts: Shout-out
Thanks to the authors of the following repositories for their great works and open-sourcing the code and models: [Diffusers](https://github.com/huggingface/diffusers), [CogVideoX](https://github.com/THUDM/CogVideo), [Mochi-1](https://github.com/genmoai/mochi), [HunyuanVideo](https://github.com/Tencent/HunyuanVideo), [FastVideo](https://github.com/hao-ai-lab/FastVideo)