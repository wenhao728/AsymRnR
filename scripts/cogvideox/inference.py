#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''
@Created :   2024/10/09 21:20:46
@Desc    :   
@Ref     :   
'''
import logging
import sys
from argparse import ArgumentParser, Namespace
from datetime import datetime
from pathlib import Path
from typing import List

import torch
from diffusers import CogVideoXDDIMScheduler, CogVideoXDPMScheduler
from omegaconf import OmegaConf
from PIL import Image
from tqdm.auto import tqdm

project_dir = str(Path(__file__).resolve().parents[2])
if project_dir not in sys.path:
    sys.path.append(project_dir)

from arnr.base_operators import update_processor_config
from arnr.callbacks import update_ops_timestep
from arnr.cogvideox import CogVideoXMeterPipeline, apply_rnr
from arnr.operators import RnROps
from arnr.utils import MetricMeter, arg_to_json, prompt_to_file_name, save_video, setup_logging

logger = logging.getLogger(__name__)


def parse_args() -> Namespace:
    parser = ArgumentParser()
    parser.add_argument("--log_latency", action="store_true", default=False, help="Enable benchmark mode")
    parser.add_argument("--log_flops", action="store_true", default=False, help="Enable benchmark mode")

    parser.add_argument("--model_name", type=str, choices=["cogvideox-2b", "cogvideox-5b"], required=True)
    parser.add_argument("--pretrained_model_path", type=str, required=True)
    parser.add_argument("--output_dir", type=Path, help="Path to log file", required=True)
    parser.add_argument("--log_level", type=lambda s: str(s).upper(), default="INFO")
    parser.add_argument("--text_prompt_file", type=Path, help="Path to text prompt file", required=True)
    parser.add_argument("--num_inference_steps", type=int, default=50)
    parser.add_argument("--cfg_scale", type=float, default=6.0)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--enable_cpu_offload", action="store_true")
    parser.add_argument("--video_size", type=int, nargs=3, default=[49, 480, 720])
    parser.add_argument("--output_fps", type=int, default=8)

    parser.add_argument("--enable_rnr", action="store_true", default=False, help="Enable")
    parser.add_argument(
        "--dst_stride", type=int, nargs=3, help="Stride for frame, height, and width", default=[2, 2, 2])
    parser.add_argument(
        "--similarity_type", type=str, default='euclidean', choices=['cosine', 'euclidean', 'dot', 'random'])
    parser.add_argument("--schedule_file", type=Path, default=None, help="Path to schedule file")
    parser.add_argument("--rnr_config_file", type=Path, default=None, help="Path to reduce config file")
    parser.add_argument("--matching_cache_steps", type=int, default=5, help="Number of steps to reuse similarity")
    parser.add_argument("--reduce_mode", type=str, default='replace', choices=['replace', 'mean'])

    args = parser.parse_args()
    args.output_dir = args.output_dir / args.model_name
    args.output_dir.mkdir(parents=True, exist_ok=True)

    # sanity check
    if args.enable_rnr:
        if args.schedule_file is None:
            raise ValueError("Please provide a schedule file")
        if args.rnr_config_file is None:
            raise ValueError("Please provide a reduce config file")

    return args


def load_model(args: Namespace, device, generator):
    logger.info(f"Loading {args.model_name} from {args.pretrained_model_path}")

    if args.model_name.startswith("cogvideox"):
        if args.model_name == "cogvideox-2b":
            torch_dtype = torch.float16
            scheduler_cls = CogVideoXDDIMScheduler
        elif args.model_name == "cogvideox-5b":
            torch_dtype = torch.bfloat16
            scheduler_cls = CogVideoXDPMScheduler
        else:
            raise ValueError(f"Invalid model name: {args.model_name}")

        pipeline: CogVideoXMeterPipeline = CogVideoXMeterPipeline.from_pretrained(
            args.pretrained_model_path, torch_dtype=torch_dtype)
        pipeline.scheduler = scheduler_cls.from_config(pipeline.scheduler.config, timestep_spacing="trailing")

    else:
        raise ValueError(f"Invalid model name: {args.model_name}")
    logger.info("Pipeline loaded")

    pipeline.vae.enable_tiling()

    if args.enable_rnr:
        # apply our attention processor
        num_frames, height, width = args.video_size

        rnr_config = OmegaConf.load(args.rnr_config_file)
        logger.info(rnr_config)

        apply_rnr(
            pipeline.transformer, 
            ops_cls=RnROps,
            input_shape=((num_frames - 1) // 4 + 1, height // 16, width // 16),
            dst_stride=args.dst_stride,
            similarity_type=args.similarity_type,
            schedule_file=args.schedule_file,
            num_inference_steps=args.num_inference_steps,
            similarity_reuse_steps=args.matching_cache_steps,
            rnr_config=rnr_config,
            reduce_mode=args.reduce_mode,
            device=device,
            # generator=generator,
        )
        logger.debug(f"{RnROps.__name__} applied to {pipeline.transformer.__class__.__name__}")

    return pipeline


def main():
    args = parse_args()
    setup_logging(
        log_level=args.log_level,
        output_file=(args.output_dir / datetime.now().strftime("%Y%m%d-%H%M%S")).with_suffix('.log'),
    )
    logger.info(arg_to_json(args))

    logger.info(f"Loading text prompts from {args.text_prompt_file}")
    with open(args.text_prompt_file, 'r', encoding="utf-8") as f:
        text_prompts = f.readlines()
    logger.info(f"Loaded {len(text_prompts)} text prompts")

    # Seed for RnR
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    generator = torch.Generator(device)
    if args.seed is not None:
        generator.manual_seed(args.seed)
    else:
        generator = None

    pipeline = load_model(args, device, generator)
    if args.enable_cpu_offload:
        pipeline.enable_model_cpu_offload()
        # pipeline.enable_sequential_cpu_offload()
        generator = torch.Generator("cpu")  # use CPU generator if CPU offload is enabled
    else:
        pipeline = pipeline.to(device)
    logger.info(f"Model loaded to {pipeline.device}")

    # calculate inference time
    time_meter = MetricMeter(name="Latency", mode="avg", fmt=":.2f")
    flops_meter = MetricMeter(name="GFLOPs", mode="avg", fmt=":.6e")

    # inference loop, keep `batch_size=1` to avoid OOM
    for prompt_idx, text_prompt in enumerate(tqdm(text_prompts)):
        file_name = (args.output_dir / prompt_to_file_name(text_prompt, prompt_idx, 0)).with_suffix('.mp4')
        if file_name.exists(): continue  # skip if the file already exists
        
        if args.seed is not None:
            torch.manual_seed(args.seed)
            generator.manual_seed(args.seed)

        # reset `timestep` index
        if args.enable_rnr:
            update_processor_config(pipeline.transformer, {'timestep': 0})

        # inference
        with torch.no_grad():
            outputs = pipeline(
                prompt=text_prompt,
                num_frames=args.video_size[0],
                height=args.video_size[1],
                width=args.video_size[2],
                num_inference_steps=args.num_inference_steps,
                generator=generator,
                guidance_scale=args.cfg_scale,
                callback_on_step_end=update_ops_timestep if args.enable_rnr else None,
                # callback to update `timestep` after each inference step
                output_type='pil',
                log_flops=args.log_flops,
                log_latency=args.log_latency,
            )

        videos: List[List[Image.Image]] = outputs.frames

        time_elapsed, gflpos = 0.0, 0.0
        if args.log_latency:
            time_elapsed = outputs.time_elapsed
            if prompt_idx > 0:  # skip the first video
                time_meter.update(time_elapsed)

        if args.log_flops:
            gflpos = outputs.gflops
            flops_meter.update(gflpos)
            break  # count 1st video only
        logger.info(f"Video {prompt_idx}: {time_elapsed:.2f} s | {gflpos:.6e} G")

        # save video
        for video_idx, video in enumerate(videos):
            file_name = (args.output_dir / prompt_to_file_name(text_prompt, prompt_idx, video_idx)).with_suffix('.mp4')
            save_video(file_name, video, fps=args.output_fps)
            logger.info(f"Video saved: {file_name}")

    output_str = f"Finished {len(text_prompts)} videos."
    if args.log_latency:
        output_str += f"\n\t| Avg Time:  {time_meter.summary:.2f} (s)"
    if args.log_flops:
        output_str += f"\n\t| Avg FLOPs: {flops_meter.summary:.6e} (G)"
    logger.info(output_str)
    logger.info("Everything done!")


if __name__ == "__main__":
    main()