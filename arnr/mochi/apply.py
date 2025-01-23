#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''
@Created :   2024/12/20 15:25:54
@Desc    :   
@Ref     :   
'''
import logging
import os
from copy import deepcopy
from pathlib import Path
from typing import Any, Dict, Optional, Tuple, Type

import torch
from diffusers.models.transformers.transformer_mochi import (
    MochiTransformer3DModel,
    MochiTransformerBlock,
)
from safetensors import safe_open

from ..base_operators import BaseOps
from ..reduce import calculate_num_unreduce_src
from ..split import split_spatiotemporal
from ..types import ReduceMode, SimilarityType, Tensor
from .attention import AutoAttnProcessor

logger = logging.getLogger(__name__)


def apply_eda(
    transformer: MochiTransformer3DModel,
    ops_cls: Type[BaseOps],
    input_shape: Tuple[int, int, int],
    dst_stride: Tuple[int, int, int],
    similarity_type: SimilarityType,
    data_file: os.PathLike,
) -> None:
    attention_block_idx = 0
    for block in transformer.transformer_blocks:
        if isinstance(block, MochiTransformerBlock):
            attn_config = dict(
                input_shape=input_shape,
                dst_stride=dst_stride,
                similarity_type=similarity_type,
                data_file=data_file,
                attention_block_idx=attention_block_idx,
                timestep=0,
            )

            # Create new processor to comaptible with additional ops
            attention_processor = AutoAttnProcessor()
            # Set additional ops for attention
            attention_processor.update_ops(ops_cls(attn_config))
            # Update attention processor
            block.attn1.processor = attention_processor
            attention_block_idx += 1


def apply_rnr(
    transformer: MochiTransformer3DModel,
    ops_cls: Type[BaseOps],
    input_shape: Tuple[int, int, int],
    dst_stride: Tuple[int, int, int],
    similarity_type: SimilarityType,
    schedule_file: os.PathLike,
    num_inference_steps: int = 50,
    similarity_reuse_steps: int = 5,
    rnr_config: Dict[str, Dict[str, float]] = {
        'h': {0: 0.0},
        'q': {0: 0.0},
        'v': {0: 0.0},
    },
    reduce_mode: ReduceMode = 'replace',
    device: Optional[torch.device] = None,
    generator: Optional[torch.Generator] = None,
) -> None:
    total_num_blocks = len(transformer.transformer_blocks)
    redundancy_schedule = _load_redundancy_schedule(
        schedule_file, total_num_blocks, num_inference_steps, similarity_type)

    attention_block_idx = 0
    for block in transformer.transformer_blocks:
        if isinstance(block, MochiTransformerBlock):
            # The configuration of additional operations for attention
            attn_config = dict(
                similarity_type=similarity_type,
                similarity_reuse_steps=similarity_reuse_steps,
                reduce_mode=reduce_mode,
                timestep_config=_get_timestep_config(
                    attention_block_idx,
                    num_inference_steps,
                    input_shape,
                    dst_stride,
                    similarity_reuse_steps,
                    redundancy_schedule,
                    rnr_config,
                    device=device,
                    generator=generator,
                ),
                timestep=0,
                encoder_first=False,
            )
            # Create new processor to comaptible with additional ops
            attention_processor = AutoAttnProcessor()
            # Set additional ops for attention
            attention_processor.update_ops(ops_cls(attn_config))
            # Update attention processor
            block.attn1.processor = attention_processor

            attention_block_idx += 1


def _get_timestep_config(
    block_idx: int,
    num_inference_steps: int,
    input_shape: Tuple[int, int, int],
    dst_stride: Tuple[int, int, int],
    similarity_reuse_steps: int,
    redundancy_schedule: Dict[str, torch.Tensor],
    rnr_config: Dict[str, Dict[str, float]],
    device: Optional[torch.device] = None,
    generator: Optional[torch.Generator] = None,
) -> Dict[str, Dict[int, Dict[str, Any]]]:
    device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
    if isinstance(device, str):
        device = torch.device(device)

    cached_info = {
        feat: {
            'current_reuse': 0,
            'dst_idx': None,
            'src_idx': None,
        }
        for feat in ['h', 'q', 'v', 'k']
    }
    timesteps_config = {}
    for feat in ['h', 'q', 'v', 'k']:
        if feat not in rnr_config:
            continue # Do not apply RnR for this feature
        
        feat_config = {}
        for t in range(num_inference_steps):
            t_config = dict(enable=False)  # default disable

            # update the cached info
            if cached_info[feat]['current_reuse'] >= similarity_reuse_steps:
                cached_info[feat]['dst_idx'], cached_info[feat]['src_idx'] = None, None
                cached_info[feat]['current_reuse'] = 0
            cached_info[feat]['current_reuse'] += 1
            use_cached = cached_info[feat]['dst_idx'] is not None and cached_info[feat]['src_idx'] is not None

            # get the current redundancy tier and RnR ratio
            tier = redundancy_schedule[feat][block_idx, t]  # current tier
            rnr_ratio = 0.0
            for threshold, rnr_ratio_ in rnr_config[feat].items():
                # find the highest rnr ratio that satisfies the threshold
                if tier >= threshold: 
                    rnr_ratio = max(rnr_ratio, rnr_ratio_)

            # check if the feature is conflict with other features
            # h is conflict with q, k, v
            # v is conflict with k
            h_disabled = ('h' not in timesteps_config) or (timesteps_config['h'][t]['enable'] is False)
            v_disabled = ('v' not in timesteps_config) or (timesteps_config['v'][t]['enable'] is False)
            is_conflict = not (
                feat == 'h' or 
                (feat == 'q' and h_disabled) or
                (feat == 'v' and h_disabled) or
                (feat == 'k' and h_disabled and v_disabled)
            )
            if is_conflict:
                logger.debug(f"b={block_idx:>2}, t={t:>2}, skip {feat} ({h_disabled=} {v_disabled=})")
                continue
            
            # BSM w/ Matching Cache
            if 0.0 < rnr_ratio < 1.0:  # valid prune ratio
                if use_cached:
                    dst_idx, src_idx = cached_info[feat]['dst_idx'], cached_info[feat]['src_idx']
                else:
                    dst_idx, src_idx = split_spatiotemporal(
                        input_shape, dst_stride, device=device, generator=generator)
                    cached_info[feat]['dst_idx'], cached_info[feat]['src_idx'] = dst_idx, src_idx

                num_unreduce_src = calculate_num_unreduce_src(dst_idx.size(0), src_idx.size(0), rnr_ratio)
                num_reduced_src = src_idx.size(0) - num_unreduce_src
                t_config.update({
                    'enable': True,
                    'dst_idx': dst_idx,
                    'src_idx': src_idx,
                    'num_unreduce_src': num_unreduce_src,
                    'num_reduced_src': num_reduced_src,
                    'use_cached': use_cached,
                })
                logger.debug(f"b={block_idx:>2}, t={t:>2}, {feat=}, {rnr_ratio=:.2f}, {use_cached=}")

            feat_config[t] = deepcopy(t_config)
        timesteps_config[feat] = deepcopy(feat_config)

    return timesteps_config


def _load_redundancy_schedule(
    schedule_file: os.PathLike,
    total_num_blocks: int,
    num_inference_steps: int,
    similarity_type: SimilarityType,
) -> Dict[str, torch.Tensor]:
    schedule_file = Path(schedule_file)
    # # Check if the redundancy file name matches the inference steps and similarity type
    # if str(num_inference_steps) not in schedule_file.stem or similarity_type not in schedule_file.stem:
    #     raise ValueError(
    #         f"Schedule file name {schedule_file.stem} does not match the inference steps {num_inference_steps} "
    #         f"or similarity type {similarity_type}")

    # Load redundancy data
    redundancy_schedule: Dict[str, Tensor] = {}
    with safe_open(schedule_file, framework="pt", device="cpu") as f:
        for key in f.keys():
            redundancy_schedule[key] = f.get_tensor(key)  # (num_blocks, num_timesteps)

    # Sanity check
    expected_shape = (total_num_blocks, num_inference_steps)
    for feat, redundancy_tensor in redundancy_schedule.items():
        if redundancy_tensor.shape != expected_shape:
            raise ValueError(f"Schedule of feature {feat} {redundancy_tensor.shape} does not match {expected_shape=}")

    return redundancy_schedule