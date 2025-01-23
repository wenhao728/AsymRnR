#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''
@Created :   2024/10/09 20:58:56
@Desc    :   
@Ref     :   
'''
from typing import Dict

from diffusers import DiffusionPipeline

from .base_operators import update_processor_config
from .types import Tensor


def update_ops_timestep(
    pipeline: DiffusionPipeline, 
    timestep_index: int, 
    timestep: int, 
    callback_kwargs: Dict[str, Tensor],
) -> Dict[str, Tensor]:
    update_processor_config(pipeline.transformer, {'timestep': timestep_index + 1})
    # if timestep_index >= 2: exit(0)
    return {}