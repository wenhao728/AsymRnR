#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''
@Created :   2024/09/19 18:55:04
@Desc    :   
@Ref     :   
'''
import logging
from typing import Optional, Tuple

import torch
import torch.nn.functional as F

from .types import Tensor

logger = logging.getLogger(__name__)


@torch.no_grad()
def split_spatiotemporal(
    input_shape: Tuple[int, int, int], 
    dst_stride: Tuple[int, int, int], 
    device: torch.device,
    generator: Optional[torch.Generator] = None,
) -> Tuple[Tensor, Tensor]:
    """Split the input sequence into spatiotemporal chunks and randomly select one token from each chunk as the 
    destination token. And the rest of the tokens are source tokens.

    Args:
        input_shape (Tuple[int, int, int]): The shape of the input (frame, height, width).
        dst_stride (Tuple[int, int, int]): The stride of the chunks along the frame, height, and width.
        device (torch.device): The device to place the indice tensors.
        generator (Optional[torch.Generator], optional): Random number generator for reproducibility. Defaults to None.

    Returns:
        Tuple[Tensor, Tensor]: The destination token indices and the source token indices. 
            Shape (num_dst_tokens,) and (num_src_tokens,), which are determined by the input_shape and dst_stride.
    """
    frame, height, width = input_shape
    chunk_size_f, chunk_size_h, chunk_size_w = dst_stride
    num_chunk_f, mod_chunk_f = divmod(frame, chunk_size_f)
    num_chunk_h, mod_chunk_h = divmod(height, chunk_size_h)
    num_chunk_w, mod_chunk_w = divmod(width, chunk_size_w)

    # evenly divide the input tensor into chunks, and randomly select one token (known as destination) from each chunk
    # the number of destination tokens is equal to the number of chunks
    num_dst_tokens = num_chunk_f * num_chunk_h * num_chunk_w

    # sample a matrix of shape (num_chunk_h, num_chunk_w), value range from 0 to (chunk_size_h * chunk_size_w - 1)
    # each value represents the index of the selected destination token of each chunk
    dst_idx = torch.randint(
        size=(num_chunk_f, num_chunk_h, num_chunk_w, 1), high=chunk_size_f * chunk_size_h * chunk_size_w, 
        device=device, generator=generator, dtype=torch.int64)

    # the destination indicator which has the same shape as the input tensor spatially, i.e. (height, width)
    # 1 indicates the token is the destination token, 0 otherwise
    dst_indicators = torch.zeros(
        *(num_chunk_f, num_chunk_h, num_chunk_w, chunk_size_f * chunk_size_h * chunk_size_w), 
        device=device, dtype=torch.int64)
    dst_indicators.scatter_(dim=-1, index=dst_idx, src=torch.ones_like(dst_indicators))
    # reshape the destination indicator to the shape of the input tensor
    dst_indicators = dst_indicators.reshape(
        num_chunk_f, num_chunk_h, num_chunk_w, chunk_size_f, chunk_size_h, chunk_size_w
    ).permute(0, 3, 1, 4, 2, 5).reshape(frame - mod_chunk_f, height - mod_chunk_h, width - mod_chunk_w)
    # potentially pad the destination indicator to the shape of the input tensor
    # logger.debug(f"Destination indicator (before padding):\n{dst_indicators}")

    if mod_chunk_f > 0 or mod_chunk_h > 0 or mod_chunk_w > 0:
        # randomly select the top/left padding size for height and width
        mod_chunk_f_left = mod_chunk_h_left = mod_chunk_w_left = 0
        if mod_chunk_f > 0:
            mod_chunk_f_left = torch.randint(size=(1,), high=mod_chunk_f, generator=generator, dtype=torch.int64)
        if mod_chunk_h > 0:
            mod_chunk_h_left = torch.randint(size=(1,), high=mod_chunk_h, generator=generator, dtype=torch.int64)
        if mod_chunk_w > 0:
            mod_chunk_w_left = torch.randint(size=(1,), high=mod_chunk_w, generator=generator, dtype=torch.int64)
        pad = (
            mod_chunk_w_left, mod_chunk_w - mod_chunk_w_left,
            mod_chunk_h_left, mod_chunk_h - mod_chunk_h_left, 
            mod_chunk_f_left, mod_chunk_f - mod_chunk_f_left,
        )
        dst_indicators = F.pad(dst_indicators, pad, mode='constant', value=0)
    # logger.debug(f"Destination indicator:\n{dst_indicators}")

    # get the destination token index when the input tensor is flattened into (batch_size * frame, height * width, ...)
    # argsort such that the destination token is at the beginning (with value 1) 
    # versus the non-destination token (with value 0)
    flatten_idx = dst_indicators.view(-1).argsort(descending=True)
    dst_idx = flatten_idx[:num_dst_tokens]
    src_idx = flatten_idx[num_dst_tokens:]
    # logger.debug(f"Destination index:\n{dst_idx}")
    # logger.debug(f"Source index:\n{src_idx}")

    return dst_idx, src_idx


@torch.no_grad()
def gather_features(
    hidden_states: Tensor,
    dst_idx: Tensor,
    src_idx: Tensor,
    skip_check: bool = False,
) -> Tuple[Tensor, Tensor]:
    """Compute the similarity matrix between the destination and source tokens.

    Args:
        hidden_states (Tensor): The feature tensor to compute the similarity matrix. Shape 
            (batch_size, num_tokens, channels) or (batch_size, num_heads, num_tokens, channels).
        dst_idx (Tensor): The destination token indices along the sequence dimension (axis=-2), shape (num_dst_tokens,)
        src_idx (Tensor): The source token indices along the sequence dimension (axis=-2), shape (num_src_tokens,)
        skip_check (bool, optional): Skip the check of the number of destination and source tokens. Defaults to False.

    Returns:
        Tuple[Tensor, Tensor, Tensor]: 
            Destination tokens of shape (batch_size, num_heads, num_dst_tokens, channels)
            Source tokens of shape (batch_size, num_heads, num_src_tokens, channels)
    """
    if len(hidden_states.shape) == 3:
        hidden_states = hidden_states.unsqueeze(1)
    batch_size, num_heads, num_tokens, channels = hidden_states.shape

    num_dst_tokens = dst_idx.size(0)
    num_src_tokens = src_idx.size(0)

    if num_dst_tokens + num_src_tokens != num_tokens:
        if not skip_check:
            raise ValueError(
                f'The number of destination tokens ({num_dst_tokens}) and source tokens ({num_src_tokens}) does not '
                f'match the total number of tokens ({num_tokens}). {hidden_states.shape=}')

    # logger.debug(f"dst_idx range: {dst_idx.min().item()} - {dst_idx.max().item()}")
    # logger.debug(f"src_idx range: {src_idx.min().item()} - {src_idx.max().item()}")
    src_idx = src_idx.to(hidden_states.device)[None, None, :, None]
    dst_idx = dst_idx.to(hidden_states.device)[None, None, :, None]

    dst_tokens = torch.gather(
        hidden_states, dim=-2, index=dst_idx.expand(batch_size, num_heads, num_dst_tokens, channels))
    src_tokens = torch.gather(
        hidden_states, dim=-2, index=src_idx.expand(batch_size, num_heads, num_src_tokens, channels))

    return dst_tokens, src_tokens