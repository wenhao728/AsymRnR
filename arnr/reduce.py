#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''
@Created :   2024/10/05 15:27:01
@Desc    :   
@Ref     :   
'''
import logging

import torch

from .types import ReduceOutput, ReduceMode, Tensor

logger = logging.getLogger(__name__)


def calculate_num_unreduce_src(
    num_dst_features: int,
    num_src_features: int,
    reduce_ratio: float,
) -> int:
    """Calculate the number of reduced tokens based on the destination and source token indices.

    Args:
        num_dst_features (int): Number of destination tokens.
        num_src_features (int): Number of source tokens.
        reduce_ratio (float): Total number of tokens reduced divided by the number of tokens before reducing.
    
    Returns:
        int: Number of source tokens after reducing.
    """
    num_unreduce_src_features = int((num_dst_features + num_src_features) * (1 - reduce_ratio)) - num_dst_features
    return max(0, num_unreduce_src_features)  # Ensure non-negative value


def reduce_sequence(
    dst_features: Tensor,
    src_features: Tensor,
    edge_idx: Tensor,
    sim_dst_idx: Tensor,
    num_unreduce_src: int,
    reduce_mode: ReduceMode,
    return_indices: bool = False,
) -> ReduceOutput:
    """reduce the source tokens into the destination tokens based on the similarity matrix.

    Args:
        dst_features (Tensor): Destination tokens. Shape (batch_size, num_heads, num_dst_features, channels).
        src_features (Tensor): Source tokens. Shape (batch_size, num_heads, num_src_features, channels).
        edge_idx (Tensor): The indices of the matching edges, sorted by distance, in descending order. 
            Shape (batch_size, num_heads, num_src_features).
        sim_dst_idx (Tensor): The indices of the matched destination tokens.
            Shape (batch_size, num_heads, num_src_features).
        num_unreduce_src (int): Number of source tokens to be left unreduced.
        reduce_mode (str): Reduce mode for the matched source-destination pairs. Options are 'mean' and 'replace'.
        return_indices (bool, optional): Whether to return the indices of the reduced tokens. Defaults to False.

    Returns:
        ReduceOutput: Reduced tokens and the corresponding indices.
    """
    batch_size, num_heads, num_src_features, channels = src_features.shape

    # The least similar soruce tokens (vs destination tokens) are left unreduced
    # num_unreduce_src_features = _num_unreduce_src_features(dst_features.size(-2), src_features.size(-2), reduce_ratio)
    # logger.debug(f"{dst_features.shape=} {src_features.shape=}")
    # logger.debug(f"{reduce_ratio=} {num_unreduce_src_features=}")

    # get the least similar source tokens
    unreduce_src_idx = edge_idx[..., :num_unreduce_src, None]
    unreduce_src_features = torch.gather(
        src_features, dim=-2, index=unreduce_src_idx.expand(batch_size, num_heads, num_unreduce_src, channels))
    # logger.debug(f"{unreduce_src_idx=}")
    # logger.debug(f"{unreduce_src_features.shape=}")

    if reduce_mode == 'mean' or return_indices:
        # get the most similar source tokens
        reduce_src_idx = edge_idx[..., num_unreduce_src:, None]

        # get the most similar destination token indices in the similiarity matrix
        reduce_sim_dst_idx = torch.gather(sim_dst_idx[..., None], dim=-2, index=reduce_src_idx)

    if reduce_mode == 'mean':
        # get the most similar source tokens
        reduce_src_features = torch.gather(
            src_features, dim=-2, 
            index=reduce_src_idx.expand(batch_size, num_heads, num_src_features - num_unreduce_src, channels))

        # reduce witht the most similar destination tokens
        dst_features = dst_features.scatter_reduce(
            dim=-2, 
            index=reduce_sim_dst_idx.expand(batch_size, num_heads, num_src_features - num_unreduce_src, channels),
            src=reduce_src_features, 
            reduce=reduce_mode,
        )
    elif reduce_mode == 'replace':
        # directly use the destination token
        pass
    else:
        raise ValueError(f"Invalid reduce mode: {reduce_mode}")
    
    if return_indices:
        return ReduceOutput(
            reduced_tokens=torch.cat([unreduce_src_features, dst_features], dim=-2),
            reduce_sim_dst_idx=reduce_sim_dst_idx,
            unreduce_src_idx=unreduce_src_idx,
            reduce_src_idx=reduce_src_idx,
        )
    else:
        return ReduceOutput(reduced_tokens=torch.cat([unreduce_src_features, dst_features], dim=-2))


def restore_sequence(
    reduced_tokens: Tensor,
    reduce_sim_dst_idx: Tensor,
    unreduce_src_idx: Tensor,
    reduce_src_idx: Tensor,
    dst_idx: Tensor,
    src_idx: Tensor,
    num_unreduce_src: int,
) -> Tensor:
    """Unreduce the reduced tokens into the source and destination tokens.

    Args:
        reduced_tokens (Tensor): reduced tokens. Shape (batch_size, num_heads, num_tokens * reduce_ratio, channels).
        The following 3 indice tensors are the col and row index in the similarity matrix
        *Note*: they are not the index in the input tensor before splitting.
            reduce_sim_dst_idx (Tensor): The mapping between reduced source tokens and destination tokens indices.
                Shape (batch_size, num_heads, num_src_features - num_unreduce_src_features, 1).
            unreduce_src_idx (Tensor): The indices of the unreduced source tokens in the similarity matrix.
                Shape (batch_size, num_heads, num_unreduce_src_features, 1).
            reduce_src_idx (Tensor): The indices of the reduced source tokens in the similarity matrix.
                Shape (batch_size, num_heads, num_src_features - num_unreduce_src_features, 1).

        The following 2 tensors are the index in the input tensor before splitting.
            dst_idx (Tensor): Destination token indices. Shape (num_dst_features,).
            src_idx (Tensor): Source token indices. Shape (num_src_features,).

        num_unreduce_src (int): Number of source tokens to be left unreduced.

    Returns:
        Tensor: Unreduced hidden states. Shape (batch_size, num_heads, num_tokens, channels).
    """
    num_dst_features, num_src_features = dst_idx.size(0), src_idx.size(0)
    # logger.debug(f"Unreduce {dst_idx.shape=} {src_idx.shape=}")
    # num_unreduce_src_features = _num_unreduce_src_features(num_dst_features, num_src_features, reduce_ratio)
    # logger.debug(f"{num_dst_features=}, {num_src_features=}, {num_unreduce_src_features=}")

    # unreduced source and destination tokens
    unreduce_src_features = reduced_tokens[..., :num_unreduce_src, :]
    dst_features = reduced_tokens[..., num_unreduce_src:, :]
    # logger.debug(f"{reduced_tokens.shape=} {unreduce_src_features.shape=}, {dst_features.shape=}")

    batch_size, num_heads, num_dst_features_, channels = dst_features.shape
    if num_dst_features_ != num_dst_features:
        raise ValueError(f"Expected {num_dst_features=}, but got {num_dst_features_=}")

    # reduced source tokens
    reduce_src_features = torch.gather(
        dst_features, dim=-2, 
        index=reduce_sim_dst_idx.expand(batch_size, num_heads, num_src_features - num_unreduce_src, channels))

    # initalizae the output tensor
    num_tokens = num_dst_features + num_src_features
    hidden_states = torch.zeros(
        batch_size, num_heads, num_tokens, channels, device=dst_features.device, dtype=dst_features.dtype)

    # destination tokens
    dst_idx = dst_idx.to(hidden_states.device)[..., None]
    hidden_states.scatter_(
        dim=-2, index=dst_idx.expand(batch_size, num_heads, num_dst_features, channels), src=dst_features)
    
    # unreduced source tokens
    src_idx = src_idx.to(hidden_states.device)[..., None].expand(batch_size, num_heads, num_src_features, 1)
    # similarity matrix col index -> the index in the input tensor
    unreduce_src_idx = torch.gather(src_idx, dim=-2, index=unreduce_src_idx)
    # scatter the unreduced source tokens to the output tensor
    hidden_states.scatter_(
        dim=-2, index=unreduce_src_idx.expand(batch_size, num_heads, num_unreduce_src, channels), 
        src=unreduce_src_features)
    
    # reduced source tokens
    # similarity matrix col index -> the index in the input tensor
    reduce_src_idx = torch.gather(src_idx, dim=-2, index=reduce_src_idx)
    # scatter the unreduced source tokens to the output tensor
    hidden_states.scatter_(
        dim=-2, index=reduce_src_idx.expand(batch_size, num_heads, num_src_features - num_unreduce_src, channels), 
        src=reduce_src_features)

    return hidden_states