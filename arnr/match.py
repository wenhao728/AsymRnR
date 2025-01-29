#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''
@Created :   2024/10/05 16:56:12
@Desc    :   
@Ref     :   
    https://github.com/choderalab/modelforge/issues/143 -> torch.gather has a simialr performance as indexing
'''
import logging

import torch
import torch.nn.functional as F

from .types import SimilarityType, MatchingOutput, Tensor

logger = logging.getLogger(__name__)

try:
    from square_dist import square_dist
except ImportError:
    square_dist = None
    warning_message = (
        "square_dist is not available. Please compile the extension following README.md/#Setup. Otherwise, the pairwise"
        " squared distance will be computed using torch.cdist, which includes redundant computation and may be slower.")
    print(warning_message)
    logger.warning(warning_message)
    # Comment out the following line at your own risk
    raise ImportError(warning_message)


@torch.no_grad()
def compute_simialrity(
    dst_features: Tensor, 
    src_features: Tensor,
    similarity_type: SimilarityType = 'euclidean',
) -> Tensor:
    """Compute the similarity between destination and source features.

    Args:
        dst_features (Tensor): (batch_size, num_heads, num_dst_tokens, channels)
        src_features (Tensor): (batch_size, num_heads, num_src_tokens, channels)
        similarity_type (SimilarityType, optional): The type of similarity to compute. Defaults to 'euclidean'.

    Returns:
        Tensor: (batch_size, num_heads, num_dst_tokens, num_src_tokens)
    """
    if len(dst_features.shape) == 3:
        dst_features = dst_features.unsqueeze(1)
        src_features = src_features.unsqueeze(1)

    batch_size, num_heads, num_dst_tokens, channels = dst_features.shape

    if similarity_type == 'dot':
        similarity = torch.einsum('bhdc,bhkc->bhdk', dst_features, src_features)
    elif similarity_type == 'cosine':
        similarity = torch.einsum(
            'bhdc,bhkc->bhdk',
            F.normalize(dst_features, p=2, dim=-1),
            F.normalize(src_features, p=2, dim=-1),
        )
    elif similarity_type == 'euclidean':
        if square_dist is not None:
            similarity = square_dist(
                dst_features.reshape(batch_size * num_heads, num_dst_tokens, channels),
                src_features.reshape(batch_size * num_heads, -1, channels),
            ).reshape(batch_size, num_heads, num_dst_tokens, -1)
        else:
            similarity = torch.cdist(
                dst_features.reshape(batch_size * num_heads, num_dst_tokens, channels),
                src_features.reshape(batch_size * num_heads, -1, channels),
                p=2.0,
            ).reshape(batch_size, num_heads, num_dst_tokens, -1)
    elif similarity_type == 'random':
        similarity = None
    else:
        raise NotImplementedError(f"{similarity_type=} is not supported.")

    return similarity


@torch.no_grad()
def match_features(
    dst_features: Tensor,
    src_features: Tensor,
    similarity_type: SimilarityType = 'euclidean',
) -> MatchingOutput:
    """Match destination and source features.

    Args:
        dst_features (Tensor): Destination features. (batch_size, num_heads, num_dst_tokens, channels)
        src_features (Tensor): Source features. (batch_size, num_heads, num_src_tokens, channels)
        similarity_type (SimilarityType, optional): The type of similarity to compute. Defaults to 'euclidean'.

    Returns:
        MatchingOutput: Matching results.
    """
    similarity = compute_simialrity(dst_features, src_features, similarity_type)

    if similarity_type in ['dot', 'cosine']:
        sim_value, sim_dst_idx = similarity.max(dim=-2)  # (batch_size, num_heads, num_src_tokens)
        edge_idx = sim_value.argsort(dim=-1, descending=False)
    elif similarity_type == 'euclidean':
        sim_value, sim_dst_idx = similarity.min(dim=-2)
        edge_idx = sim_value.argsort(dim=-1, descending=True)
    elif similarity_type == 'random':
        if len(dst_features.shape) == 3:
            batch_size, num_dst_tokens, channels = dst_features.shape
            num_heads = 1
        else:
            batch_size, num_heads, num_dst_tokens, channels = dst_features.shape
        num_src_tokens = src_features.shape[-2]

        sim_value = torch.randn(batch_size, num_heads, num_src_tokens, device=dst_features.device)
        sim_dst_idx = torch.randint(
            0, num_dst_tokens, size=(batch_size, num_heads, num_src_tokens), device=dst_features.device)
        edge_idx = sim_value.argsort(dim=-1, descending=True)
    else:
        raise NotImplementedError(f"{similarity_type=} is not supported.")

    return MatchingOutput(
        sim_value=sim_value,
        sim_dst_idx=sim_dst_idx,
        edge_idx=edge_idx,
    )