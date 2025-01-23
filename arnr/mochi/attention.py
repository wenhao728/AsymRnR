#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''
@Created :   2024/12/19 17:13:17
@Desc    :   
@Ref     :   
'''
import logging
from typing import Optional

import torch
import torch.nn.functional as F
from diffusers.models.attention_processor import Attention
from torch.nn.attention import SDPBackend, sdpa_kernel

from ..base_operators import OpsMixin

logger = logging.getLogger(__name__)


def apply_rotary_emb(x, freqs_cos, freqs_sin):
    x_even = x[..., 0::2].float()
    x_odd = x[..., 1::2].float()

    cos = (x_even * freqs_cos - x_odd * freqs_sin).to(x.dtype)
    sin = (x_even * freqs_sin + x_odd * freqs_cos).to(x.dtype)

    return torch.stack([cos, sin], dim=-1).flatten(-2)

class MochiRnRAttnProcessor2_0(OpsMixin):
    """Attention processor used in Mochi."""

    def __init__(self):
        if not hasattr(F, "scaled_dot_product_attention"):
            raise ImportError(
                "MochiRnRAttnProcessor2_0 requires PyTorch 2.0. To use it, please upgrade PyTorch to 2.0."
            )

    def __call__(
        self,
        attn: Attention,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        image_rotary_emb: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        # logger.debug(f"Input H: {hidden_states.shape=}")
        qkv_input_fn, qkv_input_fn_inv = self.ops.get_qkv_input_fn(hidden_states)
        hidden_states = qkv_input_fn(hidden_states)
        # logger.debug(f"Reduced H: {hidden_states.shape=}")

        query = attn.to_q(hidden_states)
        key = attn.to_k(hidden_states)
        value = attn.to_v(hidden_states)

        query = query.unflatten(2, (attn.heads, -1))
        key = key.unflatten(2, (attn.heads, -1))
        value = value.unflatten(2, (attn.heads, -1))

        if attn.norm_q is not None:
            query = attn.norm_q(query)
        if attn.norm_k is not None:
            key = attn.norm_k(key)

        encoder_query = attn.add_q_proj(encoder_hidden_states)
        encoder_key = attn.add_k_proj(encoder_hidden_states)
        encoder_value = attn.add_v_proj(encoder_hidden_states)

        encoder_query = encoder_query.unflatten(2, (attn.heads, -1))
        encoder_key = encoder_key.unflatten(2, (attn.heads, -1))
        encoder_value = encoder_value.unflatten(2, (attn.heads, -1))

        if attn.norm_added_q is not None:
            encoder_query = attn.norm_added_q(encoder_query)
        if attn.norm_added_k is not None:
            encoder_key = attn.norm_added_k(encoder_key)

        if image_rotary_emb is not None:
            query = apply_rotary_emb(query, *image_rotary_emb)
            key = apply_rotary_emb(key, *image_rotary_emb)

        query, key, value = query.transpose(1, 2), key.transpose(1, 2), value.transpose(1, 2)
        encoder_query, encoder_key, encoder_value = (
            encoder_query.transpose(1, 2),
            encoder_key.transpose(1, 2),
            encoder_value.transpose(1, 2),
        )

        # logger.debug(f"Query Q: {query.shape=}")
        q_output_fn, q_output_fn_inv = self.ops.get_q_output_fn(query)
        query = q_output_fn(query)
        # logger.debug(f"Reduced Q: {query.shape=}")

        # logger.debug(f"Key K: {key.shape=}")
        # logger.debug(f"Value V: {value.shape=}")
        kv_output_fn, _ = self.ops.get_kv_output_fn(key, value)
        key, value = kv_output_fn((key, value))
        # logger.debug(f"Reduced K: {key.shape=}")
        # logger.debug(f"Reduced V: {value.shape=}")

        sequence_length = query.size(2)
        encoder_sequence_length = encoder_query.size(2)
        total_length = sequence_length + encoder_sequence_length

        # issues shown in PR: https://github.com/huggingface/diffusers/pull/10033
        batch_size, heads, _, dim = query.shape
        attn_outputs = []
        for idx in range(batch_size):
            mask = attention_mask[idx][None, :]
            valid_prompt_token_indices = torch.nonzero(mask.flatten(), as_tuple=False).flatten()

            valid_encoder_query = encoder_query[idx : idx + 1, :, valid_prompt_token_indices, :]
            valid_encoder_key = encoder_key[idx : idx + 1, :, valid_prompt_token_indices, :]
            valid_encoder_value = encoder_value[idx : idx + 1, :, valid_prompt_token_indices, :]

            valid_query = torch.cat([query[idx : idx + 1], valid_encoder_query], dim=2)
            valid_key = torch.cat([key[idx : idx + 1], valid_encoder_key], dim=2)
            valid_value = torch.cat([value[idx : idx + 1], valid_encoder_value], dim=2)
            with sdpa_kernel(SDPBackend.EFFICIENT_ATTENTION):
                attn_output = F.scaled_dot_product_attention(
                    valid_query, valid_key, valid_value, dropout_p=0.0, is_causal=False
                )
            valid_sequence_length = attn_output.size(2)
            attn_output = F.pad(attn_output, (0, 0, 0, total_length - valid_sequence_length))
            attn_outputs.append(attn_output)

        hidden_states = torch.cat(attn_outputs, dim=0)

        # logger.debug(f"Q-Reduced attn_out: {hidden_states.shape=}")
        hidden_states = q_output_fn_inv(hidden_states, encoder_sequence_length)
        # logger.debug(f"Q-Restored attn_out: {hidden_states.shape=}")

        hidden_states = hidden_states.transpose(1, 2).flatten(2, 3)
        total_length = hidden_states.size(1)
        hidden_states, encoder_hidden_states = hidden_states.split_with_sizes(
            (total_length - encoder_sequence_length, encoder_sequence_length), dim=1
        )

        # linear proj
        hidden_states = attn.to_out[0](hidden_states)
        # dropout
        hidden_states = attn.to_out[1](hidden_states)

        # logger.debug(f"H-Reduced attn_out: {hidden_states.shape=}")
        hidden_states = qkv_input_fn_inv(hidden_states)
        # logger.debug(f"H-Restored attn_out: {hidden_states.shape=}")

        if getattr(attn, "to_add_out", None) is not None:
            encoder_hidden_states = attn.to_add_out(encoder_hidden_states)

        return hidden_states, encoder_hidden_states


AutoAttnProcessor = MochiRnRAttnProcessor2_0