#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''
@Created :   2024/10/09 18:08:56
@Desc    :   
@Ref     :   
'''
import logging
from typing import Optional

import torch
import torch.nn.functional as F
from diffusers.models.attention_processor import Attention

from ..base_operators import OpsMixin

logger = logging.getLogger(__name__)


class CogVideoXRnRAttnProcessor2_0(OpsMixin):
    r"""
    Processor for implementing scaled dot-product attention (enabled by default if you're using PyTorch 2.0).
    """

    def __init__(self):
        super().__init__()

        if not hasattr(F, "scaled_dot_product_attention"):
            raise ImportError(f"{self.__class__.__name__} requires PyTorch 2, please upgrade PyTorch to 2.0")

    def __call__(
        self,
        attn: Attention,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        image_rotary_emb: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        text_seq_length = encoder_hidden_states.size(1)

        # logger.debug(f"Input H: {hidden_states.shape=}")
        qkv_input_fn, qkv_input_fn_inv = self.ops.get_qkv_input_fn(hidden_states)
        hidden_states = qkv_input_fn(hidden_states)
        # logger.debug(f"Reduced H: {hidden_states.shape=}")

        batch_size, sequence_length, _ = (
            hidden_states.shape if encoder_hidden_states is None else encoder_hidden_states.shape
        )

        if attention_mask is not None:
            attention_mask = attn.prepare_attention_mask(attention_mask, sequence_length, batch_size)
            attention_mask = attention_mask.view(batch_size, attn.heads, -1, attention_mask.shape[-1])

        query = attn.to_q(hidden_states)
        key = attn.to_k(hidden_states)
        value = attn.to_v(hidden_states)

        query_encoder = attn.to_q(encoder_hidden_states)
        key_encoder = attn.to_k(encoder_hidden_states)
        value_encoder = attn.to_v(encoder_hidden_states)

        inner_dim = key.shape[-1]
        head_dim = inner_dim // attn.heads

        query = query.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
        key = key.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
        value = value.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)

        query_encoder = query_encoder.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
        key_encoder = key_encoder.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
        value_encoder = value_encoder.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)

        if attn.norm_q is not None:
            # layer norm at the last dimension, which is essentially instance norm
            query = attn.norm_q(query)
            query_encoder = attn.norm_q(query_encoder)
        if attn.norm_k is not None:
            key = attn.norm_k(key)
            key_encoder = attn.norm_k(key_encoder)

        # Apply RoPE if needed
        if image_rotary_emb is not None:
            from diffusers.models.embeddings import apply_rotary_emb

            query = apply_rotary_emb(query, image_rotary_emb)
            if not attn.is_cross_attention:
                key = apply_rotary_emb(key, image_rotary_emb)

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

        hidden_states = F.scaled_dot_product_attention(
            query=torch.cat([query_encoder, query], dim=2),
            key=torch.cat([key_encoder, key], dim=2),
            value=torch.cat([value_encoder, value], dim=2),
            attn_mask=attention_mask, dropout_p=0.0, is_causal=False
        )

        # logger.debug(f"Q-Reduced attn_out: {hidden_states.shape=}")
        hidden_states = q_output_fn_inv(hidden_states, text_seq_length)
        # logger.debug(f"Q-Restored attn_out: {hidden_states.shape=}")

        hidden_states = hidden_states.transpose(1, 2).reshape(batch_size, -1, attn.heads * head_dim)

        # linear proj
        hidden_states = attn.to_out[0](hidden_states)
        # dropout
        hidden_states = attn.to_out[1](hidden_states)

        encoder_hidden_states, hidden_states = hidden_states.split(
            [text_seq_length, hidden_states.size(1) - text_seq_length], dim=1
        )

        # logger.debug(f"H-Reduced attn_out: {hidden_states.shape=}")
        hidden_states = qkv_input_fn_inv(hidden_states)
        # logger.debug(f"H-Restored attn_out: {hidden_states.shape=}")

        return hidden_states, encoder_hidden_states


AutoAttnProcessor = CogVideoXRnRAttnProcessor2_0