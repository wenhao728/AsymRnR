#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''
@Created :   2024/12/17 15:18:46
@Desc    :   The modified forward function for MMDoubleStreamBlock and MMSingleStreamBlock
@Ref     :   
'''
import logging
from typing import Optional

import torch
import torch.nn.functional as F
from diffusers.models.attention_processor import Attention

from ..base_operators import OpsMixin

logger = logging.getLogger(__name__)


class HunyuanVideoRnRAttnProcessor2_0(OpsMixin):
    def __init__(self):
        if not hasattr(F, "scaled_dot_product_attention"):
            raise ImportError(
                "HunyuanVideoRnRAttnProcessor2_0 requires PyTorch 2.0. To use it, please upgrade PyTorch to 2.0."
            )

    def __call__(
        self,
        attn: Attention,
        hidden_states: torch.Tensor,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        image_rotary_emb: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        text_seq_length = encoder_hidden_states.size(1)
        # logger.debug(f"Input H: {hidden_states.shape=}")
        qkv_input_fn, qkv_input_fn_inv = self.ops.get_qkv_input_fn(hidden_states)
        hidden_states = qkv_input_fn(hidden_states)
        # logger.debug(f"Reduced H: {hidden_states.shape=}")

        # 1. QKV projections
        query = attn.to_q(hidden_states)
        key = attn.to_k(hidden_states)
        value = attn.to_v(hidden_states)

        query = query.unflatten(2, (attn.heads, -1)).transpose(1, 2)
        key = key.unflatten(2, (attn.heads, -1)).transpose(1, 2)
        value = value.unflatten(2, (attn.heads, -1)).transpose(1, 2)

        # 2. QK normalization
        if attn.norm_q is not None:
            query = attn.norm_q(query)
        if attn.norm_k is not None:
            key = attn.norm_k(key)

        # 3. Rotational positional embeddings applied to latent stream
        if image_rotary_emb is not None:
            from diffusers.models.embeddings import apply_rotary_emb
            query = apply_rotary_emb(query, image_rotary_emb)
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

        # 4. Encoder condition QKV projection and normalization
        is_single_stream = attn.add_q_proj is None

        if encoder_hidden_states is not None:
            if is_single_stream:
                encoder_query = attn.to_q(encoder_hidden_states)
                encoder_key = attn.to_k(encoder_hidden_states)
                encoder_value = attn.to_v(encoder_hidden_states)
            else:
                encoder_query = attn.add_q_proj(encoder_hidden_states)
                encoder_key = attn.add_k_proj(encoder_hidden_states)
                encoder_value = attn.add_v_proj(encoder_hidden_states)

            encoder_query = encoder_query.unflatten(2, (attn.heads, -1)).transpose(1, 2)
            encoder_key = encoder_key.unflatten(2, (attn.heads, -1)).transpose(1, 2)
            encoder_value = encoder_value.unflatten(2, (attn.heads, -1)).transpose(1, 2)

            if is_single_stream:
                if attn.norm_q is not None:
                    encoder_query = attn.norm_q(encoder_query)
                if attn.norm_k is not None:
                    encoder_key = attn.norm_k(encoder_key)
            else:
                if attn.norm_added_q is not None:
                    encoder_query = attn.norm_added_q(encoder_query)
                if attn.norm_added_k is not None:
                    encoder_key = attn.norm_added_k(encoder_key)

            query = torch.cat([query, encoder_query], dim=2)
            key = torch.cat([key, encoder_key], dim=2)
            value = torch.cat([value, encoder_value], dim=2)

        # modify the attention mask to consider the change in the sequence length
        # attention_mask (B, N + N_text, N + N_text), the right-bottom part is the mask for text <PAD>, i.e. False
        mask_start = self.ops.mask_start_index
        # logger.debug(f"{mask_start=}")

        # 5. Attention
        hidden_states = F.scaled_dot_product_attention(
            query, key, value, dropout_p=0.0, is_causal=False, 
            attn_mask=attention_mask[:, mask_start[0] :, mask_start[1] :],
        )

        # logger.debug(f"Q-Reduced attn_out: {hidden_states.shape=}")
        hidden_states = q_output_fn_inv(hidden_states, text_seq_length)
        # logger.debug(f"Q-Restored attn_out: {hidden_states.shape=}")

        hidden_states = hidden_states.transpose(1, 2).flatten(2, 3)
        hidden_states = hidden_states.to(query.dtype)

        # 6. Output projection
        if encoder_hidden_states is not None:
            hidden_states, encoder_hidden_states = (
                hidden_states[:, : -text_seq_length],
                hidden_states[:, -text_seq_length :],
            )

            if getattr(attn, "to_out", None) is not None:
                hidden_states = attn.to_out[0](hidden_states)
                hidden_states = attn.to_out[1](hidden_states)

            if getattr(attn, "to_add_out", None) is not None:
                encoder_hidden_states = attn.to_add_out(encoder_hidden_states)

        # logger.debug(f"H-Reduced attn_out: {hidden_states.shape=}")
        hidden_states = qkv_input_fn_inv(hidden_states)
        # logger.debug(f"H-Restored attn_out: {hidden_states.shape=}")

        return hidden_states, encoder_hidden_states


AutoAttnProcessor = HunyuanVideoRnRAttnProcessor2_0