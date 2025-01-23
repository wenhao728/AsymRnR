import logging
from typing import Any, Dict, Literal, Optional, Tuple

import torch

from .base_operators import BaseOps, do_nothing
from .match import compute_simialrity, match_features
from .reduce import reduce_sequence, restore_sequence
from .split import gather_features, split_spatiotemporal
from .types import Tensor, TensorFn

logger = logging.getLogger(__name__)


class RnROps(BaseOps):
    _required_config_keys = [
        'similarity_type', 'similarity_reuse_steps', 'reduce_mode', 'timestep_config', 'timestep', 'encoder_first']

    def __init__(self, config: Optional[Dict[str, Any]] = None) -> None:
        super().__init__(config)

        self.cache = {
            feat: {
                'current_reuse': 0,
                'edge_idx': None,
                'sim_dst_idx': None,
            } 
            for feat in ['h', 'q', 'v']
        }
        self._num_reduce = {feat: 0 for feat in ['h', 'q', 'v']}

    @property
    def mask_start_index(self) -> Tuple[int, int]:
        # only reduce h or q, v; 0 if not reduce; return the max of them
        # used for HunyuanVideoRnRAttnProcessor2_0 maskings
        return max(self._num_reduce['h'], self._num_reduce['q']), max(self._num_reduce['h'], self._num_reduce['v'])

    @torch.no_grad()
    def _shared_steps(self, hidden_states: Tensor, feature_name: Literal['h', 'q', 'v']) -> Tuple[Tensor]:
        timestep_config = self.config['timestep_config']
        timestep = self.config['timestep']

        if self.cache[feature_name]['current_reuse'] >= self.config['similarity_reuse_steps']:
            self.cache[feature_name]['edge_idx'] = None
            self.cache[feature_name]['sim_dst_idx'] = None
            self.cache[feature_name]['current_reuse'] = 0
        self.cache[feature_name]['current_reuse'] += 1

        if (
            (feature_name not in timestep_config) or 
            (timestep not in timestep_config[feature_name]) or 
            (timestep_config[feature_name][timestep]['enable'] is False)
        ):
            # no reduction
            self._num_reduce[feature_name] = 0
            return None

        # update num_reduced tokens
        self._num_reduce[feature_name] = timestep_config[feature_name][timestep]['num_reduced_src']

        dst_idx = timestep_config[feature_name][timestep]['dst_idx']
        src_idx = timestep_config[feature_name][timestep]['src_idx']
        # logger.debug(f"{feature_name} {dst_idx.shape=} {src_idx.shape=}")

        dst_tokens, src_tokens = gather_features(hidden_states, dst_idx, src_idx)

        if self.cache[feature_name]['edge_idx'] is not None and self.cache[feature_name]['sim_dst_idx'] is not None:
            edge_idx = self.cache[feature_name]['edge_idx']
            sim_dst_idx = self.cache[feature_name]['sim_dst_idx']
        else:
            sim_ouputs = match_features(dst_tokens, src_tokens, similarity_type=self.config['similarity_type'])
            edge_idx = sim_ouputs.edge_idx
            sim_dst_idx = sim_ouputs.sim_dst_idx
            # update cache
            self.cache[feature_name]['edge_idx'] = sim_ouputs.edge_idx
            self.cache[feature_name]['sim_dst_idx'] = sim_ouputs.sim_dst_idx

        # logger.debug(f"{feature_name} {dst_tokens.shape=} {src_tokens.shape=}")
        num_unreduce_src = timestep_config[feature_name][timestep]['num_unreduce_src']
        return dst_idx, src_idx, dst_tokens, src_tokens, edge_idx, sim_dst_idx, num_unreduce_src

    @torch.no_grad()
    def get_qkv_input_fn(self, hidden_states: Tensor) -> Tuple[TensorFn, TensorFn]:
        shared_outputs = self._shared_steps(hidden_states, 'h')
        if shared_outputs is None:
            return do_nothing, do_nothing
        dst_idx, src_idx, dst_tokens, src_tokens, edge_idx, sim_dst_idx, num_unreduce_src = shared_outputs

        reduce_outputs = reduce_sequence(
            dst_tokens, src_tokens, edge_idx, sim_dst_idx, num_unreduce_src, self.config['reduce_mode'], 
            return_indices=True)
        # logger.debug(f"{feature_name} {reduce_outputs.reduced_tokens.shape=}")

        reduce_func = lambda x: reduce_outputs.reduced_tokens.squeeze(1)
        restore_func = lambda x: restore_sequence(
            reduced_tokens=x.unsqueeze(1),
            reduce_sim_dst_idx=reduce_outputs.reduce_sim_dst_idx, 
            unreduce_src_idx=reduce_outputs.unreduce_src_idx, 
            reduce_src_idx=reduce_outputs.reduce_src_idx, 
            dst_idx=dst_idx, 
            src_idx=src_idx, 
            num_unreduce_src=num_unreduce_src, 
        ).squeeze(1)

        return reduce_func, restore_func

    @torch.no_grad()
    def get_q_output_fn(self, query: Tensor) -> Tuple[TensorFn, TensorFn]:
        shared_outputs = self._shared_steps(query, 'q')
        if shared_outputs is None:
            return do_nothing, do_nothing

        dst_idx, src_idx, dst_tokens, src_tokens, edge_idx, sim_dst_idx, num_unreduce_src = shared_outputs

        reduce_outputs = reduce_sequence(
            dst_tokens, src_tokens, edge_idx, sim_dst_idx, num_unreduce_src, self.config['reduce_mode'], 
            return_indices=True)
        # logger.debug(f"{feature_name} {reduce_outputs.reduced_tokens.shape=}")

        reduce_func = lambda x: reduce_outputs.reduced_tokens
        if self.config['encoder_first']:  # CogVideoX
            restore_func = lambda x, text_seq_length: torch.cat([
                x[:, :, :text_seq_length], 
                restore_sequence(
                    reduced_tokens=x[:, :, text_seq_length:],
                    reduce_sim_dst_idx=reduce_outputs.reduce_sim_dst_idx, 
                    unreduce_src_idx=reduce_outputs.unreduce_src_idx, 
                    reduce_src_idx=reduce_outputs.reduce_src_idx, 
                    dst_idx=dst_idx, 
                    src_idx=src_idx, 
                    num_unreduce_src=num_unreduce_src, 
                ),
            ], dim=-2)
        else:  # HunyuanVideo, Mochi-1
            restore_func = lambda x, text_seq_length: torch.cat([
                restore_sequence(
                    reduced_tokens=x[:, :, :-text_seq_length],
                    reduce_sim_dst_idx=reduce_outputs.reduce_sim_dst_idx, 
                    unreduce_src_idx=reduce_outputs.unreduce_src_idx, 
                    reduce_src_idx=reduce_outputs.reduce_src_idx, 
                    dst_idx=dst_idx, 
                    src_idx=src_idx, 
                    num_unreduce_src=num_unreduce_src, 
                ),
                x[:, :, -text_seq_length:],
            ], dim=-2)

        return reduce_func, restore_func

    @torch.no_grad()
    def get_kv_output_fn(self, key: Tensor, value: Tensor) -> Tuple[TensorFn, None]:
        shared_outputs = self._shared_steps(value, 'v')
        if shared_outputs is None:
            return do_nothing, None

        dst_idx, src_idx, v_dst_tokens, v_src_tokens, edge_idx, sim_dst_idx, num_unreduce_src = shared_outputs

        k_dst_tokens, k_src_tokens = gather_features(key, dst_idx, src_idx)
        key_reduce_outputs = reduce_sequence(
            k_dst_tokens, k_src_tokens, edge_idx, sim_dst_idx, num_unreduce_src, self.config['reduce_mode'])
        value_reduce_outputs = reduce_sequence(
            v_dst_tokens, v_src_tokens, edge_idx, sim_dst_idx, num_unreduce_src, self.config['reduce_mode'])
        # logger.debug(f"Key {key_reduce_outputs.reduced_tokens.shape=} {value_reduce_outputs.reduced_tokens.shape=}")

        reduce_func = lambda kv: (key_reduce_outputs.reduced_tokens, value_reduce_outputs.reduced_tokens)

        return reduce_func, None