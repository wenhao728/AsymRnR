import logging
from abc import ABC, abstractmethod
from typing import Any, Dict, Optional, Tuple

from torch import nn

from .types import Tensor, TensorFn
from .utils import isinstance_str

logger = logging.getLogger(__name__)


def do_nothing(x: Tensor, *args, **kwargs) -> Tensor:
    return x


class BaseOps(ABC):
    _required_config_keys = []

    """The base class for function sets. 
    
    For qkv, q, and mlp projections, another inverse projection is required as it influences the feature shape.
    For kv projection, the inverse projection is not required as it does not influence the feature shape.
    """
    def __init__(self, config: Optional[Dict[str, Any]] = None) -> None:
        super().__init__()
        self.config = config or {}

        for key in self._required_config_keys:
            if key not in self.config:
                raise ValueError(f'{key} is not set. Please set {key} in the config.')

    @property
    def mask_start_index(self):
        return 0, 0

    def update_config(self, **kwargs) -> None:
        self.config.update(kwargs)

    @abstractmethod
    def get_qkv_input_fn(self, hidden_states: Tensor) -> Tuple[TensorFn, TensorFn]:
        raise NotImplementedError

    @abstractmethod
    def get_q_output_fn(self, query: Tensor) -> Tuple[TensorFn, TensorFn]:
        raise NotImplementedError

    @abstractmethod
    def get_kv_output_fn(self, key: Tensor, value: Tensor) -> Tuple[TensorFn, None]:
        raise NotImplementedError


class DoNothingOps(BaseOps):
    def __init__(self, config: Optional[Dict[str, Any]] = None) -> None:
        self.config = config or {}

    def get_qkv_input_fn(self, hidden_states: Tensor) -> Tuple[TensorFn, TensorFn]:
        reduce_fn, restore_fn = do_nothing, do_nothing
        return reduce_fn, restore_fn

    def get_q_output_fn(self, query: Tensor) -> Tuple[TensorFn, TensorFn]:
        reduce_fn, restore_fn = do_nothing, do_nothing
        return reduce_fn, restore_fn

    def get_kv_output_fn(self, key: Tensor, value: Tensor) -> Tuple[TensorFn, None]:
        reduce_fn = do_nothing
        return reduce_fn, None


def update_processor_config(
    model: nn.Module,
    kv_pairs: Dict[int, Any],
):
    for name, module in model.named_modules():
        if (
            isinstance_str(module, 'Attention') or
            isinstance_str(module, 'MochiAttention')
        ) and hasattr(module.processor, 'ops'):
            module.processor.update_config(**kv_pairs)


class OpsMixin:
    def __init__(self,):
        self.ops = DoNothingOps()

    def update_ops(self, ops: BaseOps):
        self.ops = ops
    
    def update_config(self, **kwargs):
        self.ops.update_config(**kwargs)