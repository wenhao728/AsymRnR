from dataclasses import dataclass
from typing import Callable, Literal, Optional, Tuple, Union

from torch import Tensor

# type hints
MaybeTensorTuple = Union[Tensor, Tuple[Tensor, Tensor]]
TensorFn = Callable[[MaybeTensorTuple], MaybeTensorTuple]
SimilarityType = Literal['dot', 'cosine', 'euclidean', 'random']
ReduceMode = Literal['mean', 'replace']


# output types
@dataclass
class MatchingOutput:
    """Output of token matching function.
    
    Attributes:
        sim_value (Tensor): The similarity values between source tokens and their matched (closest) destination tokens.
            Shape (batch_size, num_heads, num_src_tokens).
        sim_dst_idx (Tensor): The indices of the matched destination tokens.
            Shape (batch_size, num_heads, num_src_tokens).
        edge_idx (Tensor): The indices of the matching edges, sorted by distance, in descending order. In other words,
            the edges at the ending of the tensor will be reduced first.
    """
    sim_value: Tensor
    sim_dst_idx: Tensor
    edge_idx: Tensor


@dataclass
class ReduceOutput:
    """Output of the reduction function.
    
    Attributes:
        reduced_tokens (Tensor): The reduced tokens after the reduction operation.
            Shape (batch_size, num_heads, sequence_length * reduce_ratio).
        reduce_sim_dst_idx (Tensor): The indices of the reduced tokens' closest destination tokens.
            Shape (batch_size, num_heads, sequence_length * reduce_ratio).
        unreduce_src_idx (Tensor): The indices of the reduced tokens in the original source tokens.
            Shape (batch_size, num_heads, num_src_tokens - sequence_length * reduce_ratio).
        reduce_src_idx (Tensor): The indices of the reduced tokens in the original sequence.
            Shape (batch_size, num_heads, sequence_length * reduce_ratio)
    """
    reduced_tokens: Tensor
    reduce_sim_dst_idx: Optional[Tensor] = None
    unreduce_src_idx: Optional[Tensor] = None
    reduce_src_idx: Optional[Tensor] = None