import numpy as np
from typing import Tuple
from openfermion import wedge


def slow_wedge(left_tensor: np.ndarray,
               right_tensor: np.ndarray,
               left_index_ranks: Tuple[int, int],
               right_index_ranks: Tuple[int, int]) -> np.ndarray:
    return wedge(left_tensor, right_tensor,
                 left_index_ranks, right_index_ranks)
