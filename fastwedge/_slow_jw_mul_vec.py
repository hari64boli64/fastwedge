import scipy.sparse
import openfermion
import numpy as np
from typing import Dict
from itertools import combinations
from fastwedge._basis import _getIdx


def slow_jw_mul_vec(Q: int, k: int, vec: np.ndarray) -> Dict[int, np.ndarray]:
    assert 1 <= k <= Q

    jw_operators = []
    for tensor_factor in range(Q):
        jw_operators.append(
            openfermion.jordan_wigner_ladder_sparse(Q,
                                                    tensor_factor,
                                                    0).tocsr())
    n_hilbert = 2**Q
    jordan_wigners_mul_vec = dict()

    for ps in (list(combinations(range(Q), k)) +
               list(combinations(range(Q)[::-1], k))):
        sparse_matrix = scipy.sparse.identity(n_hilbert,
                                              dtype=complex,
                                              format='csc')
        for ladder_operator in ps:
            sparse_matrix = sparse_matrix * jw_operators[ladder_operator]
        jordan_wigners_mul_vec[_getIdx(Q, *ps)] = sparse_matrix@vec

    return jordan_wigners_mul_vec
