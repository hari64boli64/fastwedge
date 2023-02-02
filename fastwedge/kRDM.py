import math
import numpy as np
import openfermion
from tqdm.notebook import tqdm
from functools import lru_cache
from typing import List, Tuple, Any
from itertools import combinations, combinations_with_replacement
from fastwedge._basis import _generate_fixed_parity_permutations,\
    _generate_parity_permutations,\
    _getIdx


@lru_cache(maxsize=10)
def __make_jw_operators(n_qubits: int) -> List[Tuple[Any, Any]]:
    """Cached function of one part of the openfermion.jordan_wigner_sparse

    Args:
        n_qubits(int): Number of qubits.

    Returns:
        List[Tuple[Any, Any]]: list of tuple of
                               openfermion.jordan_wigner_ladder_sparse

    Note:
        The max size of cache is now limited to 10, which can be modified.
    """
    # Create a list of raising and lowering operators for each orbital.
    jw_operators = []
    for tensor_factor in range(n_qubits):
        jw_operators.append(
            openfermion.jordan_wigner_ladder_sparse(n_qubits,
                                                    tensor_factor,
                                                    0).tocsr())
    return jw_operators


def _make_jordan_wigners_mul_vec(Q, k, vec):
    assert k >= 1

    jw_operators = __make_jw_operators(Q)

    if k == 1:
        return [jw_operator @ vec for jw_operator in jw_operators]

    # # slow
    # n_hilbert = 2**Q
    # for ps in permutations(range(Q), k):
    #     sparse_matrix = scipy.sparse.identity(n_hilbert,
    #                                           dtype=complex,
    #                                           format='csc')
    #     for ladder_operator in ps:
    #         sparse_matrix = sparse_matrix * jw_operators[ladder_operator]
    #     jordan_wigners_mul_vec[_getIdx(Q, *ps)] = sparse_matrix @ vec

    jordan_wigners_mul_vec = [None for _ in range(Q**k)]
    path = []
    seen = [False for _ in range(Q)]
    que = []
    for i in range(Q):
        que.append((~i, None))
        que.append((i, jw_operators[i]))

    while que:
        i, mat = que.pop()
        if i >= 0:
            seen[i] = True
            path.append(i)
            for ni in range(Q):
                if seen[ni]:
                    continue
                if len(path) < k-1:
                    que.append((~ni, None))
                    que.append((ni, mat @ jw_operators[ni]))
                elif len(path) == k-1:
                    jordan_wigners_mul_vec[_getIdx(Q, *path, ni)] =\
                        mat @ jw_operators[ni] @ vec
        else:
            seen[~i] = False
            path.pop()

    return jordan_wigners_mul_vec


def fast_compute_k_rdm(k: int, vec: np.ndarray,
                       verbose: bool = True) -> np.ndarray:
    """compute k-RDM

    Args:
        k (int): k of k-RDM
        vec (np.ndarray): Haar state
        verbose (bool, optional): Show progress. Defaults to True.

    Returns:
        np.ndarray: k-RDM of vec
    """
    assert k >= 1
    Q = int(np.log2(vec.shape[0]))
    assert 2**Q == vec.shape[0]
    rdm = [0.0+0.0j for _ in range(Q**(2*k))]
    fixed_k = _generate_fixed_parity_permutations(k)

    QCk = math.factorial(Q)//math.factorial(k)//math.factorial(Q-k)

    jordan_wigners_mul_vec = _make_jordan_wigners_mul_vec(Q, k, vec)

    idx_up = Q**k

    for ps, qs in tqdm(combinations_with_replacement(combinations(range(Q), k),
                                                     2),
                       total=QCk*(QCk+1)//2,
                       disable=not verbose):
        bra = jordan_wigners_mul_vec[_getIdx(Q, *ps[::-1])]
        ket = jordan_wigners_mul_vec[_getIdx(Q, *qs)]
        val = np.dot(bra.conj(), ket)
        val_conj = val.conj()
        # ps==qsの場合、以下は一部無駄があるが、条件分岐を挟む方が時間が掛かりそう。
        for perm1, parity1 in _generate_parity_permutations(ps, fixed_k):
            val_p1 = val*parity1
            val_conj_p1 = val_conj*parity1
            idx1 = _getIdx(Q, *perm1)
            for perm2, parity2 in _generate_parity_permutations(qs, fixed_k):
                idx2 = _getIdx(Q, *perm2)
                rdm[idx1*idx_up+idx2] = val_p1*parity2
                rdm[idx2*idx_up+idx1] = val_conj_p1*parity2

    return np.array(rdm).reshape(tuple(Q for _ in range(2*k)))
