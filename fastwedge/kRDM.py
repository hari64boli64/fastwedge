import numpy as np
from math import factorial
from scipy.sparse import csr_matrix, coo_matrix
from tqdm.notebook import tqdm
from typing import Dict
from itertools import combinations, combinations_with_replacement
from fastwedge._basis import _generate_fixed_parity_permutations,\
    _generate_parity_permutations,\
    _getIdx


# openfermion.jordan_wigner_ladder_sparse()
def _make_jordan_wigners_mul_vec(Q: int, k: int, vec: np.ndarray,
                                 ) -> Dict[int, np.ndarray]:
    assert 1 <= k <= Q

    n_hilbert = 2**Q
    jordan_wigners_mul_vec = dict()

    for ps in combinations(range(Q)[::-1], k):
        data = []
        row = []
        col = []
        mask = sum(1 << (Q-1-p) for p in ps)
        for r in range(n_hilbert-sum(1 << (Q-1-p) for p in ps)):
            if r & mask:
                continue
            data.append(+1 if (sum(bin(r >> (Q-1-p))[2:].count('1')
                                   for p in ps) % 2 == 1) else -1)
            row.append(r)
            col.append(r+sum(1 << (Q-1-p) for p in ps))

        ans = csr_matrix((data, (row, col)),
                         shape=(n_hilbert, n_hilbert)) @ vec
        jordan_wigners_mul_vec[_getIdx(Q, *ps)] = ans
        jordan_wigners_mul_vec[_getIdx(
            Q, *ps[::-1])] = ans*((-1)**((k*(k-1)//2) % 2))

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
    Q = int(np.log2(vec.shape[0]))
    assert 2**Q == vec.shape[0]

    # 要請: k <= Q でなければならない(そうでなければ全て0)
    assert 1 <= k <= Q

    rdm_data = []
    rdm_idx = []
    fixed_k = _generate_fixed_parity_permutations(k)

    QCk = factorial(Q)//factorial(k)//factorial(Q-k)

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
                rdm_data.append(val_p1*parity2)
                rdm_data.append(val_conj_p1*parity2)
                rdm_idx.append(idx1*idx_up+idx2)
                rdm_idx.append(idx2*idx_up+idx1)

    return coo_matrix((rdm_data, (rdm_idx, [0]*len(rdm_data))),
                      shape=(Q**(2*k), 1))\
        .toarray()\
        .reshape(tuple(Q for _ in range(2*k)))
