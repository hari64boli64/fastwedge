import math
import numpy as np
from tqdm.notebook import tqdm
from typing import Tuple
from itertools import product, combinations
from fastwedge._basis import _generate_fixed_partial_perms,\
    _generate_parity_permutations,\
    _generate_fixed_parity_permutations,\
    _partial_perms,\
    _getIdx


def fast_wedge(left_tensor: np.ndarray,
               right_tensor: np.ndarray,
               left_index_ranks: Tuple[int, int],
               right_index_ranks: Tuple[int, int]) -> np.ndarray:
    """fast version of openfermion.wedge

    Implement the wedge product between left_tensor and right_tensor

    The wedge product is defined as

    $$
        \\left( a \\wedge b\\right)
         ^{i_{1}, i_{2}, ..., i_{p},..., i_{p+q}}
         _{j_{1}, j_{2}, ...,j_{p}, ..., j_{p+q} } =
        \\left(\\frac{1}{N!}\\right)^{2}
        \\sum_{\\pi, \\sigma}\\epsilon(\\pi)\\epsilon(\\sigma)
        a_{\\pi(j_{1}), \\pi(j_{2}), ..., \\pi(j_{p}) }
         ^{ \\sigma(i_{1}), \\sigma(i_{2}), ..., \\sigma(i_{p})}
        b_{\\pi(j_{p+1}), \\pi(j_{p+2}), ..., \\pi(j_{p+q}) }
         ^{ \\sigma(i_{p+1}), \\sigma(i_{p+2}), ..., \\sigma(i_{p+q})}
    $$

    The top indices are those that transform contravariently.  The bottom
    indices transform covariently.

    The tensor storage convention for marginals follows the OpenFermion
    convention. tpdm[i, j, k, l] = <i^ j^ k l>,
    rtensor[u1, u2, u3, d1] = <u1^ u2^ u3^ d1>

    Args:
        left_tensor: left tensor to wedge product
        right_tensor: right tensor to wedge product
        left_index_ranks: tuple of number of indices that transform
                          contravariently and covariently
        right_index_ranks: tuple of number of indices that transform
                           contravariently and covariently
    Returns:
        new tensor constructed as the wedge product of the left_tensor and
        right_tensor
    """
    assert left_tensor.ndim == sum(left_index_ranks)                   # 必須
    assert right_tensor.ndim == sum(right_index_ranks)                 # 必須
    assert len(set(left_tensor.shape) | set(right_tensor.shape)) == 1  # 必須

    # TODO: fix(?)
    assert left_index_ranks[0] == left_index_ranks[1]                  # 必須でない
    assert right_index_ranks[0] == right_index_ranks[1]                # 必須でない

    # 定数定義
    p = left_index_ranks[0]
    q = right_index_ranks[0]
    N = p+q
    N_fact_2 = math.factorial(N)**2
    Q = left_tensor.shape[0]

    # ランダムアクセスが必要、かつ、多次元配列のままだと遅いので、通常の一次元listを使用
    tensor = [0.0+0.0j for _ in range(Q**(2*N))]
    left_tensor_list = left_tensor.flatten().tolist()
    right_tensor_list = right_tensor.flatten().tolist()

    # 符号や順列などについての事前計算
    fixed_N = _generate_fixed_parity_permutations(N)
    fixed_q = _generate_fixed_parity_permutations(q)
    fixed_Np = _generate_fixed_partial_perms(N, p)

    # right_tensorについての事前計算
    fixed_right_dict = dict()
    for iq in combinations(range(Q), q):
        for jq in combinations(range(Q), q):
            right = 0
            for niq, parity1 in _generate_parity_permutations(iq, fixed_q):
                for njq, parity2 in _generate_parity_permutations(jq, fixed_q):
                    right += right_tensor_list[_getIdx(Q, *niq, *njq)] * \
                        parity1*parity2
            fixed_right_dict[_getIdx(Q, *iq, *jq)] = right

    # TODO: Q < p+q
    # 添え字についてソートされているものを代表元としてループを回す
    for ipiq, jpjq in tqdm(product(combinations(range(Q), p+q),
                                   combinations(range(Q), p+q)),
                           total=(math.factorial(Q)
                                  // math.factorial(p+q)
                                  // math.factorial(Q-(p+q)))**2):
        parity_ipiq = _generate_parity_permutations(ipiq, fixed_N)
        parity_jpjq = _generate_parity_permutations(jpjq, fixed_N)
        ans = 0.0+0.0j

        # 代表元に当たる要素の計算
        for nip, niq, i_parity in _partial_perms(ipiq, fixed_Np):
            for njp, njq, j_parity in _partial_perms(jpjq, fixed_Np):
                ans += left_tensor_list[_getIdx(Q, *nip, *njp)] * \
                    fixed_right_dict[_getIdx(
                        Q, *niq, *njq)] * i_parity*j_parity
        ans /= N_fact_2

        # 同値類に属する要素へ、代表元の値を利用して計算
        for nipiq, i_parity in parity_ipiq:
            for njpjq, j_parity in parity_jpjq:
                tensor[_getIdx(Q, *nipiq, *njpjq)] = ans*i_parity*j_parity

    # 添字順序の逆転による影響を考慮する符号
    sign_adjustment = (-1)**(p*q)

    return np.array(tensor)\
        .reshape(tuple(Q for _ in range(2*N)))\
        * sign_adjustment
