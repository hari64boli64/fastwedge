import math
import numpy as np
from typing import Tuple
from tqdm.notebook import tqdm
from scipy.sparse import coo_matrix
from itertools import product, combinations
from fastwedge._basis import _generate_fixed_partial_perms,\
    _generate_parity_permutations,\
    _generate_fixed_parity_permutations,\
    _partial_perms,\
    _getIdx


def fast_wedge(left_tensor: np.ndarray,
               right_tensor: np.ndarray,
               left_index_ranks: Tuple[int, int],
               right_index_ranks: Tuple[int, int],
               verbose: bool = True) -> np.ndarray:
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

    Warning:
        We define a,b,p,q,N,Q as below.
        $$
            a_0^0=1,a_1^0=2,a_0^1=3,a_1^1=4 \\\\
            b_0^0=1,b_1^0=2,b_0^1=3,b_1^1=4 \\\\
            p=1,q=1,N=2,Q=2
        $$

        Then,
        $$
            \\begin{align}
                & (a \\wedge b)^{0, 1}_{0, 1} \\\\
                =& \\left(\\frac{1}{N!}\\right)^{2}
                    \\sum_{\\pi, \\sigma}\\epsilon(\\pi)\\epsilon(\\sigma)
                    a_{\\pi(0)}^{ \\sigma(0)}b_{\\pi(1)}^{ \\sigma(1)} \\\\
                =& \\frac{1}{4} ((1*1*a_0^0b_1^1)+(-1*1*a_1^0b_0^1)
                                +(1*-1*a_0^1b_1^0)+(-1*-1*a_1^1b_0^0)) \\\\
                =& \\frac{1}{4} ((1*1*1*4)+(-1*1*2*3)
                                +(1*-1*3*2)+(-1*-1*4*1)) \\\\
                =& -1
            \\end{align}
        $$

        However,The code below

        ```python
        example_a = np.array([[1, 2], [3, 4]])
        example_b = np.array([[1, 2], [3, 4]])
        example_tensor = wedge(example_a, example_b, (1, 1), (1, 1))
        print(f"{example_tensor[0,1,0,1]=}")
        ```

        returns "example_tensor[0,1,0,1]=(1+0j)", which is not -1.

        This is because
        $$
            tensor[0,1,0,1]=(a \\wedge b)^{0, 1}_{1, 0}
        $$
        (<a^ b^ c d> = D_{dc}^{ab},
        as noted in the comment of openfermion.wedge)

    Args:
        left_tensor: left tensor to wedge product
        right_tensor: right tensor to wedge product
        left_index_ranks: tuple of number of indices that transform
                          contravariently and covariently
        right_index_ranks: tuple of number of indices that transform
                           contravariently and covariently
        verbose (bool, optional): Show progress. Defaults to True.

    Returns:
        new tensor constructed as the wedge product of the left_tensor and
        right_tensor
    """
    # 要請1: left_tensorの次元は、left_index_ranksの総和と一致する(openfermionにもある仕様)
    assert left_tensor.ndim == sum(left_index_ranks)
    # 要請2: right_tensorの次元は、right_index_ranksの総和と一致する(openfermionにもある仕様)
    assert right_tensor.ndim == sum(right_index_ranks)
    # 要請3: left_tensorとright_tensorでn_qubitsが同一である
    assert len(set(left_tensor.shape) | set(right_tensor.shape)) == 1
    # 要請4: left_tensor_ranksは同じ数字のペアでなければならない(openfermionにはない仕様)
    assert left_index_ranks[0] == left_index_ranks[1]
    # 要請5: right_tensor_ranksは同じ数字のペアでなければならない(openfermionにはない仕様)
    assert right_index_ranks[0] == right_index_ranks[1]
    # 要請6: n_qubits >= p + q でなければならない(そうでなければ全て0)
    assert left_tensor.shape[0] >= left_index_ranks[0]+right_index_ranks[0]

    if left_index_ranks[0] > right_index_ranks[1]:
        left_tensor, right_tensor =\
            right_tensor, left_tensor
        left_index_ranks, right_index_ranks =\
            right_index_ranks, left_index_ranks

    # 定数定義
    p = left_index_ranks[0]
    q = right_index_ranks[0]
    N = p+q
    N_fact_2 = math.factorial(N)**2
    Q = left_tensor.shape[0]
    idx_up = Q**N
    QCN = math.factorial(Q) // math.factorial(N) // math.factorial(Q-N)
    QPN = math.factorial(Q) // math.factorial(Q-N)

    # ランダムアクセスが必要、かつ、多次元配列のままだと遅いので、通常の一次元listを使用
    tensor_data = [0j]*(QPN**2)
    tensor_idx = [0]*(QPN**2)
    left_tensor_list = left_tensor.flatten().tolist()
    right_tensor_list = right_tensor.flatten().tolist()

    # 符号や順列などについての事前計算
    fixed_N = _generate_fixed_parity_permutations(N)
    fixed_q = _generate_fixed_parity_permutations(q)
    fixed_Np = _generate_fixed_partial_perms(N, p)

    # right_tensorについての事前計算
    fixed_right_list = [0.0+0.0j for _ in range(Q**(2*q))]
    for iq in combinations(range(Q), q):
        for jq in combinations(range(Q), q):
            right = 0.0+0.0j
            for niq, parity1 in _generate_parity_permutations(iq, fixed_q):
                for njq, parity2 in _generate_parity_permutations(jq, fixed_q):
                    right += right_tensor_list[_getIdx(Q, *niq, *njq)] * \
                        parity1*parity2
            fixed_right_list[_getIdx(Q, *iq, *jq)] = right

    # 添字順序の逆転による影響を考慮する符号
    sign_adjustment = (-1)**(p*q)

    # 添え字についてソートされているものを代表元としてループを回す
    i = 0
    for ipiq, jpjq in tqdm(product(combinations(range(Q), p+q),
                                   combinations(range(Q), p+q)),
                           total=(QCN)**2,
                           disable=not verbose):
        parity_ipiq = _generate_parity_permutations(ipiq, fixed_N)
        parity_jpjq = _generate_parity_permutations(jpjq, fixed_N)
        ans = 0.0+0.0j

        # 代表元に当たる要素の計算
        for nip, niq, i_parity in _partial_perms(ipiq, fixed_Np):
            for njp, njq, j_parity in _partial_perms(jpjq, fixed_Np):
                ans += left_tensor_list[_getIdx(Q, *nip, *njp)] * \
                    fixed_right_list[_getIdx(
                        Q, *niq, *njq)] * i_parity*j_parity
        ans /= N_fact_2
        ans *= sign_adjustment

        # 同値類に属する要素へ、代表元の値を利用して計算
        for nipiq, i_parity in parity_ipiq:
            nipiq_idx = _getIdx(Q, *nipiq)*idx_up
            for njpjq, j_parity in parity_jpjq:
                tensor_idx[i] = nipiq_idx+_getIdx(Q, *njpjq)
                tensor_data[i] = ans*i_parity*j_parity
                i += 1

    return coo_matrix((tensor_data, ([0]*len(tensor_data), tensor_idx)),
                      shape=(1, Q**(2*N)), dtype=complex)\
        .toarray()\
        .reshape(tuple(Q for _ in range(2*N)))
