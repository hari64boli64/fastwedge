import math
import numpy as np
from numba import jit
from tqdm.notebook import tqdm
from typing import List, Tuple, Union
from itertools import product, combinations, permutations


@jit
def __getIdx(Q: int, *args: int):
    """Calculate 1 dim idx from muiti dim idx

    Args:
        Q (int): size of original tensor

    Returns:
        int: index (e.g. Q=10, args=(1,2,3,4) -> 1234)
    """
    # assert all(0 <= arg < Q for arg in args)
    ret = 0
    for arg in args:
        ret *= Q
        ret += arg
    return ret


def __get_sign_of_perm(perm: Union[List[int], Tuple[int, ...]]) -> int:
    """Calculate sign of permutation.

    See sympy.combinatorics / Permutation / _af_parity

    Args:
        perm (List[int] or Tuple[int, ...]): permutation(0,1,...,len(perm)-1)

    Returns:
        int: sign of permutation (+1 or -1)
    """
    n = len(perm)
    a = [0] * n
    c = 0
    for j in range(n):
        if a[j] == 0:
            c += 1
            a[j] = 1
            i = j
            while perm[i] != j:
                i = perm[i]
                a[i] = 1
    return 1 if (n - c) % 2 == 0 else -1


def __my_generate_fixed_parity_permutations(N: int
                                            ) -> List[Tuple[Tuple[int, ...],
                                                            int]]:
    """Precalcuate all permutaion of list(range(N)) and its sign

    Args:
        N (int): length of permutation

    Returns:
        List[Tuple[Tuple[int, ...], int]]: list of (perm, sign)
    """
    return [(perm, __get_sign_of_perm(perm))
            for perm in permutations(range(N))]


def __generate_parity_permutations(seq: Tuple[int, ...],
                                   fixed_parity_perms: List[Tuple[
                                       Tuple[int, ...], int]]
                                   ) -> List[Tuple[List[int], int]]:
    """Enumerate all permutaion of seq and its sign
    using the precalculation result.

    Args:
        seq (Tuple[int, ...]): target sequence
        fixed_parity_perms (List[Tuple[Tuple[int, ...], int]]):
            result of precalculation

    Returns:
        List[Tuple[List[int], int]]: list of (perm, sign)
    """
    return [([seq[idx] for idx in idxs], parity)
            for idxs, parity in fixed_parity_perms]


def __generate_fixed_partial_perms(
        N: int, p: int) -> List[Tuple[List[int], List[int], int]]:
    """Precalcuate partial permutaion of list(range(N)) and its sign

    Args:
        N (int): length of permutation
        p (int): length of perm part

    Returns:
        List[Tuple[List[int],List[int],int]]: list of (perm1, perm2, sign)

    Example:
        N=4, p=2

          (perm)  (sorted) (sign)
        [([0, 1],  [2, 3],  +1),
         ([0, 2],  [1, 3],  -1),
         ([0, 3],  [1, 2],  +1),
         ([1, 0],  [2, 3],  -1),
         ([1, 2],  [0, 3],  +1),
         ([1, 3],  [0, 2],  -1),...]
    """
    ret = []
    for _perm1 in permutations(range(N), p):
        perm1 = list(_perm1)
        perm2 = sorted(list(set(range(N))-set(perm1)))
        all_perm = perm1+perm2
        ret.append((perm1, perm2, __get_sign_of_perm(all_perm)))
    return ret


def __partial_perms(seq: Tuple[int, ...],
                    fixed_partial_perms: List[Tuple[List[int],
                                                    List[int], int]]
                    ) -> List[Tuple[List[int], List[int], int]]:
    """Enumerate all permutaion of seq and its sign
    using the precalculation result.

    Args:
        seq (Tuple[int, ...]): target sequence
        fixed_partial_perms (List[Tuple[List[int], List[int], int]]):
            result of precalculation

    Returns:
        List[Tuple[List[int], List[int], int]]: list of (perm1, perm2, sign)
    """
    return [([seq[i1] for i1 in idxs1],
             [seq[i2] for i2 in idxs2],
             parity) for idxs1, idxs2, parity in fixed_partial_perms]


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
    fixed_N = __my_generate_fixed_parity_permutations(N)
    fixed_q = __my_generate_fixed_parity_permutations(q)
    fixed_Np = __generate_fixed_partial_perms(N, p)

    # right_tensorについての事前計算
    fixed_right_dict = dict()
    for iq in combinations(range(Q), q):
        for jq in combinations(range(Q), q):
            right = 0
            for niq, parity1 in __generate_parity_permutations(iq,
                                                               fixed_q):
                for njq, parity2 in __generate_parity_permutations(jq,
                                                                   fixed_q):
                    right += right_tensor_list[__getIdx(Q, *niq, *njq)] * \
                        parity1*parity2
            fixed_right_dict[__getIdx(Q, *iq, *jq)] = right

    # TODO: Q < p+q
    for ipiq, jpjq in tqdm(product(combinations(range(Q), p+q),
                                   combinations(range(Q), p+q)),
                           total=(math.comb(Q, p+q)**2)):
        parity_ipiq = __generate_parity_permutations(ipiq, fixed_N)
        parity_jpjq = __generate_parity_permutations(jpjq, fixed_N)
        ans = 0.0+0.0j

        for nip, niq, i_parity in __partial_perms(ipiq, fixed_Np):
            for njp, njq, j_parity in __partial_perms(jpjq, fixed_Np):
                ans += left_tensor_list[__getIdx(Q, *nip, *njp)] * \
                    fixed_right_dict[__getIdx(
                        Q, *niq, *njq)] * i_parity*j_parity

        ans /= N_fact_2
        for nipiq, i_parity in parity_ipiq:
            for njpjq, j_parity in parity_jpjq:
                tensor[__getIdx(Q, *nipiq, *njpjq)] = ans*i_parity*j_parity

    # 添字順序の逆転による影響を考慮する符号
    sign_adjustment = (-1)**(p*q)

    return np.array(tensor)\
        .reshape(tuple(Q for _ in range(2*N)))\
        * sign_adjustment
