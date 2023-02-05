import math
import numpy as np
from tqdm.notebook import tqdm
from typing import List, Tuple
from itertools import product, combinations
from fastwedge._basis import _generate_fixed_partial_perms,\
    _generate_parity_permutations,\
    _generate_fixed_parity_permutations,\
    _partial_perms,\
    _getIdx


def fast_wedge_topM(left_tensor: np.ndarray,
                    right_tensor: np.ndarray,
                    left_index_ranks: Tuple[int, int],
                    right_index_ranks: Tuple[int, int],
                    M: int,
                    verbose: bool = True) -> List[Tuple[float, complex,
                                                        Tuple[int, ...],
                                                        Tuple[int, ...]]]:
    """Enumerate only the top M absolute values of the wedge product

    For more details, please see fastwedge.fast_wedge.

    Warning:
        Please pay attention to the order of the indexes.
        Since this function is consistent with the openfermion.wedge,
        <a^ b^ c d> = D_{dc}^{ab}. (where D is the wedge product)
        In other words, it should originally be denoted as
        (ipiq, jqjp) instead of (ipiq, jpjq).
        See also the section of Returns.

    Args:
        left_tensor: left tensor to wedge product
        right_tensor: right tensor to wedge product
        left_index_ranks: tuple of number of indices that transform
                          contravariently and covariently
        right_index_ranks: tuple of number of indices that transform
                           contravariently and covariently
        M: the M of topM
        verbose (bool, optional): Show progress. Defaults to True.

    Returns:
        List[Tuple[float, complex, Tuple[int, ...], Tuple[int, ...]]]:
            list of (abs(elem), elem, ipiq, jpjq)

            np.isclose(
                openfermion.wedge(...)[tuple(list(ipiq)+list(jpjq))],
                elem
            ) is True.
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

    # ランダムアクセスが必要、かつ、多次元配列のままだと遅いので、通常の一次元listを使用
    left_tensor_list = left_tensor.flatten().tolist()
    right_tensor_list = right_tensor.flatten().tolist()

    # 符号や順列などについての事前計算
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

    # 答えを保持する配列 (要素の絶対値、要素、上側添字、下側添字)の組
    answers = []

    # 添え字についてソートされているものを代表元としてループを回す
    for ipiq, jpjq in tqdm(product(combinations(range(Q), p+q),
                                   combinations(range(Q), p+q)),
                           total=(math.factorial(Q)
                                  // math.factorial(p+q)
                                  // math.factorial(Q-(p+q)))**2,
                           disable=not verbose):
        # 代表元に当たる要素の計算
        ans = 0.0+0.0j
        for nip, niq, i_parity in _partial_perms(ipiq, fixed_Np):
            for njp, njq, j_parity in _partial_perms(jpjq, fixed_Np):
                ans += left_tensor_list[_getIdx(Q, *nip, *njp)] * \
                    fixed_right_list[_getIdx(
                        Q, *niq, *njq)] * i_parity*j_parity
        ans /= N_fact_2
        ans *= sign_adjustment

        answers.append((abs(ans), ans, ipiq, jpjq))

    # 降順にソート
    total_num = len(answers)
    answers.sort(key=lambda x: x[0], reverse=True)
    answers = answers[:min(len(answers), M)]

    print(f"== top {len(answers)} elements of {total_num} elements ==")
    for i, (abs_val, val, ipiq, jpjq)\
            in enumerate(answers[:min(len(answers), M)]):
        print(f"{i:>3} | {abs_val:.3e} | {val:.3e} | {ipiq} | {jpjq}")

    return answers
