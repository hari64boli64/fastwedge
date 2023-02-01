from numba import jit
from typing import List, Tuple, Union
from itertools import permutations


@jit
def _getIdx(Q: int, *args: int) -> int:
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


def _get_sign_of_perm(perm: Union[List[int], Tuple[int, ...]]) -> int:
    """Calculate sign of permutation.

    See sympy.combinatorics / Permutation / _af_parity

    Args:
        perm (Union[List[int], Tuple[int, ...]]):
            permutation(0,1,...,len(perm)-1)

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


def _generate_fixed_partial_perms(
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
        ret.append((perm1, perm2, _get_sign_of_perm(all_perm)))
    return ret


def _partial_perms(seq: Tuple[int, ...],
                   fixed_partial_perms: List[Tuple[List[int],
                                                   List[int], int]]
                   ) -> List[Tuple[List[int], List[int], int]]:
    """Enumerate all permutaion of seq and its sign
    using the precalculated result.

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


def _generate_fixed_parity_permutations(N: int
                                        ) -> List[Tuple[Tuple[int, ...], int]]:
    """Precalcuate all permutaion of list(range(N)) and its sign

    Args:
        N (int): length of permutation

    Returns:
        List[Tuple[Tuple[int, ...], int]]: list of (perm, sign)
    """
    return [(perm, _get_sign_of_perm(perm))
            for perm in permutations(range(N))]


def _generate_parity_permutations(seq: Tuple[int, ...],
                                  fixed_parity_perms: List[Tuple[
                                      Tuple[int, ...], int]]
                                  ) -> List[Tuple[List[int], int]]:
    """Enumerate all permutaion of seq and its sign
    using the precalculated result.

    Args:
        seq (Tuple[int, ...]): target sequence
        fixed_parity_perms (List[Tuple[Tuple[int, ...], int]]):
            result of precalculation

    Returns:
        List[Tuple[List[int], int]]: list of (perm, sign)
    """
    return [([seq[idx] for idx in idxs], parity)
            for idxs, parity in fixed_parity_perms]
