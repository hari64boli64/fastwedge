import math
import numpy as np
import openfermion
import scipy.sparse
from tqdm.notebook import tqdm
from functools import lru_cache
from typing import List, Tuple, Union, Any
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
        jw_operators += [
            (openfermion.jordan_wigner_ladder_sparse(n_qubits,
                                                     tensor_factor,
                                                     0),
             openfermion.jordan_wigner_ladder_sparse(n_qubits,
                                                     tensor_factor,
                                                     1))
        ]
    return jw_operators


def __my_jordan_wigner_sparse(fermion_operator: openfermion.FermionOperator,
                              n_qubits: int) -> scipy.sparse.coo_matrix:
    r"""openfermion.jordan_wigner_sparse with the cached function.

    Args:
        fermion_operator(FermionOperator): instance of
                                           the FermionOperator class.
        n_qubits(int): Number of qubits.

    Returns:
        scipy.sparse.coo_matrix: The corresponding Scipy sparse matrix.
    """
    jw_operators = __make_jw_operators(n_qubits)
    # Construct the Scipy sparse matrix.
    n_hilbert = 2**n_qubits
    values_list = [[]]
    row_list = [[]]
    column_list = [[]]
    for term in fermion_operator.terms:
        coefficient = fermion_operator.terms[term]
        sparse_matrix = coefficient * scipy.sparse.identity(
            2**n_qubits, dtype=complex, format='csc')
        for ladder_operator in term:
            sparse_matrix = sparse_matrix * jw_operators[ladder_operator[0]][
                ladder_operator[1]]

        if coefficient:
            # Extract triplets from sparse_term.
            sparse_matrix = sparse_matrix.tocoo(copy=False)
            values_list.append(sparse_matrix.data)
            (row, column) = sparse_matrix.nonzero()
            row_list.append(row)
            column_list.append(column)

    values_list = np.concatenate(values_list)
    row_list = np.concatenate(row_list)
    column_list = np.concatenate(column_list)
    sparse_operator = scipy.sparse.coo_matrix(
        (values_list, (row_list, column_list)),
        shape=(n_hilbert, n_hilbert)).tocsc(copy=False)
    sparse_operator.eliminate_zeros()
    return sparse_operator


def __my_get_sparse_operator(operator: openfermion.FermionOperator,
                             n_qubit: int) -> scipy.sparse.coo_matrix:
    """Limited function of openfermion.get_sparse_operator

    Args:
        operator (openfermion.FermionOperator): this must be instance of
                                                the FermionOperator class.
        n_qubits(int): Number of qubits.

    Returns:
        scipy.sparse.coo_matrix: The corresponding Scipy sparse matrix.
    """
    assert not isinstance(operator, (openfermion.DiagonalCoulombHamiltonian,
                                     openfermion.PolynomialTensor))
    assert isinstance(operator, openfermion.FermionOperator)
    return __my_jordan_wigner_sparse(operator, n_qubit)


def __fast_expectation(operator: Union[np.ndarray,
                                       openfermion.FermionOperator],
                       state: np.ndarray,
                       state_conj: Union[np.ndarray, scipy.sparse.csc_matrix],
                       n_qubit: int) -> np.ndarray:
    if type(operator) == np.ndarray:
        return state_conj @ operator @ state
    else:
        return scipy.sparse.csc_matrix.toarray(state_conj @
                                               __my_get_sparse_operator(
                                                   operator,
                                                   n_qubit
                                               )
                                               ) @ state


def fast_compute_k_rdm(k: int, vec: np.ndarray) -> np.ndarray:
    """compute k-RDM

    Args:
        k (int): k of k-RDM
        vec (np.ndarray): Haar state

    Returns:
        np.ndarray: k-RDM of vec
    """
    assert k >= 1
    csc_vector_conj = scipy.sparse.csc_matrix(vec.conj())
    Q = int(np.log2(vec.shape[0]))
    assert 2**Q == vec.shape[0]
    rdm = [0.0+0.0j for _ in range(Q**(2*k))]
    fixed_k = _generate_fixed_parity_permutations(k)

    QCk = math.factorial(Q)//math.factorial(k)//math.factorial(Q-k)

    for ps, qs in tqdm(combinations_with_replacement(combinations(range(Q), k),
                                                     2),
                       total=QCk*(QCk+1)//2):
        val = __fast_expectation(openfermion.FermionOperator(
            ("".join(map(lambda p: f"{p}^ ", ps))
             + "".join(map(lambda q: f"{q} ", qs)))[:-1]),
            vec, csc_vector_conj, Q)[0]
        # ps==qsの場合、以下は一部無駄があるが、条件分岐を挟む方が時間が掛かりそう。
        for perm1, parity1 in _generate_parity_permutations(ps, fixed_k):
            for perm2, parity2 in _generate_parity_permutations(qs, fixed_k):
                rdm[_getIdx(Q, *perm1, *perm2)] = (val*parity1*parity2)
                rdm[_getIdx(Q, *perm2, *perm1)] = (val*parity1*parity2).conj()
    return np.array(rdm).reshape(tuple(Q for _ in range(2*k)))
