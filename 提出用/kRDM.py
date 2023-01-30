import numpy as np
import openfermion
import scipy.sparse
from typing import List, Tuple, Union, Any
from functools import lru_cache


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


def __fast_expectation(operator: np.ndarray or openfermion.FermionOperator,
                       state: np.ndarray,
                       state_conj: np.ndarray or scipy.sparse.csc_matrix,
                       n_qubit: int):
    if type(operator) == np.ndarray:
        return state_conj @ operator @ state
    else:
        return scipy.sparse.csc_matrix.toarray(state_conj @
                                               __my_get_sparse_operator(
                                                   operator,
                                                   n_qubit
                                               )
                                               ) @ state


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


def __get_sign_of_args(args: Union[List[int], Tuple[int, ...]]):
    """Calculate sign of args.

    Args:
        perm (List[int] or Tuple[int, ...]): args (e.g. [3,1,4], (0,0,0))

    Returns:
        int: sign of permutation (+1 or -1 or 0)
    """
    if len(args) - len(np.unique(args)) > 0:
        return 0
    return __get_sign_of_perm(np.argsort(args).tolist())


def __get_corresponding_index(args: Tuple[int, ...]
                              ) -> Tuple[Tuple[int, ...], bool, int]:
    # 渡されたnotebookに存在した関数
    k = len(args)//2
    assert len(args) == 2*k
    args_ = list(args)
    args1 = args_[:k]
    args2 = args_[k:]
    sign1 = __get_sign_of_args(args1)
    sign2 = __get_sign_of_args(args2)
    argmin1 = min(args1)
    argmin2 = min(args2)
    if_conjugate = argmin1 > argmin2
    if if_conjugate:
        args1_ref = sorted(args2)
        args2_ref = sorted(args1)
    else:
        args1_ref = sorted(args1)
        args2_ref = sorted(args2)
    return tuple(args1_ref + args2_ref), if_conjugate, sign1 * sign2


def fast_compute_one_rdm(vec: np.ndarray) -> np.ndarray:
    """compute 1-RDM

    Args:
        vec (np.ndarray): Haar random state

    Returns:
        np.ndarray: 1-RDM of vec
    """
    csc_vector_conj = scipy.sparse.csc_matrix(vec.conj())
    n_qubit = int(np.log2(vec.shape[0]))
    assert 2**n_qubit == vec.shape[0]
    rdm1 = np.zeros((n_qubit, n_qubit), dtype=complex)
    for i in range(n_qubit):
        for j in range(i, n_qubit):
            cij = __fast_expectation(openfermion.FermionOperator(
                f"{i}^ {j}"), vec, csc_vector_conj, n_qubit)
            rdm1[i, j] = np.copy(cij)
            rdm1[j, i] = np.copy(cij).conj()
    return rdm1


def fast_compute_two_rdm(vec: np.ndarray) -> np.ndarray:
    """compute 2-RDM

    Args:
        vec (np.ndarray): Haar random state

    Returns:
        np.ndarray: 2-RDM of vec
    """
    csc_vector_conj = scipy.sparse.csc_matrix(vec.conj())
    n_qubit = int(np.log2(vec.shape[0]))
    assert 2**n_qubit == vec.shape[0]
    rdm2 = np.zeros((n_qubit, n_qubit, n_qubit, n_qubit), dtype=complex)
    for i in range(n_qubit):
        for j in range(i, n_qubit):
            for k in range(j, n_qubit):
                for l in range(k, n_qubit):
                    unique_args = list(set([(i, j, k, l),
                                            (i, k, j, l),
                                            (i, l, j, k)]))
                    for args in unique_args:
                        if args[0] == args[1] or args[2] == args[3]:
                            continue
                        exp = __fast_expectation(openfermion.FermionOperator(
                            f"{args[0]}^ {args[1]}^ {args[2]} {args[3]}"),
                            vec, csc_vector_conj, n_qubit)
                        rdm2[args] = exp
    for i in range(n_qubit):
        for j in range(n_qubit):
            for k in range(n_qubit):
                for l in range(n_qubit):
                    args = (i, j, k, l)
                    args_ref, if_conjugate, sign = __get_corresponding_index(
                        args)
                    cijkl = np.copy(rdm2[args_ref]) * sign
                    if if_conjugate:
                        cijkl = cijkl.conj()
                    rdm2[i, j, k, l] = np.copy(cijkl)
    return rdm2
