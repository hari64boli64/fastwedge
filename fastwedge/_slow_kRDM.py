import numpy as np
from openfermion import FermionOperator, get_sparse_operator
from typing import Tuple


def expectation(operator, state):
    n_qubit = int(np.log2(state.shape[0]))
    if type(operator) == np.ndarray:
        return state.conj() @ operator @ state
    else:
        return state.conj() @ \
            get_sparse_operator(operator, n_qubits=n_qubit).toarray() @\
            state


def slow_compute_k_rdm(vector: np.ndarray, args: Tuple[int]) -> complex:
    assert len(args) % 2 == 0
    cij = expectation(FermionOperator(
        ("".join(map(lambda p: f"{p}^ ", args[:len(args)//2]))
         + "".join(map(lambda q: f"{q} ", args[len(args)//2:])))[:-1]), vector)
    return cij
