import random
import unittest
import numpy as np
from fastwedge._slow_kRDM import slow_compute_k_rdm
from fastwedge._slow_wedge import slow_wedge
from fastwedge.kRDM import fast_compute_k_rdm
from fastwedge.wedgeProduct import fast_wedge
from fastwedge.wedgeTopM import fast_wedge_topM


def is_particle_number_correct(arg: int, n_qubit: int,
                               n_particle: int) -> bool:
    _l = bin(arg)[2:].zfill(n_qubit)
    return sum([int(b) for b in _l]) == n_particle


# generate Haar random state
def genHaarRandomState(seed: int, n_qubits: int,
                       n_electrons: int) -> np.ndarray:
    np.random.seed(seed)
    vec = np.random.normal(size=2**n_qubits) \
        + 1j * np.random.normal(size=2**n_qubits)
    # particle number restriction
    vec = [vec[i] if is_particle_number_correct(i, n_qubits, n_electrons)
           else 0 for i in range(2**n_qubits)]
    vec /= np.linalg.norm(vec)
    return vec


class Test(unittest.TestCase):
    def _makeVec(self):
        self.Q = random.randint(5, 7)
        self.e = random.randint(2, 4)
        self.vec = genHaarRandomState(random.randrange(10000),
                                      self.Q,
                                      self.e)
        return super().setUp()

    def _krdm(self, k: int):
        fast_k_rdm = fast_compute_k_rdm(k, self.vec, False)
        for _ in range(100):
            idxs = tuple(random.randrange(self.Q) for _ in range(2*k))
            self.assertAlmostEqual(
                fast_k_rdm[idxs],
                slow_compute_k_rdm(self.vec, idxs)
            )

    def _wedge(self, Q: int, p: int, q: int):
        left_tensor = np.random.random(tuple(Q for _ in range(2*p)))\
            + 1j*np.random.random(tuple(Q for _ in range(2*p)))
        right_tensor = np.random.random(tuple(Q for _ in range(2*q)))\
            + 1j*np.random.random(tuple(Q for _ in range(2*q)))
        left_index_ranks = (p, p)
        right_index_ranks = (q, q)
        fast_ans = fast_wedge(left_tensor, right_tensor,
                              left_index_ranks, right_index_ranks, False),
        slow_ans = slow_wedge(left_tensor, right_tensor,
                              left_index_ranks, right_index_ranks)
        self.assert_(np.allclose(fast_ans, slow_ans))
        topM = fast_wedge_topM(left_tensor, right_tensor,
                               left_index_ranks, right_index_ranks, 10, False)
        for _, elem, ipiq, jpjq in topM:
            self.assertAlmostEqual(
                slow_ans[tuple(list(ipiq)+list(jpjq))], elem)

    def test_1rdm(self):
        self._makeVec()
        self._krdm(1)
        self._makeVec()
        self._krdm(1)

    def test_2rdm(self):
        self._makeVec()
        self._krdm(2)
        self._makeVec()
        self._krdm(2)

    def test_3rdm(self):
        self._makeVec()
        self._krdm(3)
        self._makeVec()
        self._krdm(3)

    def test_4rdm(self):
        self._makeVec()
        self._krdm(4)
        self._makeVec()
        self._krdm(4)

    def test_Q4p1q1wedge(self):
        self._wedge(4, 1, 1)

    def test_Q4p1q2wedge(self):
        self._wedge(4, 1, 2)

    def test_Q4p2q1wedge(self):
        self._wedge(4, 2, 1)

    def test_Q4p1q3wedge(self):
        self._wedge(4, 1, 3)

    def test_Q4p2q2wedge(self):
        self._wedge(4, 2, 2)

    def test_Q4p3q1wedge(self):
        self._wedge(4, 3, 1)

    def test_Q6p1q1wedge(self):
        self._wedge(6, 1, 1)

    def test_Q6p1q2wedge(self):
        self._wedge(6, 1, 2)

    def test_Q6p2q1wedge(self):
        self._wedge(6, 2, 1)

    def test_Q6p1q3wedge(self):
        self._wedge(6, 1, 3)

    def test_Q6p2q2wedge(self):
        self._wedge(6, 2, 2)

    def test_Q6p3q1wedge(self):
        self._wedge(6, 3, 1)


def test():
    runner = unittest.TextTestRunner(descriptions=True, verbosity=2)
    runner.run(unittest.makeSuite(Test))


if __name__ == '__main__':
    test()
