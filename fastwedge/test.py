import random
import unittest
import numpy as np
from fastwedge._slow_jw_mul_vec import slow_jw_mul_vec
from fastwedge._slow_kRDM import slow_compute_k_rdm
from fastwedge._slow_wedge import slow_wedge
from fastwedge.kRDM import fast_compute_k_rdm, _make_jordan_wigners_mul_vec
from fastwedge.wedgeProduct import fast_wedge
from fastwedge.wedgeTopM import fast_wedge_topM


class Test(unittest.TestCase):
    def _jw(self, Q: int, k: int, vec: np.ndarray):
        slow_jw = sorted(slow_jw_mul_vec(Q, k, vec).items())
        fast_jw = sorted(_make_jordan_wigners_mul_vec(Q, k, vec).items())
        for (s_key, s_val), (f_key, f_val) in zip(slow_jw, fast_jw):
            self.assertEqual(s_key, f_key)
            self.assertTrue(np.allclose(s_val, f_val))

    def _krdm(self, Q: int, k: int):
        vec = np.random.random(2**Q) + 1j*np.random.random(2**Q)
        self._jw(Q, k, vec)
        fast_k_rdm = fast_compute_k_rdm(k, vec, False)
        for _ in range(100):
            idxs = tuple(random.randrange(Q) for _ in range(2*k))
            self.assertAlmostEqual(
                fast_k_rdm[idxs],
                slow_compute_k_rdm(vec, idxs)
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
        self.assertTrue(np.allclose(fast_ans, slow_ans))
        topM = fast_wedge_topM(left_tensor, right_tensor,
                               left_index_ranks, right_index_ranks, 10, False)
        for _, elem, ipiq, jpjq in topM:
            self.assertAlmostEqual(
                slow_ans[tuple(list(ipiq)+list(jpjq))], elem)

    def test_Q1_1rdm(self):
        self._krdm(1, 1)

    def test_Q2_1rdm(self):
        self._krdm(2, 1)

    def test_Q3_1rdm(self):
        self._krdm(3, 1)

    def test_Q2_2rdm(self):
        self._krdm(2, 2)

    def test_Q3_2rdm(self):
        self._krdm(3, 2)

    def test_Q4_2rdm(self):
        self._krdm(4, 2)

    def test_Q3_3rdm(self):
        self._krdm(3, 3)

    def test_Q4_3rdm(self):
        self._krdm(4, 3)

    def test_Q5_3rdm(self):
        self._krdm(5, 3)

    def test_Q4_4rdm(self):
        self._krdm(4, 4)

    def test_Q5_4rdm(self):
        self._krdm(5, 4)

    def test_Q6_4rdm(self):
        self._krdm(6, 4)

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
