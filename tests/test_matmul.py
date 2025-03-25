import unittest

import numpy as np
import numpy.testing as npt

from utils import matmul_gfq


class TestGeneral(unittest.TestCase):
    def setUp(self):
        self.p = 8
    
    def test_general(self):
        A = np.array([[1,2,3],[4,5,6]])
        B = np.array([[1,-1],[1,1],[2,0]])
        expected = np.array([[1,1],[5,1]])
        result = matmul_gfq(A=A, B=B, p=self.p)
        npt.assert_equal(expected, result)