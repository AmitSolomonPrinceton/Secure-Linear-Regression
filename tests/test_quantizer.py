import unittest

import numpy as np
import numpy.testing as npt

from quantizer.quantizer import Quantizer


class TestGeneral(unittest.TestCase):
    def setUp(self):
        self.p = (2**25)-37
        self.lx = 16
        self.quantizer = Quantizer(p=self.p, lx=self.lx)
    
    def test_general(self):
        x = np.arange(-10, 11)
        xq = self.quantizer.quantize(x)
        xq_result = self.quantizer.dequantize(xq)
        npt.assert_allclose(x, xq_result)

class TestZeroLx(unittest.TestCase):
    def setUp(self):
        self.p = (2**25)-37
        self.lx = 0
        self.quantizer = Quantizer(p=self.p, lx=self.lx)

    def test_quantize(self):
        x = np.array([[0.5, 1, 3.2, 0], [-0.2, -0.5, -1.0, -3.3]])
        xq_expected = np.array([[1, 1, 3, 0], [0, 0, -1+self.p, -3+self.p]])
        xq_result = self.quantizer.quantize(x)
        npt.assert_equal(xq_expected, xq_result)

    def test_dequantize(self):
        x = np.array([0, 1, 0.5*(self.p-3), 0.5*(self.p-1), 0.5*(self.p+1), self.p-1])
        xq_expected = np.concatenate((x[:3], x[3:]-self.p))
        xq_result = self.quantizer.dequantize(x)
        npt.assert_equal(xq_expected, xq_result)
