import numpy as np

from utils import matmul_gfq, round


class Quantizer():
    """
    A quantizer class.

    Args:
        p (int):
            The prime number to use.
        lx (int):
            Quantization loss parameter.
    """
    def __init__(self, p: int, lx: int):
        self._p = p
        self._lx = lx
    
    @property
    def p(self) -> int:
        return self._p
    
    @p.setter
    def p(self, p: int) -> None:
        self._p = p

    @property
    def lx(self) -> int:
        return self._lx
    
    @lx.setter
    def lx(self, lx: int) -> None:
        self._lx = lx

    def quantize(self, x: np.ndarray) -> np.ndarray:
        """
        This function quantizes a matrix x.

        Args:
            x (np.ndarray):
                np.ndarray object to quantize.
        
        Returns:
            x, quantized elementwise.
        """
        x = round((2**self.lx)*x)
        x = np.where(x<0, x+self.p, x)
        return x

    def dequantize(self, x: np.ndarray) -> np.ndarray:
        """
        This function dequantizes x.

        Args:
            x (np.ndarray):
                np.ndarray object to dequantize.
        
        Returns:
            x, dequantized elementwise.
        """
        x = np.where((x<self.p) & (x>=0.5*(self.p-1)), x-self.p, x)
        return 2**(-self.lx)*x

    def multiply(self, A: np.array, B: np.array) -> np.array:
        """
        This function quantizes A and B, multiplies them, and returns the dequantized product.
        """
        Aq = self.quantize(A)
        Bq = self.quantize(B)
        prod = matmul_gfq(Aq, Bq)
        return self.dequantize(prod)
