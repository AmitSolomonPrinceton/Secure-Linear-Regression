import numpy as np

from quantizer.quantizer import Quantizer

p = (2**25)-37
lx = 5
quantizer = Quantizer(p=p, lx=lx)
x = np.array([[-1.2, 0, -1.8, -5], [1, 2.2, 3.5, 4.999]])
xq = quantizer.quantize(x)
xhat = quantizer.dequantize(xq)
