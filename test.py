import numpy as np
from statistics import median

a = np.array([[1, 2, 3, 4], [5, 5, 7, 7]])
print(a)
print(np.median(a, -1))
b = a - np.median(a, -1).reshape(2, 1)
print(b)