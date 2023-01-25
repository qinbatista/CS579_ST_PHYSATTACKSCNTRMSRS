import numpy as np

a = np.array([[True, False], [True, False]])
a[:, 0] = np.where(a[:, 0], 1, a[:, 0])
