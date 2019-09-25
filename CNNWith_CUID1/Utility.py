import numpy as np
def norm(x):
        temp = x.T - np.mean(x.T, axis=0)
        temp = temp / np.std(temp, axis=0)
        return temp.T
