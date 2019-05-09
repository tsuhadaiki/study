"""
データセットを用意するためのモジュール
>>> import datasets
>>> X, Y = datasets.load_liner_example1()
>>> X[0]
array([1, 4])
"""

import numpy as np

def load_liner_example1():
    X = np.array([[1,4],[1,8],[1,13],[1,17]]) 
    Y = np.array([7, 10, 11, 14])
    return X, Y

if __name__ == '__main__':
    import doctest
    doctest.testmod()