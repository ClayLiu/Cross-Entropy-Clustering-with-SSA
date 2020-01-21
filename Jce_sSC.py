import numpy as np 
from Common import *

def _H(U : np.ndarray, j : int, k : int):
    # 交叉熵部分
    miu_1 = U[:, j]
    miu_2 = U[:, k]
    ''' old version '''
    # return - np.sum(miu_1 * np.log(miu_2))

    ''' new version '''
    where_1 = miu_1 > threshold_value
    where_2 = miu_2 > threshold_value
    where = np.array([bool_1 and bool_2 for bool_1, bool_2 in zip(where_1, where_2)])
    entropy = miu_1[where] * np.log(miu_2[where])
    return - np.sum(entropy)

def Jce_sSC(X : np.ndarray, U : np.ndarray, V : np.ndarray, punish_matrix : np.ndarray):
    c = U.shape[1]
    n = X.shape[1]
    part_one = 0
    for i in range(c):
        for j in range(self.n):
            vector_temp = self.X[:, j] - V[:, i]
            part_one += U[i][j] * np.sum(vector_temp * vector_temp)
    
    part_two = 0
    for j in range(self.n):
        for k in range(self.n):
            if punish_matrix[j][k] != 0:
                part_two += punish_matrix[j][k] * self._H(U, j, k)

    return part_one - part_two