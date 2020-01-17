from numba import cuda
import numpy as np 
import slap_V
import math

@cuda.jit(device = True)
def __get_o_distance__(vector_1 : np.ndarray, vector_2 : np.ndarray):
    _sum = 0
    for i in range(vector_1.shape[0]):
        _sum += (vector_1[i] - vector_2[i]) ** 2

    return math.sqrt(_sum)

@cuda.jit(device = True)
def __get_m_distance__(self, vector_1 : np.ndarray, vector_2 : np.ndarray):
    _sum = 0
    for i in range(vector_1.shape[0]):
        _sum += vector_1[i] - vector_2[i]

    return _sum

print(__get_m_distance__)

@cuda.jit(device = True)
def get_U_with_dis(X : np.ndarray, V : np.ndarray, U : np.ndarray, distance_type):
        """   
        根据 V 生成对应的 类隶属度矩阵 U
        """
        c = V.shape[1]
        n = X.shape[1]
        for i in range(n):
            
            # 算出该点与各个聚类中心的距离
            if distance_type == 'o':
                distance = np.array([__get_o_distance__(X[:, i], V[:, j]) for j in range(c)]) 
            else:
                distance = np.array([__get_m_distance__(X[:, i], V[:, j]) for j in range(c)]) 
            # 最值调转
            distance = np.max(distance) - distance + np.min(distance)

            # 归一化
            distance /= np.sum(distance)
            
            # 引入softmax 将使结果对樽海鞘模拟敏感
            # # softmax
            # distance = __softmax__(distance)
            
            U[:, i] = distance
        return U

@cuda.jit(device = True)
def _H(self, U : np.ndarray, j : int, k : int):
    # 交叉熵部分
    miu_1 = U[:, j]
    miu_2 = U[:, k]
    ''' old version '''
    # return - np.sum(miu_1 * np.log(miu_2))

    ''' new version '''
    where = miu_1 > 0.00001
    entropy = miu_1[where] * np.log(miu_2[where])
    return - np.sum(entropy)

@cuda.jit(device = True)
def Jce_sSC(U, V, X, punish_matrix, c, n):

    part_one = 0
    for i in range(c):
        for j in range(n):
            vector_temp = X[:, j] - V[:, i]
            part_one += U[i][j] * np.sum(vector_temp * vector_temp)
    
    part_two = 0
    for j in range(n):
        for k in range(n):
            if punish_matrix[j][k] != 0:
                part_two += punish_matrix[j][k] * _H(U, j, k)

    return part_one - part_two

@cuda.jit
def get_fitness_with_cuda(VArray, U, punish_matrix, X, fitness_result, c, n, array_length):
    thread_id = cuda.threadIdx.x + cuda.blockIdx.x * cuda.blockDim.x 
    
    if thread_id < array_length:
        # U = get_U_with_dis(X, VArray[thread_id], distance_type='o')
        fitness_result[thread_id] = Jce_sSC(
            get_U_with_dis(X, VArray[thread_id], U, 'o'), 
            VArray[thread_id], 
            X, 
            punish_matrix, 
            c, 
            n
        )