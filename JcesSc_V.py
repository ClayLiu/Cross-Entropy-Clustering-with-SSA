import numpy as np 
import random
import matplotlib.pyplot as plt 
import slap_V
import math
# import NMI

class Jce_sSC_simple():
    def __init__(self, X : np.ndarray = None, label : np.ndarray = None, c : int = 3, slap_num : int = 100):
        """
        :param X: 样本数据 m * n, 单个数据为列向量 \n
        :param label: 标签 \n
        :param c: 类簇数目 \n
        :param slap_num: 樽海鞘数量 \n
        """

        self.X = X
        self.slap_num = slap_num
        self.n = X.shape[1]     # 样本数据容量
        self.c = c 
        self.X_bound = self._get_X_bound()
        print(self.X_bound)
        self.label = label
        # self.U = self._make_up_U()  # 相关系数矩阵 c * n
        # self.V = self._make_up_V() # 聚类中心 m * c
        self.punish_matrix = None  # 惩罚系数矩阵 n * n

        self.Slap_Swarm = slap_V.slap_swarm(
            slap_num =          slap_num, 
            V_generate_func =   self._make_up_V, 
            fitness_func =      self.Jce_sSC_slap,
            V_bound =           self.X_bound
        )
    
    def get_U_with_o_dis(self, V : np.ndarray) -> np.ndarray:
        """   
        根据 V 生成对应的 类隶属度矩阵 U(欧氏距离)
        """
        U = np.zeros((self.c, self.n))
        for i in range(self.n):
            
            # 算出该点与各个聚类中心的距离（欧式距离）
            o_distance = np.array([self.__get_o_distance__(self.X[:, i], V[:, j]) for j in range(self.c)]) 

            # 最值调转
            o_distance = np.max(o_distance) - o_distance + np.min(o_distance)

            # 归一化
            o_distance /= np.sum(o_distance)
            
            # 引入softmax 将使结果对樽海鞘模拟敏感
            # # softmax
            # o_distance = self.__softmax__(o_distance)
            
            U[:, i] = o_distance
        
        return U
    
    def get_U_with_m_dis(self, V : np.ndarray) -> np.ndarray:
        """   
        根据 V 生成对应的 类隶属度矩阵 U(曼哈顿距离)
        """
        U = np.zeros((self.c, self.n))
        for i in range(self.n):
            
            # 算出该点与各个聚类中心的距离（曼哈顿距离）
            m_distance = np.array([self.__get_m_distance__(self.X[:, i], V[:, j]) for j in range(self.c)]) 

            # 最值调转
            m_distance = np.max(m_distance) - m_distance + np.min(m_distance)

            # 归一化
            m_distance /= np.sum(m_distance)
                        
            U[:, i] = m_distance
        
        return U
    
    # def get_U_according_parper(self, V : np.array) -> np.ndarray:
    #     """   
    #     根据 V 生成对应的 类隶属度矩阵 U
    #         (老师论文的算法)
    #     """
    #     """ 老师的算法是用来迭代的 """
    #     U = np.zeros((self.c, self.n))
    #     for i in range(self.c):
    #         for j in range(self.n):
    #             pass
        
    #     return U 
    
    def __get_o_distance__(self, vector_1 : np.ndarray, vector_2 : np.ndarray):
        return np.sqrt(np.sum((vector_1 - vector_2) ** 2))
    def __get_m_distance__(self, vector_1 : np.ndarray, vector_2 : np.ndarray):
        return np.sum(np.abs(vector_1 - vector_2))
        
    def __softmax__(self, array : np.ndarray):
        max_value = np.max(array)
        exp_arr = np.exp(array - max_value)
        return exp_arr / np.sum(exp_arr)

    def _get_unique_pairs(self, n : int, n_pair : int, pick = 'random'):
        ''' 生成成对挑选矩阵，避免np.random.choice 挑选到重复的条目 '''
        random_num = 0
        pairs = np.random.choice(np.arange(0, n), size=(n_pair, 2))
        pair_dict = {}

        for pair in pairs:
            pair = np.sort(pair)
            key = str(pair[0])

            if key in pair_dict:
                # 判断是否重复选择条目了
                if pair[1] in pair_dict[key]:   # 重复了
                    if pick == 'random':
                        # 随机挑选
                        while random_num in pair_dict[key]: # 随机挑选直到随机数不重复
                            random_num = random.randint(0, n)
                        pair[1] = random_num
                        pair_dict[key].add(random_num)
                    elif pick == 'order':
                        # 按序挑选
                        next_num = pair[1] + 1
                        while next_num in pair_dict[key]:
                            next_num += 1
                        pair[1] = next_num
                        pair_dict[key].add(next_num)
                else:
                    pair_dict[key].add(pair[1])
            else:
                pair_dict[key] = {pair[1]}

        return pairs

    def pick_same_cluster(self, n_pair : int):
        ''' 先挑选同类的 '''
        pairs = np.zeros((n_pair, 2), dtype=np.int32)
        clusters = set(self.label)
        
        while n_pair:
            clusters_choose = random.randint(0 , len(clusters) - 1)
            this_cluster = np.where(self.label == clusters_choose)[0]
            
            pairs[n_pair - 1, :] = np.random.choice(this_cluster, size=(1, 2))
            n_pair -= 1
        
        return pairs

    def make_up_punish_matrix(self, gamma : tuple, n_pair = 0) -> np.ndarray:
        n = self.n
        punish_matrix = np.zeros((n, n))

        pairs = self._get_unique_pairs(n, n_pair)
        # print(pairs)
        for pair in pairs:
            j = pair[0]
            k = pair[1]
            if label[j] == label[k]:
                punish_matrix[j][k] = gamma[0]
            else:
                punish_matrix[j][k] = -gamma[0]
                
        punish_matrix += punish_matrix.T

        for i in range(n):
            punish_matrix[i][i] = gamma[1]

        return punish_matrix
    
    def make_up_punish_matrix_only_ML(self, gamma : tuple, n_pair = 0) -> np.ndarray:
        ''' 先挑选同类的 '''
        n = self.n
        punish_matrix = np.zeros((n, n))

        pairs = self.pick_same_cluster(n_pair)
        for pair in pairs:
            j = pair[0]
            k = pair[1]
            punish_matrix[j][k] = gamma[0]
            punish_matrix[k][j] = gamma[0]   

        for i in range(n):
            punish_matrix[i][i] = gamma[1]

        return punish_matrix

    def _make_up_V(self) -> np.ndarray:
        ''' 在数据范围中随机生成类中心点 '''
        V = np.zeros((self.X.shape[0], self.c))
        for i in range(self.X.shape[0]):
            for j in range(self.c):
                V[i][j] = (self.X_bound[i][1] - self.X_bound[i][0]) * random.random() + self.X_bound[i][0]
        return V
    
    def _get_X_bound(self):
        _min = np.min(self.X, axis=1)
        _max = np.max(self.X, axis=1)

        bound = [(l, u) for l, u in zip(_min, _max)]    # 转成元组列表防止被改动
        return bound

    def _H(self, U : np.ndarray, j : int, k : int):
        # 交叉熵部分
        miu_1 = U[:, j]
        miu_2 = U[:, k]
        ''' old version '''
        # return - np.sum(miu_1 * np.log(miu_2))

        ''' new version '''
        where = miu_1 > 0.0001
        entropy = miu_1[where] * np.log(miu_2[where])
        return - np.sum(entropy)

    def Jce_sSC(self, U, V):

        part_one = 0
        for i in range(self.c):
            for j in range(self.n):
                vector_temp = self.X[:, j] - V[:, i]
                part_one += U[i][j] * np.sum(vector_temp * vector_temp)
        
        part_two = 0
        for j in range(self.n):
            for k in range(self.n):
                if self.punish_matrix[j][k] != 0:
                    part_two += self.punish_matrix[j][k] * self._H(U, j, k)

        return part_one - part_two

    def Jce_sSC_slap(self, slap_i : slap_V.Slap):
        return self.Jce_sSC(self.get_U_with_o_dis(slap_i.V), slap_i.V)

    def iteration(self, iter_num):
        self.Slap_Swarm.iteration(iter_num)
        return self.Slap_Swarm.F

    def get_cluster(self, V : np.ndarray):
        """   
        根据 V 生成对应的 聚类结果
        """
        U = self.get_U_with_o_dis(V)
        return np.argmax(U, axis=0) + 1

    def SS_refresh(self):

        self.Slap_Swarm = slap_V.slap_swarm(
            slap_num =          self.slap_num, 
            V_generate_func =   self._make_up_V, 
            fitness_func =      self.Jce_sSC_slap,
            V_bound =           self.X_bound
        )

def test_data(m: int, n: int):
    x = np.ones((m, n))
    y = np.ones(n, dtype=np.int32)
    x0 = np.random.normal(10 * x, 0.2)
    x1 = np.random.normal(-3 * x, 0.2)


    x[1] = 0
    x2 = np.random.normal(4 * x, 0.2)

    X = np.hstack((x0, x1, x2))
    y = np.hstack((y, y * 2, y * 3))
    
    return X, y

if __name__ == '__main__':
    X, y = test_data(2, 20)
    
    plt.scatter(X[0], X[1], c=y, cmap='RdYlGn')
    plt.show()

    a = Jce_sSC_simple(X, y, (1, 1), 30)

    best = a.iteration(80)
    best_v = best.V
    cluster = a.get_cluster(best_v)

    # print(NMI.NMI(y, cluster))

    ''' 输出聚类结果 '''
    # plt.cla()
    # plt.scatter(X[0], X[1], c=cluster, cmap = 'RdYlGn')
    # plt.scatter(best_v[0], best_v[1], c=np.arange(1, c + 1), cmap = 'RdYlGn')


    ''' 输出适应度函数对比图 '''    
    # F_fitness = np.array(a.Slap_Swarm.F_fitness)
    # head_fitness = np.array(a.Slap_Swarm.head_fitness)
    # x = np.arange(1, len(head_fitness) + 1)
    # plt.plot(x, F_fitness, color = 'red')
    # plt.plot(x, head_fitness, color = 'blue')
    
    # for f, h in zip(F_fitness, head_fitness):
    #     print(f, h)

    # print(best.get_fitness(a.Jce_sSC_slap))
    # print('hello world')
    
    # plt.show()
