import random

import numpy as np
from sklearn.datasets import load_iris
from sklearn import metrics

import JcesSc_V
from NMI import my_NMI
import matplotlib.pyplot as plt

iris_data = load_iris()

X = iris_data.data
label = iris_data.target

# print(X.shape)
# print(label.shape)

# X, label = JcesSc_V.test_data(2, 20)

# 一种获得n对成对信息惩罚矩阵的实现，已弃用 2019年12月5日17:12:07
def get_n_pair_punish_matrix(original_punish_matrix : np.ndarray, n : int, dropout_way = 'order'):
    x_n = original_punish_matrix.shape[0]
    dropout_num = x_n * (x_n - 1) // 2 - n
    if dropout_num < 0:
        # raise 
        print('dropout is negetive')
        return original_punish_matrix

    punish_matrix = original_punish_matrix.copy()

    if dropout_way == 'random':
        ''' 随机丢弃 '''
        while dropout_num != 0:
            i = random.randint(0, x_n)
            j = random.randint(0, x_n)
            
            while punish_matrix[i][j] == 0 or i == j:
                i = random.randint(0, x_n)
                j = random.randint(0, x_n)

            punish_matrix[i][j] = 0
            dropout_num -= 1
    else:    
        ''' 按顺序丢弃 '''
        for i in range(x_n - 1):
            for j in range(i + 1, x_n):
                punish_matrix[i][j] = 0
                dropout_num -= 1
                if dropout_num == 0:
                    break
            if dropout_num == 0:
                break
                    
    punish_matrix += punish_matrix.T
    for i in range(x_n):
        punish_matrix[i][i] -= 1

    return punish_matrix

def get_fig(F_fitness, head_fitness, fig_name, show = True):
    plt.figure()
    plt.title('Fitness value in x_th interation')
    plt.xlabel('x_th interation')
    plt.ylabel('fitness value')
    
    x = np.arange(1, len(head_fitness) + 1)
    plt.plot(x, F_fitness, color = 'red')
    plt.plot(x, head_fitness, color = 'blue')

    plt.legend(['F_fitness', 'head_slap_fitness'])
    if show:
        plt.show()
    plt.savefig(fig_name)
    plt.close()

def jichahua(X : np.ndarray, axis = 0):
    
    if axis:
        X = X.T
    _min = np.min(X, axis = 0)
    _max = np.max(X, axis = 0)

    across = _max - _min
    for i in range(X.shape[0]):
        X[i, :] = (X[i] - _min) / across
    return X

def cal_n_pair_nmi():
    
    times = 10
    nmi_table = np.zeros((11, times))   # 记录每一次若干代迭代后的 NMI
    cluster_maker = JcesSc_V.Jce_sSC_simple(X = X, label = label, c = 4, slap_num = 30)
    print(X)
    nmi_for_every_time = np.zeros(times)
    
    # 成对信息，从 0 到 100 对尝试
    for i in range(0, 110, 10):
        
        # 重复十次实验
        for t in range(times):
            
            ''' 随机乱选old '''
            # cluster_maker.punish_matrix = cluster_maker.make_up_punish_matrix((0.01, 1), i)

            ''' 先选都是同类的 2020年1月15日17:32:55 '''
            cluster_maker.punish_matrix = cluster_maker.make_up_punish_matrix_only_ML((0.01, 1), i)

            best = cluster_maker.iteration(100)
            cluster = cluster_maker.get_cluster(best.V)

            # current_nmi = NMI(label, cluster)
            # current_nmi = metrics.normalized_mutual_info_score(label, cluster)
            current_nmi = my_NMI(label, cluster)
            print(current_nmi)
            nmi_for_every_time[t] = current_nmi
            
            fig_name = 'new_pun_mat/jichahuahou/fitness_value_in' + str(i) + 'pairs_' + str(t) + '.png'
            
            get_fig(
                cluster_maker.Slap_Swarm.F_fitness, 
                cluster_maker.Slap_Swarm.head_fitness, 
                fig_name, 
                show = False
            )
            cluster_maker.SS_refresh()
        
        nmi_table[i // 10, :] = nmi_for_every_time
    
    nmi_mean_table = np.sum(nmi_table, axis=1) / times

    nmi_table = nmi_table.reshape((1, -1))
    times_one = np.ones((1, times))
    nmi_x = times_one * 0
    for i in range(1, 11):
        nmi_x = np.hstack((nmi_x, times_one * i))
    nmi_x *= 10
    print(nmi_x.shape)
    print(nmi_table.shape)

    plt.figure()
    plt.xlabel('pair information number')
    plt.ylabel('NMI')
    plt.scatter(nmi_x[0], nmi_table[0])
    # plt.show()
    plt.savefig('new_pun_mat/jichahuahou/NMI值.png')
    plt.close()

    plt.figure()
    plt.xlabel('x_th time')
    plt.ylabel('NMI')
    plt.plot(np.arange(0, nmi_table.shape[1]), nmi_table[0])
    plt.savefig('new_pun_mat/jichahuahou/NMI值_逐次.png')
    plt.close()

    plt.figure()
    plt.xlabel('pair information number')
    plt.ylabel('mean NMI')
    plt.plot(np.arange(0, 11) * 10, nmi_mean_table)
    plt.savefig('new_pun_mat/jichahuahou/NMI均值.png')
    plt.close()

if __name__ == '__main__':
    X = jichahua(X)
    X = X.T
    cal_n_pair_nmi()
    
    # cluster_maker = JcesSc_V.Jce_sSC_simple(X = X, label = label, c = 4, slap_num = 30)
    # test = cluster_maker.pick_same_cluster(20)
    # print(test)

    # import os
    # os.system('shutdown -p')
    

# a = cluster_maker._get_unique_pairs(150, 10)
# print(a)