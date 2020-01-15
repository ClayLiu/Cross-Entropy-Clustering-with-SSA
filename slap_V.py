import math
import random
import numpy as np 
import array
from tqdm import tqdm
import matplotlib.pyplot as plt 
# import multiprocessing as mp 
# import processing

class Slap():
    def __init__(self, V : np.ndarray, V_bound):
        
        self.m = V.shape[0]
        self.c = V.shape[1]

        self.V = V

        self.V_bound = V_bound
        self.j_round = [j - i for i, j in self.V_bound] # 记录X在第j维的取值跨度(最高 - 最低)，避免重复计算

    def __repr__(self):
        return self.V.__repr__()
    
    def step_head(self, F, c_1 : float):
        self.V = F.V
        
        # 对 V 迭代
        for i in range(self.c):
            for j in range(self.m):
                c_3 = random.random()
        
                if c_3 < 0.5:
                    c_1 = -c_1
                
                c_2 = random.random()
                self.V[j][i] += c_1 * (self.j_round[j] * c_2 + self.V_bound[j][0])

    def step(self, prev):
        self.V += prev.V
        self.V /= 2

    def amend(self):
        # 对 V 进行返回范围
        for i in range(self.c):
            for j in range(self.m):
                if self.V[j][i] < self.V_bound[j][0]:
                    self.V[j][i] = self.V_bound[j][0]
                if self.V[j][i] > self.V_bound[j][1]:
                    self.V[j][i] = self.V_bound[j][1]

    def copy(self):
        new_slap = Slap(self.V.copy(), self.V_bound)
        return new_slap

class slap_swarm():
    def __init__(self, slap_num = 30, V_generate_func = None, fitness_func = None, V_bound = None):
        self.slap_num = slap_num
        self.fitness_func = fitness_func
        self.F = Slap(V_generate_func(), V_bound)

        self.fitness = np.zeros(slap_num)

        temp_list = []
        for i in range(slap_num):
            temp_list.append(Slap(V_generate_func(), V_bound))

        self.Slap_Swarm = np.array(temp_list)
        temp_list.clear()

        self.F_fitness = []
        self.head_fitness = []
        
    def get_fitness(self):
        self.fitness = np.array([self.fitness_func(slap_i) for slap_i in self.Slap_Swarm])
    
    def get_fitness_mp(self):
        # import multiprocessing as mp 
        # if __name__ == 'slap_V':
        #     pool = mp.Pool()
        #     self.fitness = np.array(pool.map(self.fitness_func, self.Slap_Swarm))
        self.fitness = np.array(array.array('d', map(self.fitness_func, self.Slap_Swarm)))

    def iteration(self, iter_num):
        
        for j in tqdm(range(1, iter_num + 1)):
            c_1 = 2 * math.exp(-(4 * j / iter_num)**2)

            self.get_fitness()
            # print(self.fitness)
            best_index = np.argmin(self.fitness)
            # print(self.fitness)
            ''' F 停留 '''
            # if best_fitness == None:
            #     best_fitness = fitness[best_index]
            #     self.F = self.Slap_Swarm[best_index].copy()
            
            # # F 停留
            # if best_fitness > fitness[best_index]:
            #     self.F = self.Slap_Swarm[best_index].copy()
            #     best_fitness = fitness[best_index]
            
            ''' F 不停留 '''    # F 不应停留，F停留则会陷入局部最优解
            self.F = self.Slap_Swarm[best_index]
            best_fitness = self.fitness[best_index]
            # print(self.F)

            # 迭代
            self.Slap_Swarm[0].step_head(self.F.copy(), c_1)
            
            for i in range(1, self.slap_num):
                self.Slap_Swarm[i].step(self.Slap_Swarm[i - 1])

            # 把超出范围的樽海鞘拉回范围内
            for i in range(self.slap_num):
                self.Slap_Swarm[i].amend()
            
            ''' 记录适应度函数 '''
            self.F_fitness.append(best_fitness)
            self.head_fitness.append(self.fitness_func(self.Slap_Swarm[0]))


        
        
        

