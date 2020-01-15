import numpy as np
import math
from sklearn import metrics

def NMI(A,B):
    # len(A) should be equal to len(B)
    total = len(A)
    A_ids = set(A)
    B_ids = set(B)

    #Mutual information
    MI = 0
    eps = 1.4e-45
    for idA in A_ids:
        for idB in B_ids:
            idAOccur = np.where(A==idA)
            idBOccur = np.where(B==idB)
            idABOccur = np.intersect1d(idAOccur,idBOccur)
            px = len(idAOccur[0]) / total
            py = len(idBOccur[0]) / total
            pxy = len(idABOccur) / total
            MI = MI + pxy * math.log(pxy / (px * py) + eps)
    # Normalized Mutual information
    Hx = 0
    for idA in A_ids:
        idAOccurCount = 1.0*len(np.where(A==idA)[0])
        Hx = Hx - (idAOccurCount/total)*math.log(idAOccurCount/total+eps)
    Hy = 0
    for idB in B_ids:
        idBOccurCount = 1.0*len(np.where(B==idB)[0])
        Hy = Hy - (idBOccurCount/total)*math.log(idBOccurCount/total+eps)
    MIhat = 2.0*MI/(Hx+Hy)
    return MIhat

def H(x : np.ndarray):
    where = x > 0.0001
    entropy = x[where] * np.log(x[where])
    return - np.sum(entropy)

def my_NMI(A, B):
    length = len(A)
    set_A = set(A)
    set_B = set(B)

    p_a = np.array([np.sum(A == i) / length for i in set_A])
    p_b = np.array([np.sum(B == i) / length for i in set_B])

    H_a = H(p_a)
    H_b = H(p_b)

    eps = 1e-10
    MI = 0
    for i, x in enumerate(set_A):
        a_equals_x = np.where(A == x)
        for j, y in enumerate(set_B):
            
            b_equals_y = np.where(B == y)

            joint = np.intersect1d(a_equals_x, b_equals_y)
            joint_p = len(joint) / length
            MI += joint_p * math.log(joint_p / (p_a[i] * p_b[j]) + eps)
    
    NMI = 2 * MI / (H_a + H_b)
    return NMI


if __name__ == '__main__':
    A = np.array([1,1,1,1,1,1,2,2,2,2,2,2,3,3,3,3,3])
    B = np.array([1,2,1,1,1,1,1,2,2,2,2,3,1,1,3,3,3])
    
    # print(metrics.normalized_mutual_info_score(A, B))
    print(my_NMI(A, B))
    # print(NMI(A, B))