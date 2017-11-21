import numpy as np


def Get_Pair_data(data, label):
    minor = 0
    major = 1
    true_num = np.sum(label == 1)
    if true_num * 2 < len(label):
        minor = 1
        major = 0
    minor_data = [data[i] for i in range(len(label)) if label[i] == minor]
    major_data = [data[i] for i in range(len(label)) if label[i] == major]
    P = len(minor_data)
    Q = len(major_data)
    print("P: {} Q: {}".format(P, Q))
    pair_data = np.zeros((Q, 2, data.shape[1]))
    for k in range(Q):
        pair_data[k][0] = minor_data[k % P]
        pair_data[k][1] = major_data[k]
    return pair_data
