import numpy as np
from scipy.special import expit as sigmoid

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
    pair_len = 5 * Q
    print("P: {} Q: {}".format(P, Q))
    pair_data = np.zeros((pair_len, 2, data.shape[1]), dtype = 'float32')
    for k in range(pair_len):
        pair_data[k][0] = minor_data[k % P].astype(np.float32)
        pair_data[k][1] = major_data[k % Q].astype(np.float32)
    print('shape: {} dtype: {}'.format(pair_data.shape, pair_data.dtype))
    return pair_data

def rank_score(minor_preds, major_preds):
    score = np.mean(sigmoid(minor_preds - major_preds))
    return score
