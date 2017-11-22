import numpy as np
from scipy.special import expit as sigmoid
from sklearn.metrics import roc_auc_score
from multiprocessing import Pool
import concurrent.futures

def heaviside(x):
    g = 0
    epsilon = 1e-6
    if abs(x) < epsilon:
        g = 0.5
    elif x < 0:
        g = 0
    else:
        g = 1
    return g

def sum_rank_score(minor, major_data):
    #vfunc = np.vectorize(heaviside)
    #return np.sum(vfunc(minor - major_data))
    return np.sum(np.heaviside(minor - major_data, 0.5))

def lgbm_full_rank_score(minor_data, major_data):
    worker_num = 8
    minor_len = minor_data.shape[0]
    major_len = major_data.shape[0]
    begin = 0
    end = min(begin + worker_num, minor_len)
    sum_score = 0
    while begin < minor_len:
        with concurrent.futures.ThreadPoolExecutor(max_workers = worker_num) as executor:
            future_predict = {executor.submit(sum_rank_score, minor_data[ind],
                        major_data): ind for ind in range(begin, end)}
            for future in concurrent.futures.as_completed(future_predict):
                ind = future_predict[future]
                try:
                    sum_score += future.result()
                except Exception as exc:
                    print('%dth feature generated an exception: %s' % (ind, exc))
        if end % 1000 == 0:
            print('Process pair: {} * Q'.format(end))
        begin = end
        end = min(begin + worker_num, minor_len)

    return sum_score


def Get_Pair_data(data, label):
    minor = 0
    major = 1
    true_num = np.sum(label == 1)
    if true_num * 2 < len(label):
        minor = 1
        major = 0
    minor_ind = [i for i in range(len(label)) if label[i] == minor]
    major_ind = [i for i in range(len(label)) if label[i] == major]
    minor_data = data[minor_ind, :]
    major_data = data[major_ind, :]
    P = len(minor_data)
    Q = len(major_data)
    print("P: {} Q: {}".format(P, Q))
    # lgbm_rank = lgbm_full_rank_score(minor_data[:, -1], major_data[:, -1])
    # auc = roc_auc_score(label, sigmoid(data[:, -1]))
    # lgbm_rank /= float(P * Q)
    #print('lgbm full rank: {} roc_auc: {}'.format(lgbm_rank, auc))
    #exit(0)
    pair_len = 5 * Q
    pair_data = np.zeros((pair_len, 2, data.shape[1]), dtype = 'float32')
    for k in range(pair_len):
        pair_data[k][0] = minor_data[k % P].astype(np.float32)
        pair_data[k][1] = major_data[k % Q].astype(np.float32)
    sub_data = pair_data[:, 0, -1] - pair_data[:, 1, -1]
    #for i in range(100):
    #    factor = 1. + 0.1 * i
    #    lgbm_rank = np.mean(sigmoid(factor * sub_data))
    #    print('lgbm full rank: {} roc_auc: {} factor: {}'.format(lgbm_rank, auc, factor))
    #exit(0)
    print('shape: {} dtype: {}'.format(pair_data.shape, pair_data.dtype))
    return pair_data

def rank_score(minor_preds, major_preds, act = 'sigmoid', factor = 1):
    if act == 'sigmoid':
        score = np.mean(sigmoid(factor * (minor_preds - major_preds)))
    elif act == 'heaviside':
        score = np.mean(np.heaviside(minor_preds - major_preds, 0.5))
    return score
