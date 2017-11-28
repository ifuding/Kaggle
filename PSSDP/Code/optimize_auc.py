import numpy as np
from scipy.special import expit as sigmoid
from sklearn.metrics import roc_auc_score
from multiprocessing import Pool
import concurrent.futures
from sklearn import preprocessing
import pandas as pd

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


def MinMaxNormalize(input_data, scale_factor = 65535):
    # pd.DataFrame(input_data[1:1000]).to_csv('float_data', index = False)
    min_max_scaler = preprocessing.MinMaxScaler()
    normalize_data = min_max_scaler.fit_transform(input_data)
    normalize_data = (normalize_data * scale_factor).astype(np.uint8)
    # pd.DataFrame(normalize_data[1:1000]).to_csv('normalize_data', index = False)
    # exit(0)
    return normalize_data


def copy_pair(target_data, minor_data, major_data, minor_ind, pair_avg):
    # q_indice = np.arange(major_data.shape[0])
    # np.random.shuffle(q_indice)
    begin = minor_ind * pair_avg
    end = begin + pair_avg
    q_range = np.arange(begin, end) % major_data.shape[0]
    target_data[begin:end, 0] = minor_data[minor_ind].astype(np.uint8) #float32)
    target_data[begin:end, 1] = major_data[q_range].astype(np.uint8) #float32)
    #re = np.zeros((pair_avg, 2, minor_data.shape[1]), dtype = np.float32)
    #re[:, 0] = minor_data[minor_ind].astype(np.float32)
    #re[:, 1] = major_data[q_indice[:pair_avg]].astype(np.float32)
    return True #re


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
    #pair_len = 5 * Q
    #pair_data = np.zeros((pair_len, 2, data.shape[1]), dtype = 'float32')
    #for k in range(pair_len):
    #    pair_data[k][0] = minor_data[k % P].astype(np.float32)
    #    pair_data[k][1] = major_data[k % Q].astype(np.float32)
    data = MinMaxNormalize(data, 255)
    pair_avg = 500
    pair_len = pair_avg * P
    pair_data = np.zeros((pair_len, 2, data.shape[1]), dtype = np.uint8) #float32)
    q_indice = np.arange(Q)
    #i = 0
    #for k in range(P):
    #    np.random.shuffle(q_indice)
    #    pair_data[i:i+pair_avg, 0] = minor_data[k].astype(np.float32)
    #    pair_data[i:i+pair_avg, 1] = major_data[q_indice[:pair_avg]].astype(np.float32)
    #    i += pair_avg
    # pair_data = []
    worker_num = 8
    begin = 0
    end = min(begin + worker_num, P)
    while begin < P:
        with concurrent.futures.ThreadPoolExecutor(max_workers = worker_num) as executor:
            future_predict = {executor.submit(copy_pair, pair_data, minor_data,
                major_data, ind, pair_avg): ind for ind in range(begin, end)}
            for future in concurrent.futures.as_completed(future_predict):
                ind = future_predict[future]
                try:
                    re = future.result()
                    # pair_data.append(re)
                except Exception as exc:
                    print('%dth feature generated an exception: %s' % (ind, exc))
        if end % 500 == 0:
            print('Process pair: {} * {}'.format(end, pair_avg))
        begin = end
        end = min(begin + worker_num, P)
    # pd.DataFrame(pair_data.reshape((pair_len, -1))).to_csv('pair_test', index = False)
    print('shape: {} dtype: {}'.format(pair_data.shape, pair_data.dtype))
    return pair_data


def rank_score(minor_preds, major_preds, act = 'sigmoid', factor = 1):
    if act == 'sigmoid':
        score = np.mean(sigmoid(factor * (minor_preds - major_preds)))
    elif act == 'heaviside':
        score = np.mean(np.heaviside(minor_preds - major_preds, 0.5))
    return score
