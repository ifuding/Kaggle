
from sklearn import metrics, preprocessing, pipeline, \
    feature_extraction, decomposition, model_selection
import sklearn
import pandas as pd
import numpy as np
from time import gmtime, strftime

nrs = np.random.RandomState(0)

def lcc_sample(lables, preds, input_data, C = 1):
    """
    Param:
    labels shape: (n_sample,)
    preds shape: (n_sample,)
    input_data shape: (n_sample, feature_dim)
    C: times based on accepte_rate
    return:
    data after sampling
    """
    accept_rate = np.abs(lables - preds) * C
    bernoulli_z = nrs.binomial(1, np.clip(accept_rate, 0, 1))
    select_ind = [i for i in range(bernoulli_z.shape[0]) if bernoulli_z[i] == 1]
    sample_data = input_data[select_ind, :]
    sample_lables = lables[select_ind]
    weight = np.ones(len(lables))
    adjust_weight_ind = [i for i in range(len(accept_rate)) if accept_rate[i] > 1]
    weight[adjust_weight_ind] = accept_rate[adjust_weight_ind]
    weight = weight[select_ind]
    print('-----LCC Sampling Before All: {} Pos: {} Neg: {}'.format(len(lables), np.sum(lables == 1), np.sum(lables == 0)))
    print('-----LCC Sampling After All: {} Pos: {} Neg: {}'.format(len(sample_lables), np.sum(sample_lables == 1), np.sum(sample_lables == 0)))
    print('-----LCC Sampling Rate: {}'.format(float(len(sample_lables)) / float(len(lables))))
    return sample_data, sample_lables, weight
