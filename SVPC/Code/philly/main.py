"""
This version has improvements based on new feature engg techniques observed from different kernels. Below are few of them:
- https://www.kaggle.com/graf10a/lightgbm-lb-0-9675
- https://www.kaggle.com/rteja1113/lightgbm-with-count-features?scriptVersionId=2815638
- https://www.kaggle.com/nuhsikander/lgbm-new-features-corrected?scriptVersionId=2852561
- https://www.kaggle.com/aloisiodn/lgbm-starter-early-stopping-0-9539 (Original script)
"""

import pandas as pd
import time
import numpy as np
import gc
from feature_engineer import gen_features
from feature_engineer import timer
import keras_train
from nfold_train import nfold_train, models_eval
import tensorflow as tf
import os
import shutil
from lcc_sample import neg_sample
from tensorflow.python.keras.models import load_model,Model
from sklearn import preprocessing
from tensorflow.python.keras.preprocessing.text import Tokenizer, text_to_word_sequence
from tensorflow.python.keras.preprocessing.sequence import pad_sequences
from CNN_Keras import get_word2vec_embedding
import lightgbm as lgb
import pickle
from RankGauss import rank_INT, rank_INT_DF
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from multiprocessing import Pool
import concurrent.futures
import glob

flags = tf.app.flags
flags.DEFINE_string('input-training-data-path', "../../Data/", 'data dir override')
flags.DEFINE_string('output-model-path', ".", 'model dir override')
flags.DEFINE_string('model_type', "k", 'model type')
flags.DEFINE_integer('nfold', 10, 'number of folds')
flags.DEFINE_integer('ensemble_nfold', 5, 'number of ensemble models')
flags.DEFINE_string('emb_dim', '5', 'term embedding dim')
flags.DEFINE_integer('epochs', 1, 'number of Epochs')
flags.DEFINE_integer('batch_size', 128, 'Batch size')
flags.DEFINE_integer('batch_interval', 1000, 'batch print interval')
flags.DEFINE_float("emb_dropout", 0, "embedding dropout")
flags.DEFINE_string('full_connect_hn', "64, 32", 'full connect hidden units')
flags.DEFINE_float("full_connect_dropout", 0, "full connect drop out")
flags.DEFINE_bool("stacking", False, "Whether to stacking")
flags.DEFINE_bool("load_stacking_data", False, "Whether to load stacking data")
flags.DEFINE_bool("debug", False, "Whether to load small data for debuging")
flags.DEFINE_bool("neg_sample", False, "Whether to do negative sample")
flags.DEFINE_bool("lcc_sample", False, "Whether to do lcc sample")
flags.DEFINE_integer("sample_C", 1, "sample rate")
flags.DEFINE_bool("load_only_singleCnt", False, "Whether to load only singleCnt")
flags.DEFINE_bool("log_transform", False, "Whether to do log transform")
flags.DEFINE_bool("split_train_val", False, "Whether to split train and validate")
flags.DEFINE_integer("train_eval_len", 25000000, "train_eval_len")
flags.DEFINE_integer("eval_len", 2500000, "eval_len")
flags.DEFINE_bool("test_for_train", False, "Whether to use test data for train")
flags.DEFINE_bool("search_best_iteration", True, "Whether to search best iteration")
flags.DEFINE_integer("best_iteration", 1, "best iteration")
flags.DEFINE_string('search_iterations', "100,1500,100", 'search iterations')
flags.DEFINE_string('input-previous-model-path', "../../Data/", 'data dir override')
flags.DEFINE_bool("blend_tune", False, "Whether to tune the blen")
flags.DEFINE_integer('vocab_size', 300000, 'vocab size')
flags.DEFINE_string('max_len', 100, 'max description sequence length')
# flags.DEFINE_integer('max_title_len', 100, 'max title sequence length')
flags.DEFINE_bool("load_wv_model", True, "Whether to load word2vec model")
flags.DEFINE_string('wv_model_type', "fast_text", 'word2vec model type')
flags.DEFINE_string('wv_model_file', "wiki.en.vec.indata", 'word2vec model file')
flags.DEFINE_integer('gram_embedding_dim', '300', 'gram embedding dim')
flags.DEFINE_string('kernel_size_list', "1,2,3", 'kernel size list')
flags.DEFINE_string('filter_size', "32", 'cnn filter size list')
flags.DEFINE_string('rnn_units', "0", 'rnn_units')
flags.DEFINE_bool("uniform_init_emb", False, "Whether to uniform init the embedding")
flags.DEFINE_bool("fix_wv_model", True, "Whether to fix word2vec model")
flags.DEFINE_bool("lgb_boost_dnn", True, "Whether to fix word2vec model")
flags.DEFINE_integer('lgb_ensemble_nfold', 5, 'number of lgb ensemble models')
flags.DEFINE_bool("load_from_pickle", True, "Whether to load from pickle")
flags.DEFINE_bool("vae_mse", True, "vae_mse")
flags.DEFINE_integer('vae_intermediate_dim', 100, 'vae_intermediate_dim')
flags.DEFINE_integer('vae_latent_dim', 100, 'vae_latent_dim')
flags.DEFINE_bool("load_from_vae", False, "load_from_vae")
flags.DEFINE_bool("predict_feature", False, "predict_feature")
flags.DEFINE_bool("aug_data", False, "aug_data")
FLAGS = flags.FLAGS

path = FLAGS.input_training_data_path
HIST_SIZE = 1000
SORT_LEN = 1000

top_cols = ['f190486d6', '58e2e02e6', 'eeb9cd3aa', '9fd594eec', '6eef030c1',
        '15ace8c9f', 'fb0f5dbfe', '58e056e12', '20aa07010', '024c577b9',
        'd6bb78916', 'b43a7cfd5', '58232a6fb', '1702b5bf0', '324921c7b', 
        '62e59a501', '2ec5b290f', '241f0f867', 'fb49e4212',  '66ace2992',
        'f74e8f13d', '5c6487af1', '963a49cdc', '26fc93eb7', '1931ccfdd', 
        '703885424', '70feb1494', '491b9ee45', '23310aa6f', 'e176a204a',
        '6619d81fc', '1db387535', 'fc99f9426', '91f701ba2',  '0572565c2',
        '190db8488',  'adb64ff71', 'c47340d97', 'c5a231d81', '0ff32eb98'] 

def select_pred(df, col):
    print ('Append column: ', col)
    file_name = glob.glob(path + '_' + col + '_2018_07_30_*.csv')[0]
    print(file_name)
    pred_col = pd.read_csv(file_name, index_col = 'ID')
    # exit(0)
    pred_col[col + '_p'] = pred_col['target']
    pred_col[col + '_new'] = df[col]
    select_s = (df[col] == 0) #& (df[col + '_p'] >= 319) & (df[col + '_p'] <= 319612000)
    pred_col[col + '_new'][select_s] = pred_col[col + '_p'][select_s]
    return pred_col

# def select_pred(s):
#     col = s.name
#     print ('Append column: ', col)
#     pred_col = pd.read_csv(path + '_' + col + '_2018_07_25_07.csv', index_col = 'ID')
#     pred_col[col + '_p'] = pred_col['target']
#     pred_col[col + '_new'] = s
#     select_s = (s == 0) #& (df[col + '_p'] >= 319) & (df[col + '_p'] <= 319612000)
#     pred_col[col + '_new'][select_s] = pred_col[col + '_p'][select_s]
#     return pred_col

def append_pred_columns(df):
    # select_pred(df, top_cols[0])
    MAX_WORKERS = 8
    cols = top_cols #[:5]
    print(cols)
    col_ind_begin = 0
    col_len = len(cols)
    while col_ind_begin < col_len:
        col_ind_end = min(col_ind_begin + MAX_WORKERS, col_len)
        with concurrent.futures.ThreadPoolExecutor(max_workers = MAX_WORKERS) as executor:
            future_predict = {executor.submit(select_pred, df, cols[ind]): ind for ind in range(col_ind_begin, col_ind_end)}
            for future in concurrent.futures.as_completed(future_predict):
                ind = future_predict[future]
                try:
                    pred_cols = future.result()
                    df[[cols[ind] + '_p', cols[ind] + '_new']] = pred_cols[[cols[ind] + '_p', cols[ind] + '_new']]
                except Exception as exc:
                    print('%dth feature normalize generate an exception: %s' % (ind, exc))
        col_ind_begin = col_ind_end
        if col_ind_begin % 100 == 0:
            print('Gen %d normalized features' % col_ind_begin)

def Min_Max_Normalize(s):
    return (s - s.min()) / (s.max() - s.min() + 1e-9)

def Avg_Std_Normalize(s):
    return (s - s.avg()) / s.std()

def Normalize(df, func):
    MAX_WORKERS = 8
    cols = list(df.columns.values)
    # print(cols)
    col_ind_begin = 0
    col_len = len(cols)
    while col_ind_begin < col_len:
        col_ind_end = min(col_ind_begin + MAX_WORKERS, col_len)
        with concurrent.futures.ThreadPoolExecutor(max_workers = MAX_WORKERS) as executor:
            future_predict = {executor.submit(func, df[cols[ind]]): ind for ind in range(col_ind_begin, col_ind_end)}
            for future in concurrent.futures.as_completed(future_predict):
                ind = future_predict[future]
                try:
                    df[cols[ind]] = future.result()
                except Exception as exc:
                    print('%dth feature normalize generate an exception: %s' % (ind, exc))
        col_ind_begin = col_ind_end
        if col_ind_begin % 100 == 0:
            print('Gen %d normalized features' % col_ind_begin)

def SortColumn(s, r):
    temp = np.array(sorted(s,reverse=True))
    if r % 1000 == 0:
        print ("sort rows: ", r)
    return r, temp, np.sum(temp > 0)

def SortData(df):
    CPU_CORES = 8
    sort_array = df.values
    res = [SortColumn(sort_array[r, :], r) for r in range(sort_array.shape[0])]
    # with Pool(processes=CPU_CORES) as p:
    #     res = [p.apply_async(SortColumn, args=(sort_array[r, :], r)) for r in range(sort_array.shape[0])]
    max_valid_column = 0
    for r in res:
        r, temp, valid_column  = r #.get()
        sort_array[r] = temp
        if valid_column > max_valid_column:
            max_valid_column = valid_column
    print ("max_valid_column: ", max_valid_column)
    return pd.DataFrame(sort_array[:, :SORT_LEN], index = df.index, columns = ['sort_' + str(i) for i in range(SORT_LEN)])

def CalcHistMeta(r, HistSize):
    hist = np.zeros(HistSize)
    # print (r)
    for d in r:
        # if d != 0:
        vid = int(d * (HistSize - 1.0));
        hist[vid] += 1
    return hist

def HistProcess(df):
    df = df.copy()
    df = df.apply(np.log1p)
    df_max = df.max().max()
    df_min = df.min().min()
    print ("df min: ", df_min, "max: ", df_max)
    df_local = (df - df_min) / (df_max - df_min)
    return df_local

def CalcHist(df, HistSize):
    df_local = df #HistProcess(df)
    hist_list = []
    for i in range(df_local.shape[0]):
        hist_list.append(CalcHistMeta(df_local.iloc[i], HistSize))
        # print (hist_list)
        # exit(0)
        if i % 1000 == 0:
            print ("Calc Hist Rows: ", i)
    return pd.DataFrame(np.array(hist_list), index = df.index, columns = ['hist_' + str(i) for i in range(HistSize)])

def AugData(df, df_local, col_select_rate):
    print ('Aug data using col_select_rate: ', col_select_rate)
    df_shape = df.shape
    hist_array = np.zeros((df_shape[0], HIST_SIZE))
    # sort_len = int(df_shape[1] * col_select_rate / 2)
    sort_array = np.zeros((df_shape[0], SORT_LEN))

    max_array = np.zeros(df_shape[0])
    min_array = np.zeros(df_shape[0])
    mean_array = np.zeros(df_shape[0])
    nz_mean_array = np.zeros(df_shape[0])
    nz_min_array = np.zeros(df_shape[0])

    pred_max_array = np.zeros(df_shape[0])
    pred_min_array = np.zeros(df_shape[0])
    pred_mean_array = np.zeros(df_shape[0])
    pred_nz_mean_array = np.zeros(df_shape[0])
    pred_nz_min_array = np.zeros(df_shape[0])

    rest_empty_num = 0
    select_array = np.random.choice([True, False], df_shape, p = [col_select_rate, 1 - col_select_rate])
    for i in range(df.shape[0]):
        r_select = df_local.iloc[i].values[select_array[i]]
        r_pred = df.iloc[i].values[~select_array[i]]
        hist_array[i] = CalcHistMeta(r_select, HIST_SIZE)
        sort_array[i] = np.array(sorted(r_select,reverse=True)[:SORT_LEN])

        max_array[i] = r_select.max()
        min_array[i] = r_select.min()
        mean_array[i] = r_select.mean()
        if r_select[r_select != 0].size > 0:
            nz_mean_array[i] = r_select[r_select != 0].mean()
            nz_min_array[i] = r_select[r_select != 0].min()

        if r_pred[r_pred != 0].size > 0:
            pred_max_array[i] = r_pred.max()
            pred_min_array[i] = r_pred.min()
            pred_mean_array[i] = r_pred.mean()
            pred_nz_mean_array[i] = r_pred[r_pred != 0].mean()
            pred_nz_min_array[i] = r_pred[r_pred != 0].min()
        else:
            rest_empty_num += 1
        if i % 1000 == 0:
            print ("Aug Data Rows: ", i)
    print ('Rest empty number when doing data augmentation data: ', rest_empty_num)
    df_array = np.c_[hist_array, sort_array, max_array, min_array, mean_array, nz_mean_array, nz_min_array, 
        pred_max_array, pred_min_array, pred_mean_array, pred_nz_mean_array, pred_nz_min_array]
    statistic_cols = ['max', 'min', 'mean', 'nz_mean', 'nz_min', 'pred_max', 'pred_min', 'pred_mean', 'pred_nz_mean', 'pred_nz_min']
    train_cols = ['hist_' + str(i) for i in range(HIST_SIZE)] + ['sort_' + str(i) for i in range(SORT_LEN)] + ['max', 'min', 'mean', 'nz_mean', 'nz_min']
    cols = ['hist_' + str(i) for i in range(HIST_SIZE)] + ['sort_' + str(i) for i in range(SORT_LEN)] + statistic_cols
    return pd.DataFrame(df_array, index = df.index, columns = cols), train_cols

def LoadAugDdata(target):
    print("\nData Load Stage")
    # if FLAGS.load_from_pickle:
    with open(path + 'train_test_nonormalize.pickle', 'rb') as handle:
        df, test_ID, y_train, train_row = pickle.load(handle)
    # target = 'pred_nz_mean'
    aug_data_list = []
    # df = df[:2000]
    # df_local = HistProcess(df)
    train_cols = ['hist_' + str(i) for i in range(HIST_SIZE)] + ['sort_' + str(i) for i in range(SORT_LEN)] + ['max', 'min', 'mean', 'nz_mean', 'nz_min']
    # train_cols = ['sort_' + str(i) for i in range(SORT_LEN)] + ['max', 'min', 'mean', 'nz_mean', 'nz_min']
    select_rates = np.arange(0.6, 0.85, 0.025)
    # with Pool(processes=8) as p:
    #     res = [p.apply_async(AugData, args=(df, df_local, select_rates[i])) for i in range(len(select_rates))]
    #     res = [r.get() for r in res]
    #     for aug_data, train_cols in res:
    #         # with open(path + 'aug_data_sort_hist_' + str(i) + '.pickle', 'wb+') as handle:
    #         #     pickle.dump(aug_data, handle, protocol=pickle.HIGHEST_PROTOCOL)
    #         aug_data = aug_data[aug_data[target] != 0]
    #         aug_data = aug_data.apply(np.log1p)
    #         aug_data_list.append(aug_data)
    for i in select_rates:
        # if os.path.isfile(path + 'aug_data_sort_hist_' + str(i) + '.pickle'):
        #     continue
        # aug_data, train_cols = AugData(df, df_local, i)
        # with open(path + 'aug_data_sort_hist_' + str(i) + '.pickle', 'wb+') as handle:
        #     pickle.dump(aug_data, handle, protocol=pickle.HIGHEST_PROTOCOL)
        with open(path + 'aug_data_sort_hist_' + str(i) + '.pickle', 'rb') as handle:
             aug_data = pickle.load(handle)
        aug_data = aug_data[aug_data[target] != 0]
        # aug_data = aug_data.apply(np.log1p).astype(np.float32)
        aug_data_list.append(aug_data)

    # Load Test Data
    with open(path + 'sort_df_log1p_minmaxnorm.pickle', 'rb+') as handle:
        sort_df = pickle.load(handle)
    with open(path + 'hist_df_log1p_minmaxnorm.pickle', 'rb+') as handle:
        hist_df = pickle.load(handle)  
    with open(path + 'statistic_features_use_full_cols_log1p_minmaxnorm.pickle', 'rb+') as handle:
        statistic_features = pickle.load(handle)
    test_data = pd.concat([hist_df, sort_df, statistic_features], axis = 1, sort = False)
    print ("test_data: \n", test_data.head())
    test_id = test_data.index

    df = pd.concat(aug_data_list, axis = 0, sort = False)
    print ("df: \n", df.head())
    train_data = df[train_cols]
    train_label = df[target].apply(np.log1p)

    # test_data = None
    # test_id = None
    leak_target = None
    valide_data = None
    valide_label = None
    weight = None
    keras_train.USED_FEATURE_LIST = list(train_data.columns.values)
    return train_data, train_label, test_data, test_id, valide_data, valide_label, weight, leak_target

def load_data(col):
    print("\nData Load Stage")
    if FLAGS.load_from_pickle:
        with open(path + 'train_test_nonormalize.pickle', 'rb') as handle:
             df, test_ID, y_train, train_row = pickle.load(handle)
        leak_target = pd.read_csv(path + 'target_leaktarget_30_1.csv', index_col = 'ID')
        leak_target = leak_target.loc[df[train_row:].index, 'leak_target']
        leak_target = leak_target[leak_target != 0]
        print ('leak_target shape: ', leak_target.shape)

        print("Shape before append columns: ", df.shape)
        origin_cols = df.columns
        # df = HistProcess(df)
        # hist_df = CalcHist(df, HIST_SIZE)
        # with open(path + 'hist_df_log1p_minmaxnorm.pickle', 'wb+') as handle:
        #     pickle.dump(hist_df, handle, protocol=pickle.HIGHEST_PROTOCOL)
        # # Sort every row by column value
        # sort_df = SortData(df)
        # with open(path + 'sort_df_log1p_minmaxnorm.pickle', 'wb+') as handle:
        #     pickle.dump(sort_df, handle, protocol=pickle.HIGHEST_PROTOCOL)
        # exit(0)
        with open(path + 'sort_df.pickle', 'rb+') as handle:
            sort_df = pickle.load(handle)
        with open(path + 'hist_df.pickle', 'rb+') as handle:
            hist_df = pickle.load(handle)        

        # append_pred_columns(df)
        top_cols_pred = [col + '_p' for col in top_cols]
        top_cols_new = [col + '_new' for col in top_cols]
        # df[top_cols_pred + top_cols_new].to_csv('new_cols.csv')
        if FLAGS.model_type == 'r':
            df = df[top_cols_new[::-1]]
        # rnn_pred = pd.read_csv(path + 'sub_2018_07_31_11_15_04.csv', index_col = 'ID')
        # df['rnn_pred'] = rnn_pred['target']
        # df["nz_mean"] = df[top_cols].apply(lambda x: x[x!=0].mean(), axis=1)
        # df["nz_max"] = df[top_cols].apply(lambda x: x[x!=0].max(), axis=1)
        # df["nz_min"] = df[top_cols].apply(lambda x: x[x!=0].min(), axis=1)
        # df["ez"] = df[top_cols].apply(lambda x: len(x[x==0]), axis=1)
        # df["mean"] = df[top_cols].apply(lambda x: x.mean(), axis=1)
        # df["max"] = df[top_cols].apply(lambda x: x.max(), axis=1)
        # df["min"] = df[top_cols].apply(lambda x: x.min(), axis=1)

        # df["nz_mean"] = df.apply(lambda x: x[x!=0].mean(), axis=1)
        # df["nz_min"] = df.apply(lambda x: x[x!=0].min(), axis=1)
        # df["mean"] = df.apply(lambda x: x.mean(), axis=1)
        # df["max"] = df.apply(lambda x: x.max(), axis=1)
        # df["min"] = df.apply(lambda x: x.min(), axis=1)
        # statistic_features_cols = ['max', 'min', 'mean', 'nz_mean', 'nz_min']
        # with open(path + 'statistic_features_use_full_cols_log1p_minmaxnorm.pickle', 'wb+') as handle:
        #     pickle.dump(df[statistic_features_cols], handle, protocol=pickle.HIGHEST_PROTOCOL)
        # exit(0)

        # df["pred_mean"] = df[top_cols_pred].apply(lambda x: x.mean(), axis=1)
        # df["pred_max"] = df[top_cols_pred].apply(lambda x: x.max(), axis=1)
        # df["pred_min"] = df[top_cols_pred].apply(lambda x: x.min(), axis=1)

        # statistic_features_cols = ["nz_mean", "nz_max", "nz_min", "ez", "mean", "max", "min", "pred_mean", "pred_max", "pred_min"]
        # with open(path + 'statistic_features.pickle', 'wb+') as handle:
        #     pickle.dump(df[statistic_features_cols], handle, protocol=pickle.HIGHEST_PROTOCOL)
        with open(path + 'statistic_features.pickle', 'rb+') as handle:
            statistic_features = pickle.load(handle)

        pred_mean = pd.read_csv(path + '_pred_mean_2018_08_14_14.csv', index_col = 'ID').rename(columns = {'target': 'aug_pred_mean'})
        pred_max = pd.read_csv(path + '_pred_max_2018_08_14_14.csv', index_col = 'ID').rename(columns = {'target': 'aug_pred_max'})
        pred_nz_mean = pd.read_csv(path + '_pred_nz_mean_2018_08_14_16.csv', index_col = 'ID').rename(columns = {'target': 'aug_pred_nz_mean'})
        pred_nz_min = pd.read_csv(path + '_pred_nz_min_2018_08_14_17.csv', index_col = 'ID').rename(columns = {'target': 'aug_pred_nz_min'})
        df.drop(columns = origin_cols, inplace = True)
        df = pd.concat([sort_df, hist_df, pred_mean, pred_max, pred_nz_mean, pred_nz_min], axis = 1, sort = False)
        print("Shape after append columns: ", df.shape)
        # with Pool(processes=8) as p:
        #     res = [p.apply_async(select_pred, args=(df, col)) for col in top_cols[:5]]
        #     res = [r.get() for r in res]
        # exit(0)
        # pred_col1_filter = pred_col1[(pred_col1['target'] >= 319) & (pred_col1['target'] <= 319612000)]
        # exit(0)
        # df[cols].to_csv(path + 'df.csv')
        # df = df.apply(np.log1p) #df[cols].apply(np.log1p)
        # print(df.head)
        # exit(0)
        # print(df)
        print("Do normalize...")
        df = df.apply(np.log1p)
        # df = (df - df.mean())/ df.std()
        # Normalize(df, Avg_Std_Normalize)
        print(df.head())
        # df = rank_INT_DF(df) #df.apply(rank_INT)
        # with open(path + 'train_test_rank_int_rand_tie.pickle', 'wb+') as handle:
        #     pickle.dump([df, test_ID, y_train, train_row], handle, protocol=pickle.HIGHEST_PROTOCOL)
        # exit(0)
    else:
        if FLAGS.debug:
            nrow = 100
        else:
            nrow = None
        train = pd.read_csv(path + '/train.csv', nrows = nrow, index_col = 'ID')
        test = pd.read_csv(path + '/test.csv', nrows = nrow, index_col = 'ID')
        test_ID = test.index #['ID']
        y_train = train['target']
        y_train = np.log1p(y_train)
        # train.drop("ID", axis = 1, inplace = True)
        train.drop("target", axis = 1, inplace = True)
        # test.drop("ID", axis = 1, inplace = True)
        # cols_with_onlyone_val = train.columns[train.nunique() == 1]
        # with open('cols_with_onlyone_val.pickle', 'wb+') as handle:
        #     pickle.dump(cols_with_onlyone_val, handle, protocol=pickle.HIGHEST_PROTOCOL)
        with open('cols_with_onlyone_val.pickle', 'rb') as handle:
            cols_with_onlyone_val = pickle.load(handle)
        print ("cols_with_onlyone_val: ", cols_with_onlyone_val)
        df = train.append(test)
        train_row = train.shape[0]
        df.drop(cols_with_onlyone_val.values, axis=1, inplace=True)
        # test.drop(cols_with_onlyone_val.values, axis=1, inplace=True)
        NUM_OF_DECIMALS = 32
        df = df.round(NUM_OF_DECIMALS)
        # test = test.round(NUM_OF_DECIMALS)
        # colsToRemove = []
        # columns = train.columns
        # for i in range(len(columns)-1):
        #     v = train[columns[i]].values
        #     for j in range(i + 1,len(columns)):
        #         if np.array_equal(v, train[columns[j]].values):
        #             colsToRemove.append(columns[j])
        # with open('colsToRemove.pickle', 'wb+') as handle:
        #     pickle.dump(colsToRemove, handle, protocol=pickle.HIGHEST_PROTOCOL)
        with open('colsToRemove.pickle', 'rb') as handle:
            colsToRemove = pickle.load(handle)
        print ("dupCols: ", colsToRemove)
        df.drop(colsToRemove, axis=1, inplace=True)
        with open(path + 'train_test_nonormalize.pickle', 'wb+') as handle:
            pickle.dump([df, test_ID, y_train, train_row], handle, protocol=pickle.HIGHEST_PROTOCOL)
        exit(0)
        for col in df.columns.values:
            figure = df[col].hist(histtype = 'step')
            plt.savefig(path + "/figures/df_" + col + ".png")
            exit(0) 
        print("Do normalize...")
        # df = (df - df.mean())/ df.std()
        df = (df - df.min())/ (df.max() - df.min())
        # df = rank_INT_DF(df) #df.apply(rank_INT)

        with open(path + 'train_test_minmax.pickle', 'wb+') as handle:
            pickle.dump([df, test_ID, y_train, train_row], handle, protocol=pickle.HIGHEST_PROTOCOL)
        # exit(0)
    if FLAGS.load_from_vae:
        vae_data = np.load(path + 'vae_data.npy')
        cols = ["vae_" + str(i) for i in range(vae_data.shape[1])]
        vae_df = pd.DataFrame(vae_data, columns = cols, index = df.index)
        df[cols] = vae_df
        print("after add vae shape: ", df.shape)
    # print(df.head) 

    if FLAGS.lgb_boost_dnn:
        keras_train.USED_FEATURE_LIST += ['lgb_pred']

    if FLAGS.model_type == 'v' or FLAGS.model_type == 'r':
        train_data = df
    else:
        train_data = df.iloc[:train_row, :]

    # Append leak rows
    # train_data = train_data.append(df.loc[leak_target.index])
    # y_train = y_train.append(np.log1p(leak_target))

    if FLAGS.predict_feature:
        valid_idx = (df[col] != 0)
        # print(valid_idx)
        train_data = df.loc[valid_idx, df.columns != col]
        test_data = df.loc[:, df.columns != col]
        train_label = df.loc[valid_idx, col]
        test_id = df.index
    else:
        if FLAGS.model_type == 'r':
            test_data = df
            test_id = df.index
        else:
            test_data = df.iloc[train_row:, :]
            test_id = test_ID
        train_label = y_train.values

    valide_data = None
    valide_label = None
    weight = None
    keras_train.USED_FEATURE_LIST = list(train_data.columns.values)
    return train_data, train_label, test_data, test_id, valide_data, valide_label, weight, leak_target


def sub(models, stacking_data = None, stacking_label = None, stacking_test_data = None, test = None, \
        scores_text = None, tid = None, sub_re = None, col = None, leak_target = None, aug_data_target = None):
    tmp_model_dir = "./model_dir/"
    if not os.path.isdir(tmp_model_dir):
        os.makedirs(tmp_model_dir, exist_ok=True)
    if FLAGS.stacking:
        np.save(os.path.join(tmp_model_dir, "stacking_train_data.npy"), stacking_data)
        np.save(os.path.join(tmp_model_dir, "stacking_train_label.npy"), stacking_label)
        np.save(os.path.join(tmp_model_dir, "stacking_test_data.npy"), stacking_test_data)
    elif FLAGS.model_type == 'v':
        np.save(os.path.join(tmp_model_dir, "vae_data.npy"), stacking_data)
    else:
        # if FLAGS.load_stacking_data:
        #     sub2[coly] = sub_re
        # else:
        sub_re = pd.DataFrame(models_eval(models, test),columns=["target"],index=tid)
        sub_re["target"] = np.expm1(sub_re["target"].values)
        # sub_re["target"][leak_target.index] = leak_target
        # blend = sub2 #blend[sub2.columns]
        if FLAGS.predict_feature:
            time_label = "_" + col + time.strftime('_%Y_%m_%d_%H', time.gmtime())
            sub_name = tmp_model_dir + time_label + ".csv"
        elif FLAGS.aug_data:
            time_label = "_" + aug_data_target + time.strftime('_%Y_%m_%d_%H', time.gmtime())
            sub_name = tmp_model_dir + time_label + ".csv"
        else:
            time_label = time.strftime('_%Y_%m_%d_%H_%M_%S', time.gmtime())
            sub_name = tmp_model_dir + "sub" + time_label + ".csv"
        sub_re.to_csv(sub_name)

        # save model to file
        for i, model in enumerate(models):
            if (model[1] == 'l'):
                model_name = tmp_model_dir + "model_" + str(i) + time_label + ".txt"
                model[0].save_model(model_name)
            elif (model[1] == 'k' or model[1] == 'r'):
                model_name = tmp_model_dir + "model_" + str(i) + time_label + ".h5"
                model[0].model.save(model_name)

        scores_text_frame = pd.DataFrame(scores_text, columns = ["score_text"])
        score_text_file = tmp_model_dir + "score_text" + time_label + ".csv"
        scores_text_frame.to_csv(score_text_file, index=False)
        scores = scores_text_frame["score_text"]
        for i in range(FLAGS.epochs):
            scores_epoch = scores.loc[scores.str.startswith('epoch:{0}'.format(i + 1))].map(lambda s: float(s.split()[1]))
            print ("Epoch{0} mean:{1} std:{2} min:{3} max:{4} median:{5}".format(i + 1, \
                scores_epoch.mean(), scores_epoch.std(), scores_epoch.min(), scores_epoch.max(), scores_epoch.median()))

    if not os.path.isdir(FLAGS.output_model_path):
        os.makedirs(FLAGS.output_model_path, exist_ok=True)
    for fileName in os.listdir(tmp_model_dir):
        dst_file = os.path.join(FLAGS.output_model_path, fileName)
        if os.path.exists(dst_file):
            os.remove(dst_file)
        shutil.move(os.path.join(tmp_model_dir, fileName), FLAGS.output_model_path)


if __name__ == "__main__":
    def train_sub(col):
        scores_text = []
        aug_data_target = None
        if FLAGS.aug_data:
            aug_data_target = 'pred_nz_min'
            train_data, train_label, test_data, tid, valide_data, valide_label, weight, leak_target = LoadAugDdata(aug_data_target)
        else:
            train_data, train_label, test_data, tid, valide_data, valide_label, weight, leak_target = load_data(col)
        if not FLAGS.load_stacking_data:
            models, stacking_data, stacking_label, stacking_test_data = nfold_train(train_data, train_label, flags = FLAGS, \
                    model_types = [FLAGS.model_type], scores = scores_text, test_data = test_data, \
                    valide_data = valide_data, valide_label = valide_label, cat_max = None, emb_weight = None)
        else:
            for i in range(train_label.shape[1]):
                models, stacking_data, stacking_label, stacking_test_data = nfold_train(train_data, train_label[:, i], flags = FLAGS, \
                    model_types = [FLAGS.model_type], scores = scores_text, emb_weight = emb_weight, test_data = test_data \
                    # , valide_data = train_data[:100], valide_label = train_label[:100, i]
                    )
                sub_re[:, i] = models_eval(models, test_data)
        sub(models, stacking_data = stacking_data, stacking_label = stacking_label, stacking_test_data = stacking_test_data, \
            test = test_data, scores_text = scores_text, tid = tid, col = col, leak_target = leak_target, aug_data_target = aug_data_target)
    if FLAGS.predict_feature:
        col_num = 0
        for col in top_cols:
            train_sub(col)
            col_num += 1
            # if col_num == 5:
            #     break
    else:
        train_sub(None)