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
flags.DEFINE_bool("load_from_vae", True, "load_from_vae")
flags.DEFINE_bool("predict_feature", False, "predict_feature")
FLAGS = flags.FLAGS

path = FLAGS.input_training_data_path

top_cols = ['f190486d6', '58e2e02e6', '2288333b4', '26ab20ff9', '491b9ee45', '9fd594eec', 'd3245937e', '6786ea46d', 
'bb1113dbb', 'b30e932ba', 'ba4ceabc5', 'eeb9cd3aa', '402b0d650', '17b81a716', '20aa07010', 'f32763afc', 'b6fa5a5fd', 
'edc84139a', 'fb387ea33', '9e3aea49a', 'f74e8f13d', '4edc3388d']

def select_pred(df, col):
    print ('Append column: ', col)
    pred_col = pd.read_csv(path + '_' + col + '_2018_07_25_07.csv', index_col = 'ID')
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
    MAX_WORKERS = 8
    cols = top_cols[:5]
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
    return (s - s.min()) / (s.max() - s.min())

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

def load_data(col):
    print("\nData Load Stage")
    if FLAGS.load_from_pickle:
        with open(path + 'train_test_nonormalize.pickle', 'rb') as handle:
             df, test_ID, y_train, train_row = pickle.load(handle)
            #  for col in df.columns.values:
            #     # print (df[col].value_counts())
            #     figure = df[col].value_counts().apply(np.log).plot(kind = 'line')
            #     plt.savefig(path + "/figures/df_" + col + ".png")
                # exit(0)
            #  exit(0)
        cols = ['f190486d6', '58e2e02e6', 'eeb9cd3aa', '9fd594eec', '6eef030c1', '15ace8c9f', 
        'fb0f5dbfe', '58e056e12', '20aa07010', '024c577b9', 'd6bb78916', 'b43a7cfd5',
        '58232a6fb', '1702b5bf0', '324921c7b', '62e59a501', '2ec5b290f', '241f0f867',
        'fb49e4212', '66ace2992', 'f74e8f13d', '5c6487af1', '963a49cdc', '26fc93eb7',
        '1931ccfdd', '703885424', '70feb1494', '491b9ee45', '23310aa6f', 'e176a204a',
        '6619d81fc', '1db387535']
        # def select_pred(df, col):
        #     print ('Append column: ', col)
        #     pred_col = pd.read_csv(path + '_' + col + '_2018_07_25_07.csv', index_col = 'ID')
        #     df[col + '_p'] = pred_col['target']
        #     df[col + '_new'] = df[col]
        #     select_s = (df[col] == 0) #& (df[col + '_p'] >= 319) & (df[col + '_p'] <= 319612000)
        #     df[col + '_new'][select_s] = df[col + '_p'][select_s]
            # print (df[[col, col + '_p', col + '_new']][:100])
        # for col in top_cols[:5]:
        #     select_pred(df, col)
        print("Shape before append columns: ", df.shape)
        append_pred_columns(df)
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
        # df = (df - df.mean())/ df.std()
        Normalize(df, Min_Max_Normalize)
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
        train_data = df[:train_row]

    if FLAGS.predict_feature:
        valid_idx = (df[col] != 0)
        # print(valid_idx)
        train_data = df.loc[valid_idx, df.columns != col]
        test_data = df.loc[:, df.columns != col]
        train_label = df.loc[valid_idx, col]
        test_id = df.index
    else:
        test_data = df[train_row:]
        train_label = y_train.values
        test_id = test_ID
    valide_data = None
    valide_label = None
    weight = None
    keras_train.USED_FEATURE_LIST = list(train_data.columns.values)
    return train_data, train_label, test_data, test_id, valide_data, valide_label, weight


def sub(models, stacking_data = None, stacking_label = None, stacking_test_data = None, test = None, \
        scores_text = None, tid = None, sub_re = None, col = None):
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
        # blend = sub2 #blend[sub2.columns]
        if FLAGS.predict_feature:
            time_label = "_" + col + time.strftime('_%Y_%m_%d_%H', time.gmtime())
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
        train_data, train_label, test_data, tid, valide_data, valide_label, weight = load_data(col)
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
            test = test_data, scores_text = scores_text, tid = tid, col = col)
    if FLAGS.predict_feature:
        col_num = 0
        for col in top_cols:
            train_sub(col)
            col_num += 1
            if col_num == 5:
                break
    else:
        train_sub(None)