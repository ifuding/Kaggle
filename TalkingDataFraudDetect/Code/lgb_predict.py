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
from sklearn import metrics
import lightgbm as lgb

target = 'is_attributed'
predictors = ['app','device','os', 'channel', 'hour', 'day', 
              'ip_tcount', 'ip_tchan_count', 'ip_app_count',
              'ip_app_os_count', 'ip_app_os_var',
              'ip_app_channel_var_day','ip_app_channel_mean_hour']
categorical = ['app', 'device', 'os', 'channel', 'hour', 'day']

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
flags.DEFINE_bool("search_best_iteration", True, "Whether to search best iteration")
flags.DEFINE_integer("best_iteration", 1, "best iteration")
FLAGS = flags.FLAGS

path = FLAGS.input_training_data_path
dtypes = {
'ip' : 'uint32', 'app' : 'uint16', 'device' : 'uint16', 'os' : 'uint16', 'channel' : 'uint16', 'is_attributed' : 'uint8', 
'click_id' : 'uint32', 'day' : 'uint8', 'hour' : 'uint8', 'yesterday' : 'uint8',
    }
for dense_feature in keras_train.DENSE_FEATURE_LIST:
    dtypes[dense_feature] = 'uint16'

def find_best_iteration_search(bst):
    """
    """
    with timer("loading valide data"):
        print('loading valide data...')
        path_prefix = "valide_Cnt"
        if FLAGS.debug:
            valide_data_path = path + path_prefix + "_Top.ss.csv"
            valide_df = pd.read_csv(valide_data_path, dtype=dtypes,
            usecols = ['is_attributed'] + keras_train.USED_FEATURE_LIST)
        else:
            valide_data_path = path + path_prefix + ".csv"
            valide_df = pd.read_csv(valide_data_path, dtype=dtypes, header = None, sep = '\t',
            names=['is_attributed'] + keras_train.DATA_HEADER,
            usecols = ['is_attributed'] + keras_train.USED_FEATURE_LIST)
        print(valide_df.info())
        valide_data = valide_df[keras_train.USED_FEATURE_LIST].values.astype(np.uint16)
        valide_label = valide_df['is_attributed'].values.astype(np.uint8)
        print ("valide type {0}".format(valide_data.dtype))
        pos_cnt = valide_label.sum()
        neg_cnt = len(valide_label) - pos_cnt
        print ("valide data pos : {0} neg : {1}".format(pos_cnt, neg_cnt))
        del valide_df
        gc.collect()
    with timer("finding best iteration..."):
        for i in range(800, 1600, 200):
            y_pred = bst.predict(valide_data, num_iteration=i)
            score = metrics.roc_auc_score(valide_label, y_pred)
            print ("Iteration: {0} AUC: {1}".format(i, score))


def predict_test(bst):
    with timer("loading test data"):
        print('loading test data...')
        path_prefix = "test_Cnt"
        if FLAGS.debug:
            test_data_path = path + path_prefix + "_Top.ss.csv"
            test_df = pd.read_csv(test_data_path, dtype=dtypes, usecols = ['click_id'] + keras_train.USED_FEATURE_LIST)
        else:
            test_data_path = path + path_prefix + ".csv"
            test_df = pd.read_csv(test_data_path, dtype=dtypes, header = None, sep = '\t', 
        names=['click_id'] + keras_train.DATA_HEADER, #nrows = 10000,
        usecols = ['click_id'] + keras_train.USED_FEATURE_LIST)
        print(test_df.info())
        test_data = test_df[keras_train.USED_FEATURE_LIST].values.astype(np.uint16)
        test_id = test_df['click_id'].astype('uint32').values.astype(np.uint32)
        print ("test type {0}".format(test_data.dtype))
        del test_df
        gc.collect()
    with timer("predicting test data"):
        print('predicting test data...')
        sub_re = pd.DataFrame(test_id, columns = ['click_id'])
        sub_re['is_attributed'] = bst.predict(test_data, num_iteration=FLAGS.best_iteration)
        time_label = time.strftime('_%Y_%m_%d_%H_%M_%S', time.gmtime())
        sub_name = FLAGS.output_model_path + "sub" + time_label + ".csv"
        sub_re.to_csv(sub_name, index=False)


if __name__ == "__main__":
    # load model to predict
    bst = lgb.Booster(model_file= FLAGS.output_model_path + 'model_2018_04_25_13_16_57.txt')
    if FLAGS.search_best_iteration:
        find_best_iteration_search(bst)
    else:
        predict_test(bst)