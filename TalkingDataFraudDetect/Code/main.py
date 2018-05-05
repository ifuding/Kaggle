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
FLAGS = flags.FLAGS

path = FLAGS.input_training_data_path
dtypes = {
'ip' : 'uint32', 'app' : 'uint16', 'device' : 'uint16', 'os' : 'uint16', 'channel' : 'uint16', 'is_attributed' : 'uint8', 
'click_id' : 'uint32', 'day' : 'uint8', 'hour' : 'uint8', 'yesterday' : 'uint8', 'minute' : 'uint8', 'second' : 'uint8',
'id' : 'uint32', 'is_attributed_test' : 'float32'
        }
DENSE_FEATURE_TYPE = keras_train.DENSE_FEATURE_TYPE
for dense_feature in keras_train.DENSE_FEATURE_LIST:
    dtypes[dense_feature] = DENSE_FEATURE_TYPE

def load_train_data():
    with timer("loading train data"):
        print('loading train data...')
        if FLAGS.split_train_val:
            path_prefix = "train_Cnt_Id"
        else:
            path_prefix = "train_part_Cnt_Neg20"
        if FLAGS.debug:
            train_data_path = path + path_prefix + "_Top.ss.csv"
            train_df = pd.read_csv(train_data_path, dtype=dtypes, 
            usecols = ['is_attributed'] + keras_train.USED_FEATURE_LIST)
        else:
            train_data_path = path + path_prefix + ".csv"
            if FLAGS.split_train_val:
                train_df = pd.read_csv(train_data_path, dtype=dtypes, header = None, sep = '\t', 
    names=['is_attributed'] + keras_train.DATA_HEADER, skiprows=range(0,184903890-FLAGS.train_eval_len),
    usecols = ['is_attributed'] + keras_train.USED_FEATURE_LIST)
            else:
                if FLAGS.stacking:
                    train_df = pd.read_csv(train_data_path, dtype=dtypes, header = None, sep = '\t', 
            names=['is_attributed'] + keras_train.DATA_HEADER, #skiprows=range(0,10000000),
            usecols = ['is_attributed'] + keras_train.USED_FEATURE_LIST)
                else:
                    train_df = pd.read_csv(train_data_path, dtype=dtypes, header = None, sep = '\t', 
            names=['is_attributed'] + keras_train.DATA_HEADER, #nrows = 10000000, #skiprows=range(0,10000000),
            usecols = ['is_attributed'] + keras_train.USED_FEATURE_LIST)
        print(train_df.info())
    return train_df


def load_valide_data():
    with timer("loading valide data"):
        print('loading valide data...')
        if not FLAGS.split_train_val:
            path_prefix = "valide_Cnt"
            if FLAGS.debug:
                valide_data_path = path + path_prefix + "_Top.ss.csv"
                valide_df = pd.read_csv(valide_data_path, dtype=dtypes,
                usecols = ['is_attributed'] + keras_train.USED_FEATURE_LIST)
            else:
                valide_data_path = path + path_prefix + ".csv"
                valide_df = pd.read_csv(valide_data_path, dtype=dtypes, header = None, sep = '\t',
    names=['id', 'is_attributed'] + keras_train.DATA_HEADER, #nrows = 10000,
    usecols = ['is_attributed'] + keras_train.USED_FEATURE_LIST)
            print(valide_df.info())
    return valide_df


def load_test_data():
    with timer("loading test data"):
        print('loading test data...')
        if FLAGS.test_for_train:
            path_prefix = "test_Cnt_ForTrain"
        else:
            path_prefix = "test_Cnt"
        if FLAGS.debug:
            test_data_path = path + path_prefix + "_Top.ss.csv"
            test_df = pd.read_csv(test_data_path, dtype=dtypes, usecols = ['click_id'] + keras_train.USED_FEATURE_LIST)
        else:
            test_data_path = path + path_prefix + ".csv"
            test_df = pd.read_csv(test_data_path, dtype=dtypes, header = None, sep = '\t', 
        names=['id', 'click_id'] + keras_train.DATA_HEADER, nrows = 10000,
        usecols = ['click_id'] + keras_train.USED_FEATURE_LIST)
        if FLAGS.test_for_train:
            train_df=train_df.append(test_df[['is_attributed'] + keras_train.USED_FEATURE_LIST])
            test_df = test_df[:100000]
        print(test_df.info())
        gc.collect()
    return test_df

def gen_stacking_data(in_data):
    k_model = load_model(FLAGS.input_previous_model_path + '/model_allSparse_09744.h5')
    print (k_model.summary())
    exit(0)
    emb_model = Model(inputs = k_model.inputs, outputs = k_model.get_layer(name = 'merge_sparse_emb').output)
    emb_vector = emb_model.predict(keras_train.DNN_Model.DNN_DataSet(None, in_data, sparse = True, dense = False), 
                verbose=0, batch_size=10240)
    k_pred = k_model.predict(keras_train.DNN_Model.DNN_DataSet(None, in_data, sparse = True, dense = False), 
                verbose=0, batch_size=10240)
    # k_pred = k_model.predict(in_data[:len(keras_train.USED_CATEGORY_FEATURES)], verbose=0, batch_size=10240)
    out_data = np.c_[in_data, emb_vector, k_pred]
    print ("Shape Before stacking: {0}".format(in_data.shape))
    print ("Shape After stacking: {0}".format(out_data.shape))
    return out_data


def load_data():
    """
    """
    train_df = load_train_data()
    valide_df = load_valide_data()
    test_df = load_test_data()

    # len_train = 20905996
    # len_valide = 20000001
    # df = pd.read_pickle('AvgStd_TrainValTest.pickle')
    # train_df = df[: len_train]
    # valide_df = df[len_train: len_train + len_valide]
    # test_df = df[len_train + len_valide: len_train + len_valide + 100000]
    if FLAGS.load_only_singleCnt:
        train_data = train_df[keras_train.USED_FEATURE_LIST].values.astype(DENSE_FEATURE_TYPE)
        train_label = train_df['is_attributed'].values.astype(np.uint8)
        del train_df
        gc.collect()
        if FLAGS.split_train_val:
            train_len = len(train_label)
            valide_data = train_data[train_len - FLAGS.eval_len:]
            valide_label = train_label[train_len - FLAGS.eval_len:]
            train_data = train_data[:train_len - FLAGS.eval_len]
            train_label = train_label[:train_len - FLAGS.eval_len]
        else:
            valide_data = valide_df[keras_train.USED_FEATURE_LIST].values.astype(DENSE_FEATURE_TYPE)
            valide_label = valide_df['is_attributed'].values.astype(np.uint8)
            del valide_df
            gc.collect()
        test_data = test_df[keras_train.USED_FEATURE_LIST].values.astype(DENSE_FEATURE_TYPE)
        test_id = test_df['click_id'].astype('uint32').values.astype(np.uint32)
        del test_df
    else:
        # valide_len = len_train // 5
        train_data = train_df[train_df["day"] != 9][:len_train][keras_train.SPARSE_FEATURE_LIST].values
        train_label = train_df[train_df["day"] != 9][:len_train]['is_attributed'].values
        valide_data = train_df[train_df["day"] == 9][:len_train][keras_train.SPARSE_FEATURE_LIST].values
        valide_label = train_df[train_df["day"] == 9][:len_train]['is_attributed'].values

        test_data = train_df[len_train:][keras_train.SPARSE_FEATURE_LIST].values
        test_id = train_df[len_train:]['click_id'].astype('uint32').values
        del train_df
    gc.collect()

    weight = None
    if FLAGS.neg_sample:
        train_data, train_label, weight = neg_sample(train_data, train_label, FLAGS.sample_C)
    if (FLAGS.log_transform):
        train_data = (np.log(train_data) * 10000).astype(np.uint32)
        valide_data = (np.log(valide_data) * 10000).astype(np.uint32)
        test_data = (np.log(test_data) * 10000).astype(np.uint32)

    if FLAGS.stacking:
        train_data = gen_stacking_data(train_data)
        valide_data = gen_stacking_data(valide_data)
        test_data = gen_stacking_data(test_data)

    pos_cnt = train_label.sum()
    neg_cnt = len(train_label) - pos_cnt
    print ("train type: {0} train size: {1} train data pos: {2} neg: {3}".format(
            train_data.dtype, len(train_data), pos_cnt, neg_cnt))
    pos_cnt = valide_label.sum()
    neg_cnt = len(valide_label) - pos_cnt
    print ("valide type: {0} valide size: {1} valide data pos: {2} neg: {3}".format(
            valide_data.dtype, len(valide_data), pos_cnt, neg_cnt))
    print ("test type: {0} test size: {1}".format(test_data.dtype, len(test_data)))
    return train_data, train_label, test_data, test_id, valide_data, valide_label, weight


def sub(mdoels, stacking_data = None, stacking_label = None, stacking_test_data = None, test = None, \
        scores_text = None, tid = None, sub_re = None):
    tmp_model_dir = "./model_dir/"
    if not os.path.isdir(tmp_model_dir):
        os.makedirs(tmp_model_dir, exist_ok=True)
    if False:
        #FLAGS.stacking:
        np.save(os.path.join(tmp_model_dir, "stacking_train_data.npy"), stacking_data)
        np.save(os.path.join(tmp_model_dir, "stacking_train_label.npy"), stacking_label)
        np.save(os.path.join(tmp_model_dir, "stacking_test_data.npy"), stacking_test_data)
    else:
        # if FLAGS.load_stacking_data:
        #     sub2[coly] = sub_re
        # else:
        sub_re = pd.DataFrame(tid, columns = ['click_id'])
        sub_re['is_attributed'] = models_eval(models, test)
        # sub2[c] = sub2[c].clip(0+1e12, 1-1e12)
        # blend = sub2 #blend[sub2.columns]
        time_label = time.strftime('_%Y_%m_%d_%H_%M_%S', time.gmtime())
        sub_name = tmp_model_dir + "sub" + time_label + ".csv"
        sub_re.to_csv(sub_name, index=False)

        # save model to file
        if (models[0][1] == 'l'):
            model_name = tmp_model_dir + "model" + ".txt"
            models[0][0].save_model(model_name)
        elif (models[0][1] == 'k'):
            model_name = tmp_model_dir + "model" + ".h5"
            models[0][0].model.save(model_name)

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
    scores_text = []
    train_data, train_label, test_data, tid, valide_data, valide_label, weight = load_data()
    if not FLAGS.load_only_singleCnt and FLAGS.model_type == 'k':
        test_data = list(test_data.transpose())
    if not FLAGS.load_stacking_data:
        models, stacking_data, stacking_label, stacking_test_data = nfold_train(train_data, train_label, flags = FLAGS, \
                model_types = [FLAGS.model_type], scores = scores_text, test_data = test_data, \
                valide_data = valide_data, valide_label = valide_label)
    else:
        for i in range(train_label.shape[1]):
            models, stacking_data, stacking_label, stacking_test_data = nfold_train(train_data, train_label[:, i], flags = FLAGS, \
                model_types = [FLAGS.model_type], scores = scores_text, emb_weight = emb_weight, test_data = test_data \
                # , valide_data = train_data[:100], valide_label = train_label[:100, i]
                )
            sub_re[:, i] = models_eval(models, test_data)
    sub(models, stacking_data = stacking_data, stacking_label = stacking_label, stacking_test_data = stacking_test_data, \
            test = test_data, scores_text = scores_text, tid = tid)