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
FLAGS = flags.FLAGS

path = FLAGS.input_training_data_path

def load_data():
    print("\nData Load Stage")
    if FLAGS.load_from_pickle:
        df = pd.read_pickle(path + 'lgb_pred.pickle')
        emb_weight = np.load(path + 'emb_weight.pickle.npy')
        y = pd.read_pickle(path + 'y.pickle')
        traindex = pd.read_csv(path + 'traindex.csv', header = None, names  = ['item_id'], index_col  = 0).index
        testdex = pd.read_csv(path + 'testdex.csv', header = None, names  = ['item_id'], index_col  = 0).index
        # print(testdex.info())
        # print(testdex.head())
        # exit(0)
    else:
        if FLAGS.debug:
            nrow = 100
        else:
            nrow = None
        train = pd.read_csv(path + '/train.csv', nrows = nrow)
        test = pd.read_csv(path + '/test.csv', nrows = nrow)
        test_ID = test['ID']
        y_train = train['target']
        y_train = np.log1p(y_train)
        train.drop("ID", axis = 1, inplace = True)
        train.drop("target", axis = 1, inplace = True)
        test.drop("ID", axis = 1, inplace = True)
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
        # test.drop(colsToRemove, axis=1, inplace=True) 
        keras_train.USED_FEATURE_LIST = list(df.columns.values)

        print("Do normalize...")
        # df = (df - df.mean())/ df.std()
        df = (df - df.min())/ (df.max() - df.min())

        if FLAGS.lgb_boost_dnn:
            models = []
            for i in range(FLAGS.lgb_ensemble_nfold):
                bst = lgb.Booster(model_file= FLAGS.input_previous_model_path + '/model_' + str(0) + '.txt')
                models.append((bst, 'l'))
            print('LightGbm model predicting...')
            df['lgb_pred'] = models_eval(models, df)
            keras_train.USED_FEATURE_LIST += ['lgb_pred']

        if FLAGS.model_type == 'k':
            pass

    if FLAGS.lgb_boost_dnn:
        keras_train.USED_FEATURE_LIST += ['lgb_pred']

    train_data = df[:train_row]
    train_label = y_train.values
    test_data = df[train_row:]
    test_id = test_ID
    valide_data = None
    valide_label = None
    weight = None
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
        sub_re = pd.DataFrame(models_eval(models, test),columns=["target"],index=tid)
        sub_re["target"] = np.expm1(sub_re["target"].values)
        # blend = sub2 #blend[sub2.columns]
        time_label = time.strftime('_%Y_%m_%d_%H_%M_%S', time.gmtime())
        sub_name = tmp_model_dir + "sub" + time_label + ".csv"
        sub_re.to_csv(sub_name)

        # save model to file
        for i, model in enumerate(models):
            if (model[1] == 'l'):
                model_name = tmp_model_dir + "model_" + str(i) + time_label + ".txt"
                model[0].save_model(model_name)
            elif (model[1] == 'k'):
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
    scores_text = []
    train_data, train_label, test_data, tid, valide_data, valide_label, weight = load_data()
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
    # sub(models, stacking_data = stacking_data, stacking_label = stacking_label, stacking_test_data = stacking_test_data, \
    #        test = test_data, scores_text = scores_text, tid = tid)