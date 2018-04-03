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
FLAGS = flags.FLAGS


def load_data():
    """
    """
    path = FLAGS.input_training_data_path

    dtypes = {
            'ip'            : 'uint32',
            'app'           : 'uint16',
            'device'        : 'uint16',
            'os'            : 'uint16',
            'channel'       : 'uint16',
            'is_attributed' : 'uint8',
            'click_id'      : 'uint32'
            }

    with timer("loading train data"):
        print('loading train data...')
        if FLAGS.debug:
            train_data_path = path+"train_sample.csv"
        else:
            train_data_path = path+"train.csv"
        train_df = pd.read_csv(train_data_path, \
                    # skiprows=range(1,144903891), nrows=40000000, \
                    dtype=dtypes, usecols=['ip','app','device','os', 'channel', 'click_time', 'is_attributed'])

    with timer("loading test data"):
        print('loading test data...')
        if FLAGS.debug:
            test_data_path = path+"test_sample.csv"
        else:
            test_data_path = path+"test.csv"
        test_df = pd.read_csv(test_data_path, \
                    dtype=dtypes, usecols=['ip','app','device','os', 'channel', 'click_time', 'click_id'])
        len_train = len(train_df)
        train_df=train_df.append(test_df)
        del test_df
        gc.collect()

    with timer("Extracting new features"):
        print('Extracting new features...')
        train_df['hour'] = pd.to_datetime(train_df.click_time).dt.hour.astype('uint8')
        train_df['day'] = pd.to_datetime(train_df.click_time).dt.day.astype('uint8')
        gc.collect()
        # for col in train_df:
        #     uniq = train_df[col].unique()
        #     print ("{0} {1} {2}".format(col, len(uniq), uniq.max()))
        # exit(0)

    # train_df = gen_features(train_df)

    # train_df.drop(["click_time"], axis = 1, inplace = True)
    # print("vars and data type: ")
    # train_df.info()

    # # np.save("train_test_df.npy", train_df[predictors].values)
    # unit_size = train_df.size #(train_df.size // 4) + 1
    # begin = 0
    # partition = 0
    # try:
    #     while begin < train_df.size:
    #         train_df.iloc[begin: min(train_df.size, begin + unit_size)].to_pickle( \
    #                 "train_test_df_" + str(partition) + ".pkl", compression = "bz2")
    #         begin += unit_size
    #         partition += 1
    # finally:
    #     print ("Save failed!")
    #     pass
    valide_len = 37000000
    train_data = train_df[:len_train - valide_len][keras_train.SPARSE_FEATURE_LIST].values
    train_label = train_df[:len_train - valide_len]['is_attributed'].values
    valide_data = train_df[len_train - valide_len:len_train][keras_train.SPARSE_FEATURE_LIST].values
    valide_label = train_df[len_train - valide_len:len_train]['is_attributed'].values

    test_data = train_df[len_train:][keras_train.SPARSE_FEATURE_LIST].values
    test_id = train_df[len_train:]['click_id'].astype('uint32').values
    del train_df
    gc.collect()

    print("train size: ", len(train_data))
    # print("valid size: ", len(val_df))
    print("test size : ", len(test_data))

    return train_data, train_label, test_data, test_id, valide_data, valide_label


def sub(mdoels, stacking_data = None, stacking_label = None, stacking_test_data = None, test = None, \
        scores_text = None, tid = None, sub_re = None):
    tmp_model_dir = "./model_dir/"
    if not os.path.isdir(tmp_model_dir):
        os.makedirs(tmp_model_dir, exist_ok=True)
    if FLAGS.stacking:
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
    train_data, train_label, test_data, tid, valide_data, valide_label = load_data()
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
            test = list(test_data.transpose()), scores_text = scores_text, tid = tid)