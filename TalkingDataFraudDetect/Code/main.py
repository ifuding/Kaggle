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
FLAGS = flags.FLAGS


def load_data():
    """
    """
    path = FLAGS.input_training_data_path

    dtypes = {
'ip' : 'uint32', 'app' : 'uint16', 'device' : 'uint16', 'os' : 'uint16', 'channel' : 'uint16', 'is_attributed' : 'uint8', 
'click_id' : 'uint32', 'day' : 'uint8', 'hour' : 'uint8', 'yesterday' : 'uint8',
# 'ipCnt' : 'uint32', 'ipAttCnt' : 'uint16', 'appCnt' : 'uint32', 'appAttCnt' : 'uint32', 'deviceCnt' : 'uint32', 'deviceAttCnt' : 'uint32',
#  'osCnt' : 'uint32', 'osAttCnt' : 'uint32', 'channelCnt' : 'uint32', 'channelAttCnt' : 'uint32',
#  'ipappCnt' : 'uint32', 'ipappAttCnt' : 'uint16', 'ipdeviceCnt' : 'uint32', 'ipdeviceAttCnt': 'uint16', 'iposCnt' : 'uint32', 'iposAttCnt' : 'uint16',
#  'ipchannelCnt' : 'uint32', 'ipchannelAttCnt' : 'uint16', 'appdeviceCnt' : 'uint32', 'appdeviceAttCnt' : 'uint32', 'apposCnt' : 'uint32', 'apposAttCnt' : 'uint16',
#  'appchannelCnt' : 'uint32', 'appchannelAttCnt' : 'uint32', 'deviceosCnt' : 'uint32', 'deviceosAttCnt' : 'uint32', 'devicechannelCnt' : 'uint32', 'devicechannelAttCnt' : 'uint16',
#  'oschannelCnt' : 'uint32', 'oschannelAttCnt' : 'uint16',
#  'ipappdeviceCnt' : 'uint32', 'ipappdeviceAttCnt' : 'uint16',  'ipapposCnt' : 'uint16', 'ipapposAttCnt' : 'uint16',  'ipappchannelCnt' : 'uint32', 'ipappchannelAttCnt' : 'uint16', 
#  'ipdeviceosCnt' : 'uint32', 'ipdeviceosAttCnt' : 'uint16', 'ipdevicechannelCnt' : 'uint32', 'ipdevicechannelAttCnt' : 'uint16',  'iposchannelCnt' : 'uint16', 'iposchannelAttCnt' : 'uint8',  
#  'appdeviceosCnt' : 'uint32', 'appdeviceosAttCnt' : 'uint16',  'appdevicechannelCnt' : 'uint32', 'appdevicechannelAttCnt': 'uint16',  'apposchannelCnt' : 'uint32', 'apposchannelAttCnt' : 'uint16', 
#  'deviceoschannelCnt' : 'uint32', 'deviceoschannelAttCnt' : 'uint16', 
#  'ipappdeviceosCnt' : 'uint16', 'ipappdeviceosAttCnt' : 'uint16',  'ipappdevicechannelCnt' : 'uint32', 'ipappdevicechannelAttCnt' : 'uint16',  'ipapposchannelCnt' : 'uint16', 'ipapposchannelAttCnt' : 'uint8', 
#  'ipdeviceoschannelCnt' : 'uint16', 'ipdeviceoschannelAttCnt' : 'uint8',  'appdeviceoschannelCnt' : 'uint32', 'appdeviceoschannelAttCnt' : 'uint16',
#  'hourCnt' : 'uint32', 'hourAttCnt' : 'uint16', 'iphourCnt' : 'uint16', 'iphourAttCnt' : 'uint8',  'apphourCnt' : 'uint32', 'apphourAttCnt' : 'uint16',  'devicehourCnt' : 'uint32', 'devicehourAttCnt' : 'uint16', 
#  'oshourCnt' : 'uint32', 'oshourAttCnt' : 'uint16', 'channelhourCnt' : 'uint32', 'channelhourAttCnt' : 'uint16', 
#  'ipapphourCnt' : 'uint16', 'ipapphourAttCnt' : 'uint8',  'ipdevicehourCnt' : 'uint16', 'ipdevicehourAttCnt' : 'uint8',  'iposhourCnt' : 'uint16', 'iposhourAttCnt' : 'uint8',  'ipchannelhourCnt' : 'uint16', 
#  'ipchannelhourAttCnt' : 'uint8', 'appdevicehourCnt' : 'uint32', 'appdevicehourAttCnt' : 'uint16',  'apposhourCnt' : 'uint32', 'apposhourAttCnt' : 'uint16', 'appchannelhourCnt' : 'uint32', 'appchannelhourAttCnt' : 'uint16', 
#  'deviceoshourCnt' : 'uint32', 'deviceoshourAttCnt' : 'uint16',  'devicechannelhourCnt' : 'uint32', 'devicechannelhourAttCnt' : 'uint16',  'oschannelhourCnt' : 'uint32', 'oschannelhourAttCnt' : 'uint16',
#  'ipAttrOverCnt' : 'uint16', 'appAttrOverCnt' : 'uint16',  'deviceAttrOverCnt' : 'uint16', 'osAttrOverCnt' : 'uint16',  'channelAttrOverCnt' : 'uint16', 'hourAttrOverCnt' : 'uint16', 
#  'ipappAttrOverCnt' : 'uint16', 'ipdeviceAttrOverCnt' : 'uint16',  'iposAttrOverCnt' : 'uint16', 'ipchannelAttrOverCnt' : 'uint16',  'iphourAttrOverCnt' : 'uint16', 'appdeviceAttrOverCnt' : 'uint16', 
#  'apposAttrOverCnt' : 'uint16', 'appchannelAttrOverCnt' : 'uint16', 'apphourAttrOverCnt' : 'uint16', 'deviceosAttrOverCnt' : 'uint16', 'devicechannelAttrOverCnt' : 'uint16', 'devicehourAttrOverCnt' : 'uint16', 
#  'oschannelAttrOverCnt' : 'uint16', 'oshourAttrOverCnt' : 'uint16', 'channelhourAttrOverCnt' : 'uint16',
#  'ipappdeviceAttrOverCnt':'uint16', 'ipapposAttrOverCnt':'uint16', 'ipappchannelAttrOverCnt':'uint16', 'ipapphourAttrOverCnt':'uint16', 'ipdeviceosAttrOverCnt':'uint16', 'ipdevicechannelAttrOverCnt':'uint16', 
#  'ipdevicehourAttrOverCnt':'uint16', 'iposchannelAttrOverCnt':'uint16', 'iposhourAttrOverCnt':'uint16', 'ipchannelhourAttrOverCnt':'uint16', 'appdeviceosAttrOverCnt':'uint16', 'appdevicechannelAttrOverCnt':'uint16', 
#  'appdevicehourAttrOverCnt':'uint16', 'apposchannelAttrOverCnt':'uint16', 'apposhourAttrOverCnt':'uint16', 'appchannelhourAttrOverCnt':'uint16', 'deviceoschannelAttrOverCnt':'uint16', 'deviceoshourAttrOverCnt':'uint16', 
#  'devicechannelhourAttrOverCnt':'uint16', 'oschannelhour' : 'uint16',
            }
    for dense_feature in keras_train.DENSE_FEATURE_LIST:
        dtypes[dense_feature] = 'uint16'

    with timer("loading train data"):
        print('loading train data...')
        if FLAGS.load_only_singleCnt:
            path_prefix = "train_part_Cnt_Neg20"
        else:
            path_prefix = "train"
        if FLAGS.debug:
            train_data_path = path + path_prefix + "_Top.ss.csv"
            train_df = pd.read_csv(train_data_path, dtype=dtypes, 
            usecols = ['is_attributed'] + keras_train.USED_FEATURE_LIST)
        else:
            train_data_path = path + path_prefix + ".csv"
            train_df = pd.read_csv(train_data_path, dtype=dtypes, header = None, sep = '\t', 
    names=['is_attributed'] + keras_train.DATA_HEADER, 
    usecols = ['is_attributed'] + keras_train.USED_FEATURE_LIST)
        print(train_df.info())

    with timer("loading valide data"):
        print('loading valide data...')
        if FLAGS.load_only_singleCnt:
            path_prefix = "valide_Cnt"
            if FLAGS.debug:
                valide_data_path = path + path_prefix + "_Top.ss.csv"
                valide_df = pd.read_csv(valide_data_path, dtype=dtypes,
                usecols = ['is_attributed'] + keras_train.USED_FEATURE_LIST)
            else:
                valide_data_path = path + path_prefix + ".csv"
                valide_df = pd.read_csv(valide_data_path, dtype=dtypes, header = None, sep = '\t',
    names=['is_attributed'] + keras_train.DATA_HEADER, #nrows = 10000,
    usecols = ['is_attributed'] + keras_train.USED_FEATURE_LIST)
            print(valide_df.info())

    with timer("loading test data"):
        print('loading test data...')
        if FLAGS.load_only_singleCnt:
            path_prefix = "test_Cnt"
        else:
            path_prefix = "test"
        if FLAGS.debug:
            test_data_path = path + path_prefix + "_Top.ss.csv"
            test_df = pd.read_csv(test_data_path, dtype=dtypes, usecols = ['click_id'] + keras_train.USED_FEATURE_LIST)
        else:
            test_data_path = path + path_prefix + ".csv"
            test_df = pd.read_csv(test_data_path, dtype=dtypes, header = None, sep = '\t', 
        names=['click_id'] + keras_train.DATA_HEADER, nrows = 10000,
        usecols = ['click_id'] + keras_train.USED_FEATURE_LIST)
        print(test_df.info())
        gc.collect()

    with timer("Extracting new features"):
        print('Extracting new features...')
        if not FLAGS.load_only_singleCnt:
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
    if FLAGS.load_only_singleCnt:
        train_data = train_df[keras_train.USED_FEATURE_LIST].values.astype(np.uint16)
        train_label = train_df['is_attributed'].values.astype(np.uint8)
        print ("train type {0}".format(train_data.dtype))
        pos_cnt = train_label.sum()
        neg_cnt = len(train_label) - pos_cnt
        print ("train data pos : {0} neg : {1}".format(pos_cnt, neg_cnt))
        del train_df
        gc.collect()
        valide_data = valide_df[keras_train.USED_FEATURE_LIST].values.astype(np.uint16)
        valide_label = valide_df['is_attributed'].values.astype(np.uint8)
        print ("valide type {0}".format(valide_data.dtype))
        pos_cnt = valide_label.sum()
        neg_cnt = len(valide_label) - pos_cnt
        print ("valide data pos : {0} neg : {1}".format(pos_cnt, neg_cnt))
        del valide_df
        gc.collect()
        test_data = test_df[keras_train.USED_FEATURE_LIST].values.astype(np.uint16)
        test_id = test_df['click_id'].astype('uint32').values.astype(np.uint32)
        print ("test type {0}".format(test_data.dtype))
        del test_df
        gc.collect()
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
    print("train size: ", len(train_data))
    print("valid size: ", len(valide_data))
    print("test size : ", len(test_data))
    return train_data, train_label, test_data, test_id, valide_data, valide_label, weight


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

        # save model to file
        model_name = tmp_model_dir + "model" + time_label + ".txt"
        models[0][0].save_model(model_name)

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