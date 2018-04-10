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
FLAGS = flags.FLAGS


def load_data():
    """
    """
    path = FLAGS.input_training_data_path

    dtypes = {
            'ip'            : 'uint32', 'app'           : 'uint16',
            'device'        : 'uint16', 'os'            : 'uint16',
            'channel'       : 'uint16', 'is_attributed' : 'uint8',
            'click_id'      : 'uint32',

            'ipCnt'         : 'uint32', 'ipAttCnt'      : 'uint16',
            'appCnt'        : 'uint32', 'appAttCnt'     : 'uint32',
            'deviceCnt'     : 'uint32', 'deviceAttCnt'  : 'uint32',
            'osCnt'         : 'uint32', 'osAttCnt'      : 'uint32',
            'channelCnt'    : 'uint32', 'channelAttCnt' : 'uint32',

            'ipAppCnt'      : 'uint32', 'ipAppAttCnt'   : 'uint16',
            'ipDeviceCnt'   : 'uint32', 'ipDeviceAttCnt': 'uint16',
            'ipOsCnt'       : 'uint32', 'ipOsAttCnt'    : 'uint16',
            'ipChannelCnt'  : 'uint32', 'ipChannelAttCnt'   : 'uint16',
            'appDeviceCnt'  : 'uint32', 'appDeviceAttCnt'   : 'uint32',
            'appOsCnt'      : 'uint32', 'appOsAttCnt'   : 'uint16',
            'appChannelCnt' : 'uint32', 'appChannelAttCnt'  : 'uint32',
            'deviceOsCnt'   : 'uint32', 'deviceOsAttCnt'    : 'uint32',
            'deviceChannelCnt'  : 'uint32', 'deviceChannelAttCnt' : 'uint16',
            'osChannelCnt'      : 'uint32', 'osChannelAttCnt'     : 'uint16',

            'ipappdeviceCnt'        : 'uint32', 'ipappdeviceAttCnt'     : 'uint16', 
            'ipapposCnt'            : 'uint16', 'ipapposAttCnt'         : 'uint16', 
            'ipappchannelCnt'       : 'uint32', 'ipappchannelAttCnt'    : 'uint16', 
            'ipdeviceosCnt'         : 'uint32', 'ipdeviceosAttCnt'      : 'uint16', 
            'ipdevicechannelCnt'    : 'uint32', 'ipdevicechannelAttCnt' : 'uint16', 
            'iposchannelCnt'        : 'uint16', 'iposchannelAttCnt'     : 'uint8', 
            'appdeviceosCnt'        : 'uint32', 'appdeviceosAttCnt'     : 'uint16', 
            'appdevicechannelCnt'   : 'uint32', 'appdevicechannelAttCnt': 'uint16', 
            'apposchannelCnt'       : 'uint32', 'apposchannelAttCnt'    : 'uint16', 
            'deviceoschannelCnt'    : 'uint32', 'deviceoschannelAttCnt' : 'uint16', 

            'ipappdeviceosCnt'      : 'uint16', 'ipappdeviceosAttCnt'   : 'uint16', 
            'ipappdevicechannelCnt' : 'uint32', 'ipappdevicechannelAttCnt'      : 'uint16', 
            'ipapposchannelCnt'      : 'uint16', 'ipapposchannelAttCnt'      : 'uint8', 
            'ipdeviceoschannelCnt'      : 'uint16', 'ipdeviceoschannelAttCnt'      : 'uint8', 
            'appdeviceoschannelCnt'      : 'uint32', 'appdeviceoschannelAttCnt' : 'uint16', 
            }

    with timer("loading train data"):
        print('loading train data...')
        if FLAGS.load_only_singleCnt:
            path_prefix = "train_part_Cnt_Neg20"
        else:
            path_prefix = "train_part_Cnt_Neg20"
        if FLAGS.debug:
            train_data_path = path + path_prefix + "_sample.csv"
        else:
            train_data_path = path + path_prefix + ".csv"
        if FLAGS.load_only_singleCnt:
            train_df = pd.read_csv(train_data_path, \
                    dtype=dtypes, header = None, sep = '\t', 
                    #nrows=40000000,  
                    #skiprows = 9970280, \
                    names=['is_attributed', 'ipCnt', 'ipAttCnt', 'appCnt', 'appAttCnt', 'deviceCnt', 'deviceAttCnt', 'osCnt', 'osAttCnt', 'channelCnt', 'channelAttCnt', 
                        'ipAppCnt','ipAppAttCnt','ipDeviceCnt','ipDeviceAttCnt','ipOsCnt','ipOsAttCnt','ipChannelCnt','ipChannelAttCnt','appDeviceCnt','appDeviceAttCnt',
                        'appOsCnt','appOsAttCnt','appChannelCnt','appChannelAttCnt','deviceOsCnt','deviceOsAttCnt','deviceChannelCnt','deviceChannelAttCnt','osChannelCnt',
                        'osChannelAttCnt', 
                        'ipappdeviceCnt','ipappdeviceAttCnt','ipapposCnt','ipapposAttCnt','ipappchannelCnt','ipappchannelAttCnt','ipdeviceosCnt',
                        'ipdeviceosAttCnt','ipdevicechannelCnt','ipdevicechannelAttCnt','iposchannelCnt','iposchannelAttCnt','appdeviceosCnt','appdeviceosAttCnt',
                        'appdevicechannelCnt','appdevicechannelAttCnt','apposchannelCnt','apposchannelAttCnt','deviceoschannelCnt','deviceoschannelAttCnt',
                        'ipappdeviceosCnt','ipappdeviceosAttCnt','ipappdevicechannelCnt','ipappdevicechannelAttCnt','ipapposchannelCnt','ipapposchannelAttCnt',
                        'ipdeviceoschannelCnt','ipdeviceoschannelAttCnt','appdeviceoschannelCnt','appdeviceoschannelAttCnt'])
            #train_df.fillna(0)
        else:
            train_df = pd.read_csv(train_data_path, \
                    # skiprows=range(1,144903891), nrows=40000000, \
                    dtype=dtypes, usecols=['ip','app','device','os', 'channel', 'click_time', 'is_attributed'])
        print(train_df.info())

    with timer("loading valide data"):
        print('loading valide data...')
        if FLAGS.load_only_singleCnt:
            path_prefix = "valide_Cnt"
            if FLAGS.debug:
                valide_data_path = path + path_prefix + "_sample.csv"
            else:
                valide_data_path = path + path_prefix + ".csv"
            valide_df = pd.read_csv(valide_data_path, \
                        dtype=dtypes, header = None, sep = '\t',
                        names=['is_attributed', 'ipCnt', 'ipAttCnt', 'appCnt', 'appAttCnt', 'deviceCnt', 'deviceAttCnt', 'osCnt', 'osAttCnt', 'channelCnt', 'channelAttCnt', 
                        'ipAppCnt','ipAppAttCnt','ipDeviceCnt','ipDeviceAttCnt','ipOsCnt','ipOsAttCnt','ipChannelCnt','ipChannelAttCnt','appDeviceCnt','appDeviceAttCnt',
                        'appOsCnt','appOsAttCnt','appChannelCnt','appChannelAttCnt','deviceOsCnt','deviceOsAttCnt','deviceChannelCnt','deviceChannelAttCnt','osChannelCnt',
                        'osChannelAttCnt', 
                        'ipappdeviceCnt','ipappdeviceAttCnt','ipapposCnt','ipapposAttCnt','ipappchannelCnt','ipappchannelAttCnt','ipdeviceosCnt',
                        'ipdeviceosAttCnt','ipdevicechannelCnt','ipdevicechannelAttCnt','iposchannelCnt','iposchannelAttCnt','appdeviceosCnt','appdeviceosAttCnt',
                        'appdevicechannelCnt','appdevicechannelAttCnt','apposchannelCnt','apposchannelAttCnt','deviceoschannelCnt','deviceoschannelAttCnt',
                        'ipappdeviceosCnt','ipappdeviceosAttCnt','ipappdevicechannelCnt','ipappdevicechannelAttCnt','ipapposchannelCnt','ipapposchannelAttCnt',
                        'ipdeviceoschannelCnt','ipdeviceoschannelAttCnt','appdeviceoschannelCnt','appdeviceoschannelAttCnt'])

    with timer("loading test data"):
        print('loading test data...')
        if FLAGS.load_only_singleCnt:
            path_prefix = "test_Cnt"
        else:
            path_prefix = "test_Cnt"
        if FLAGS.debug:
            test_data_path = path + path_prefix + "_sample.csv"
        else:
            test_data_path = path + path_prefix + ".csv"
        if FLAGS.load_only_singleCnt:
            test_df = pd.read_csv(test_data_path, \
                    dtype=dtypes, header = None, sep = '\t', 
                    # skiprows = 8609683, \
                    names=['click_id', 'ipCnt', 'ipAttCnt', 'appCnt', 'appAttCnt', 'deviceCnt', 'deviceAttCnt', 'osCnt', 'osAttCnt', 'channelCnt', 'channelAttCnt', 
                        'ipAppCnt','ipAppAttCnt','ipDeviceCnt','ipDeviceAttCnt','ipOsCnt','ipOsAttCnt','ipChannelCnt','ipChannelAttCnt','appDeviceCnt','appDeviceAttCnt',
                        'appOsCnt','appOsAttCnt','appChannelCnt','appChannelAttCnt','deviceOsCnt','deviceOsAttCnt','deviceChannelCnt','deviceChannelAttCnt','osChannelCnt',
                        'osChannelAttCnt', 
                        'ipappdeviceCnt','ipappdeviceAttCnt','ipapposCnt','ipapposAttCnt','ipappchannelCnt','ipappchannelAttCnt','ipdeviceosCnt',
                        'ipdeviceosAttCnt','ipdevicechannelCnt','ipdevicechannelAttCnt','iposchannelCnt','iposchannelAttCnt','appdeviceosCnt','appdeviceosAttCnt',
                        'appdevicechannelCnt','appdevicechannelAttCnt','apposchannelCnt','apposchannelAttCnt','deviceoschannelCnt','deviceoschannelAttCnt',
                        'ipappdeviceosCnt','ipappdeviceosAttCnt','ipappdevicechannelCnt','ipappdevicechannelAttCnt','ipapposchannelCnt','ipapposchannelAttCnt',
                        'ipdeviceoschannelCnt','ipdeviceoschannelAttCnt','appdeviceoschannelCnt','appdeviceoschannelAttCnt'])
            #test_df.fillna(0)
        else:
            test_df = pd.read_csv(test_data_path, \
                    dtype=dtypes, usecols=['ip','app','device','os', 'channel', 'click_time', 'click_id'])
        len_train = len(train_df)
        if not FLAGS.load_only_singleCnt:
            train_df=train_df.append(test_df)
            del test_df
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
        train_data = train_df[keras_train.DENSE_FEATURE_LIST].values
        train_label = train_df['is_attributed'].values
        del train_df
        valide_data = valide_df[keras_train.DENSE_FEATURE_LIST].values
        valide_label = valide_df['is_attributed'].values
        del valide_df
        test_data = test_df[keras_train.DENSE_FEATURE_LIST].values
        test_id = test_df['click_id'].astype('uint32').values
        del test_df
    else:
        valide_len = len_train // 5
        train_data = train_df[:len_train - valide_len][keras_train.SPARSE_FEATURE_LIST].values
        train_label = train_df[:len_train - valide_len]['is_attributed'].values
        valide_data = train_df[len_train - valide_len:len_train][keras_train.SPARSE_FEATURE_LIST].values
        valide_label = train_df[len_train - valide_len:len_train]['is_attributed'].values

        test_data = train_df[len_train:][keras_train.SPARSE_FEATURE_LIST].values
        test_id = train_df[len_train:]['click_id'].astype('uint32').values
        del train_df
    gc.collect()

    weight = None
    if FLAGS.neg_sample:
        train_data, train_label, weight = neg_sample(train_data, train_label, FLAGS.sample_C)
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
    if not FLAGS.load_only_singleCnt:
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