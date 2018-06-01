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
FLAGS = flags.FLAGS

path = FLAGS.input_training_data_path
FLAGS.max_len = [int(l) for l in FLAGS.max_len.strip().split(',')]

def load_data():
    print("\nData Load Stage")
    if FLAGS.debug:
        nrow = 10000
    else:
        nrow = None
    training = pd.read_csv(path + '/train.csv', index_col = "item_id", parse_dates = ["activation_date"], nrows = nrow)
    traindex = training.index
    testing = pd.read_csv(path + '/test.csv', index_col = "item_id", parse_dates = ["activation_date"], nrows = nrow)
    testdex = testing.index

    ntrain = training.shape[0]
    ntest = testing.shape[0]

    y = training.deal_probability.copy()
    training.drop("deal_probability",axis=1, inplace=True)
    print('Train shape: {} Rows, {} Columns'.format(*training.shape))
    print('Test shape: {} Rows, {} Columns'.format(*testing.shape))

    print("Combine Train and Test")
    df = pd.concat([training,testing],axis=0)
    del training, testing
    gc.collect()
    print('\nAll Data shape: {} Rows, {} Columns'.format(*df.shape))

    print("Feature Engineering")
    df["price"] = np.log(df["price"]+0.001)
    df["price"].fillna(-999,inplace=True)
    df["image_top_1"].fillna(-999,inplace=True)

    print("\nCreate Time Variables")
    df["Weekday"] = df['activation_date'].dt.weekday
    df["WeekdOfYear"] = df['activation_date'].dt.week
    df["DayOfMonth"] = df['activation_date'].dt.day

    # Create Validation Index and Remove Dead Variables
    # training_index = df.loc[df.activation_date<=pd.to_datetime('2017-04-07')].index
    # validation_index = df.loc[df.activation_date>=pd.to_datetime('2017-04-08')].index
    df.drop(["activation_date","image"],axis=1,inplace=True)

    print("\nEncode Variables")
    categorical = ["user_id","region","city","parent_category_name","category_name","user_type","image_top_1","param_1","param_2","param_3"]
    print("Encoding :",categorical)

    # Encoder:
    lbl = preprocessing.LabelEncoder()
    for col in categorical:
        df[col].fillna('Unknown')
        df[col] = lbl.fit_transform(df[col].astype(str))
    cat_max = df[keras_train.USED_CATEGORY_FEATURES].max().astype('int64')
    print (cat_max)

    textfeats = ["description", "title"]
    # print(df.head)
    # exit(0)

    if FLAGS.lgb_boost_dnn:
        models = []
        for i in range(FLAGS.lgb_ensemble_nfold):
            bst = lgb.Booster(model_file= FLAGS.input_previous_model_path + '/model_' + str(0) + '_2018_05_31_04_04_47.txt')
            models.append((bst, 'l'))
        df['lgb_pred'] = models_eval(models, df)
        keras_train.USED_FEATURE_LIST += ['lgb_pred']

    emb_weight = None
    if FLAGS.model_type == 'k':
        print('Tokenizer...')
        for cols in textfeats:
            df[cols] = df[cols].astype(str).fillna('missing') # FILL NA
        data = df[textfeats].apply(lambda x: ' '.join(x), axis=1).values
        tokenizer = Tokenizer(num_words = FLAGS.vocab_size)
        tokenizer.fit_on_texts(data)
        for i, cols in enumerate(textfeats):
            data = pad_sequences(tokenizer.texts_to_sequences(df[cols]), maxlen = FLAGS.max_len[i])
            df[cols] = data.tolist()

        if FLAGS.load_wv_model:
            emb_weight = get_word2vec_embedding(location = FLAGS.input_training_data_path + FLAGS.wv_model_file, \
                    tokenizer = tokenizer, nb_words = FLAGS.vocab_size, embed_size = FLAGS.gram_embedding_dim, \
                    model_type = FLAGS.wv_model_type, uniform_init_emb = FLAGS.uniform_init_emb)
        else:
            if FLAGS.uniform_init_emb:
                emb_weight = np.random.uniform(0, 1, (FLAGS.vocab_size, FLAGS.emb_dim))
            else:
                emb_weight = np.zeros((FLAGS.vocab_size, FLAGS.emb_dim))

    # df.drop(textfeats, axis=1,inplace=True)
    print(df.info())
    # df.to_pickle('lgb_pred.pickle')
    # exit(0)

    train_data = df.loc[traindex, keras_train.USED_FEATURE_LIST]
    train_label = y.values
    test_data = df.loc[testdex, keras_train.USED_FEATURE_LIST]
    test_id = testdex
    valide_data = None
    valide_label = None
    weight = None
    return train_data, train_label, test_data, test_id, valide_data, valide_label, weight, cat_max, emb_weight


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
        sub_re = pd.DataFrame(models_eval(models, test),columns=["deal_probability"],index=tid)
        sub_re["deal_probability"].clip(0+1e-6, 1-1e-6, inplace=True)
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
    train_data, train_label, test_data, tid, valide_data, valide_label, weight, cat_max, emb_weight = load_data()
    if not FLAGS.load_stacking_data:
        models, stacking_data, stacking_label, stacking_test_data = nfold_train(train_data, train_label, flags = FLAGS, \
                model_types = [FLAGS.model_type], scores = scores_text, test_data = test_data, \
                valide_data = valide_data, valide_label = valide_label, cat_max = cat_max, emb_weight = emb_weight)
    else:
        for i in range(train_label.shape[1]):
            models, stacking_data, stacking_label, stacking_test_data = nfold_train(train_data, train_label[:, i], flags = FLAGS, \
                model_types = [FLAGS.model_type], scores = scores_text, emb_weight = emb_weight, test_data = test_data \
                # , valide_data = train_data[:100], valide_label = train_label[:100, i]
                )
            sub_re[:, i] = models_eval(models, test_data)
    sub(models, stacking_data = stacking_data, stacking_label = stacking_label, stacking_test_data = stacking_test_data, \
            test = test_data, scores_text = scores_text, tid = tid)