
from sklearn import *
import sklearn
import pandas as pd
import numpy as np
import xgboost as xgb
import lightgbm as lgb
from time import gmtime, strftime
import numpy.random as rng
from multiprocessing.dummy import Pool
import h5py
import concurrent.futures
import tensorflow as tf
import multiprocessing as mp

from sklearn.cross_validation import KFold
from keras.models import Sequential, Model
from keras.layers.core import Dense, Dropout, Flatten, Reshape
from keras.layers.normalization import BatchNormalization
from keras.layers.embeddings import Embedding
from keras.layers import Input, concatenate, merge, LSTM
from keras.layers.convolutional import Convolution2D, MaxPooling2D, ZeroPadding2D, AveragePooling2D
from keras.optimizers import SGD, RMSprop, Adam
from keras.callbacks import EarlyStopping
from keras.utils import np_utils
from keras import backend as K
from sklearn.metrics import log_loss
from keras import __version__ as keras_version
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras import regularizers

# DNN_PARAMS
HIDDEN_UNITS = [40, 20, 4]
DNN_EPOCHS = 40
BATCH_SIZE = 5
DNN_BN = True
DROPOUT_RATE = 0.5
SIAMESE_PAIR_SIZE = 100000
MAX_WORKERS = 8
EMBEDDING_SIZE = 6

# RNN_PARAMS
MAX_NUM_WORDS = 2000
RNN_EMBEDDING_DIM = 10
MAX_SEQUENCE_LEN = 1000
LSTM_OUT = 32

full_feature = True

data_folder = '../Data/'
train = pd.read_csv(data_folder + 'training_variants')
#print train.dtypes
test = pd.read_csv(data_folder + 'test_variants')
trainx = pd.read_csv(data_folder + 'training_text', sep="\|\|", engine='python', header=None, skiprows=1, names=["ID","Text"])
#print trainx.dtypes
testx = pd.read_csv(data_folder + 'test_text', sep="\|\|", engine='python', header=None, skiprows=1, names=["ID","Text"])

train = pd.merge(train, trainx, how='left', on='ID').fillna('')
#train = train.iloc[1:1000]
y = train['Class'].values
train = train.drop(['Class'], axis=1)
train_text = train['Text'].values

test = pd.merge(test, testx, how='left', on='ID').fillna('')
pid = test['ID'].values

#df_all = pd.concat((train, test), axis=0, ignore_index=True)
#df_all['Gene_Share'] = df_all.apply(lambda r: sum([1 for w in r['Gene'].split(' ') if w in r['Text'].split(' ')]), axis=1).astype(np.int8)
#df_all['Variation_Share'] = df_all.apply(lambda r: sum([1 for w in r['Variation'].split(' ') if w in r['Text'].split(' ')]), axis=1).astype(np.int8)
#
#print df_all[['Gene_Share', 'Variation_Share']].max()
## exit(0)
#if full_feature:
#    #commented for Kaggle Limits
#    for i in range(5):
#        df_all['Gene_'+str(i)] = df_all['Gene'].map(lambda x: str(x[i]) if len(x)>i else '')
#        df_all['Variation'+str(i)] = df_all['Variation'].map(lambda x: str(x[i]) if len(x)>i else '')
#    print df_all.dtypes
#
#    gen_var_lst = sorted(list(train.Gene.unique()) + list(train.Variation.unique()))
#    print(len(gen_var_lst))
#    gen_var_lst = [x for x in gen_var_lst if len(x.split(' '))==1]
#    print(len(gen_var_lst))
#    i_ = 0
#    #commented for Kaggle Limits
#    for gen_var_lst_itm in gen_var_lst:
#        if i_ % 100 == 0: print(i_)
#        df_all['GV_'+str(gen_var_lst_itm)] = df_all['Text'].map(lambda x: str(x).count(str(gen_var_lst_itm))).astype(np.int8)
#        i_ += 1
#        if i_ == 5:
#            break
#
#for c in df_all.columns:
#    if df_all[c].dtype == 'object':
#        if c in ['Gene','Variation']:
#            lbl = preprocessing.LabelEncoder()
#            df_all[c+'_lbl_enc'] = lbl.fit_transform(df_all[c].values)
#            df_all[c+'_len'] = df_all[c].map(lambda x: len(str(x)))
#            df_all[c+'_words'] = df_all[c].map(lambda x: len(str(x).split(' ')))
#        elif c != 'Text':
#            lbl = preprocessing.LabelEncoder()
#            df_all[c] = lbl.fit_transform(df_all[c].values)
#        if c=='Text':
#            df_all[c+'_len'] = df_all[c].map(lambda x: len(str(x)))
#            df_all[c+'_words'] = df_all[c].map(lambda x: len(str(x).split(' ')))
#
#train = df_all.iloc[:len(train)]
#print "... train dtypes before svd ..."
#print train.dtypes
#print train.head()
#exit(0)
#test = df_all.iloc[len(train):]
#
#class cust_regression_vals(sklearn.base.BaseEstimator, sklearn.base.TransformerMixin):
#    def fit(self, x, y=None):
#        return self
#    def transform(self, x):
#        x = x.drop(['Gene', 'Variation','ID','Text'],axis=1).values
#        return x
#
#class cust_txt_col(sklearn.base.BaseEstimator, sklearn.base.TransformerMixin):
#    def __init__(self, key):
#        self.key = key
#    def fit(self, x, y=None):
#        return self
#    def transform(self, x):
#        return x[self.key].apply(str)
#
#print('Pipeline...')
#fp = pipeline.Pipeline([
#    ('union', pipeline.FeatureUnion(
#        n_jobs = -1,
#        transformer_list = [
#            ('standard', cust_regression_vals()),
#            ('pi1', pipeline.Pipeline([('Gene', cust_txt_col('Gene')), ('count_Gene', feature_extraction.text.CountVectorizer(analyzer=u'char', ngram_range=(1, 8))), ('tsvd1', decomposition.TruncatedSVD(n_components=20, n_iter=25, random_state=12))])),
#            ('pi2', pipeline.Pipeline([('Variation', cust_txt_col('Variation')), ('count_Variation', feature_extraction.text.CountVectorizer(analyzer=u'char', ngram_range=(1, 8))), ('tsvd2', decomposition.TruncatedSVD(n_components=20, n_iter=25, random_state=12))])),
#            #commented for Kaggle Limits
#            ('pi3', pipeline.Pipeline([('Text', cust_txt_col('Text')), ('tfidf_Text', feature_extraction.text.TfidfVectorizer(ngram_range=(1, 2))), ('tsvd3', decomposition.TruncatedSVD(n_components=50, n_iter=25, random_state=12))]))
#        ])
#    )])
#
#train = fp.fit_transform(train);
#print type(train)
#print(train.shape)
#print (train.nbytes)
#np.save("train_array", train)
## print(df.dtypes)
## print(df.memory_usage())
#test = fp.transform(test); print(test.shape)
#np.save("test_array", test)
#exit(0)
train = np.load("./train_array.npy")
test = np.load("./test_array.npy")
# siamese_features_array = np.load("./siamese_features_array_2017_09_15_07_57_44.npy")
y = y - 1 #fix for zero bound array

kf = KFold(len(y), n_folds=10, shuffle=True, random_state=1)
for TRAIN_INDEX, TEST_INDEX in kf:
    TRAIN_DATA = train[TRAIN_INDEX]
    TRAIN_LABEL = y[TRAIN_INDEX]
    VALIDE_DATA = train[TEST_INDEX]
    VALIDE_LABEL = y[TEST_INDEX]
    break

CONTINUOUS_INDICES = []
SPARSE_INDICES = []

for i in range((train.shape)[1]):
    if (i >= 3205 and i <= 3212):
        pass
    elif (i >= 2 and i <= 113): # or (i >= 114 and i <= 3204):
        SPARSE_INDICES.append(i)
    else:
        CONTINUOUS_INDICES.append(i)
#train = train[:, CONTINUOUS_INDICES]
#test = test[:, CONTINUOUS_INDICES]

print('train shape after loading and selecting trainging columns: %s' % str(train.shape))

siamese_train_len = len(train) // 3
print('siamese_train_len is %d' % (siamese_train_len))
siamese_train_data = train[:siamese_train_len]
siamese_train_label = y[:siamese_train_len]

lgbm_train_data = train[siamese_train_len:]
lgbm_train_label = y[siamese_train_len:]
#train = train[:200]
#y = y[:200]
#test = test[:200]
#pid = pid[:200]

def xgbTrain(train_data, train_label, fold = 5, valide_data = None, valide_label = None):
    """
    """
    denom = 0
    # fold = 5 #Change to 5, 1 for Kaggle Limits
    models = []
    for i in range(fold):
        params = {
            'eta': 0.03333,
            'max_depth': 4,
            'objective': 'multi:softprob',
            'eval_metric': 'mlogloss',
            'num_class': 9,
            'seed': i,
            'silent': True
        }
        if valide_data is None:
            x1, x2, y1, y2 = model_selection.train_test_split(train_data, train_label, test_size=0.18, random_state=i)
        else:
            x1, x2, y1, y2 = train_data, valide_data, train_label, valide_label
        watchlist = [(xgb.DMatrix(x1, y1), 'train'), (xgb.DMatrix(x2, y2), 'valid')]
        model = xgb.train(params, xgb.DMatrix(x1, y1), 1000,  watchlist, verbose_eval=50, early_stopping_rounds=100)
        score1 = metrics.log_loss(y2, model.predict(xgb.DMatrix(x2), ntree_limit=model.best_ntree_limit), labels = list(range(9)))
        #print(score1)

        models.append((model, 'x'))

    return models


def lgbm_train(train_data, train_label, fold = 5, valide_data = None, valide_label = None):
    """
    LGB Training
    """
    print("Over all training size:")
    print(train_data.shape)

    kf = KFold(len(train_label), n_folds=fold, shuffle=True)

    num_fold = 0
    models = []
    for train_index, test_index in kf:
        if valide_label is None:
            d_train = lgb.Dataset(train_data[train_index], train_label[train_index]) #, categorical_feature = SPARCE_INDICES)
            d_valide = lgb.Dataset(train_data[test_index], train_label[test_index])
        else:
            d_train = lgb.Dataset(train_data, train_label) #, categorical_feature = SPARCE_INDICES)
            d_valide = lgb.Dataset(valide_data, valide_label)

        params = {
            'task': 'train',
            'boosting_type': 'gbdt',
            'objective': 'multiclass',
            'metric': {'multi_logloss'},
            'num_class': 9,
          #  'num_leaves': 256,
          #  'max_depth': 12,
          #  'feature_fraction': 0.9,
          #  'bagging_fraction': 0.95,
          #  'bagging_freq': 5,
            'num_leaves': 60, # 60,
          #  'min_sum_hessian_in_leaf': 20,
            'max_depth': 10, # 10,
            'learning_rate': 0.028, # 0.025,
           'feature_fraction': 0.5, # 0.6
            'verbose': 0,
          #   'valid_sets': [d_valide],
            'num_boost_round': 405,
            'feature_fraction_seed': num_fold,
            # 'bagging_fraction': 0.9,
            # 'bagging_freq': 15,
            # 'bagging_seed': i,
            'early_stopping_round': 10,
            # 'random_state': 10
            # 'verbose_eval': 20
            #'min_data_in_leaf': 665
        }

        print('fold: %d th light GBM train :-)' % (num_fold))
        num_fold += 1
        bst = lgb.train(
                        params ,
                        d_train,
                        verbose_eval = 50,
                        valid_sets = [d_valide]
                        #num_boost_round = 1
                        )
        #cv_result = lgb.cv(params, d_train, nfold=10)
        #pd.DataFrame(cv_result).to_csv('cv_result', index = False)
        # exit(0)
        models.append((bst, 'l'))

    return models


def create_model(input_len):
    model = Sequential()
    model.add(Dense(HIDDEN_UNITS[1], activation='sigmoid', input_dim = input_len))
    if DNN_BN:
        model.add(BatchNormalization())
    if DROPOUT_RATE > 0:
        model.add(Dropout(DROPOUT_RATE))
   # model.add(Dense(HIDDEN_UNITS[2], activation='sigmoid'))
   # if DNN_BN:
   #     model.add(BatchNormalization())
   # if DROPOUT_RATE > 0:
   #     model.add(Dropout(DROPOUT_RATE))
   # model.add(Dense(HIDDEN_UNITS[2], activation='sigmoid'))
   # if DNN_BN:
   #     model.add(BatchNormalization())
   # if DROPOUT_RATE > 0:
   #     model.add(Dropout(DROPOUT_RATE))
    # model.add(Dropout(0.1))
    #model.add(Dense(32, activation='relu'))
    #model.add(Dropout(0.2))
    model.add(Dense(9, activation='softmax'))

    # optimizer = SGD(lr=1e-3, decay=1e-6, momentum=0.9, nesterov=True)
    optimizer = RMSprop(lr=1e-3, rho = 0.9, epsilon = 1e-8)
    model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics = ['accuracy'])

    return model


def create_lr_model(input_len):
    model = Sequential()
    model.add(Dense(9, activation='softmax', input_dim = input_len, kernel_regularizer=regularizers.l2(0.01),
                        activity_regularizer=regularizers.l1(0.01)))

    # optimizer = SGD(lr=1e-3, decay=1e-6, momentum=0.9, nesterov=True)
    # optimizer = RMSprop(lr=1e-3, rho = 0.9, epsilon = 1e-8)
    model.compile(optimizer=Adam(), loss='categorical_crossentropy', metrics = ['accuracy'])

    return model


def create_embedding_model(CONTINUE_SIZE, SPARSE_SIZE):
    """
    """
    print('CONTINUOUS_SIZE = %d' % CONTINUE_SIZE)
    print('SPARSE_SIZE = %d' % SPARSE_SIZE)
    sparse_feature = Input(shape=(SPARSE_SIZE,))
    sparse_embedding = Embedding(55, EMBEDDING_SIZE, input_length = SPARSE_SIZE)(sparse_feature)
    sparse_embedding = Reshape((EMBEDDING_SIZE * SPARSE_SIZE,))(sparse_embedding)

    # print "model input size: %d" % CONTINUOUS_COLUMNS
    dense_input = Input(shape=(CONTINUE_SIZE,))
    # dense_input = BatchNormalization()(dense_input)
    dense_layer = Dense(HIDDEN_UNITS[0], activation='sigmoid')(dense_input)
    dense_layer = BatchNormalization()(dense_layer)
    dense_layer = Dropout(DROPOUT_RATE)(dense_layer)
    merge_input = concatenate([dense_layer, sparse_embedding], axis = 1)

    merge_len = HIDDEN_UNITS[0] + EMBEDDING_SIZE * SPARSE_SIZE
    output = create_model(merge_len)(merge_input)

    model = Model([dense_input, sparse_feature], output)
    optimizer = RMSprop(lr=1e-3, rho = 0.9, epsilon = 1e-8)
    # optimizer = SGD(lr=1e-2, decay=1e-6, momentum=0.9, nesterov=True)
    model.compile(optimizer = Adam(),
            loss='categorical_crossentropy', metrics = ['accuracy'])

    return model

def gen_dnn_input(input_array):
    """
    """
    return [input_array[:, CONTINUOUS_INDICES], input_array[:, SPARSE_INDICES]]


def keras_train(train_data, train_label, nfolds = 10, valide_data = None, valide_label = None, model_type = 'DNN'):
    """
    Detect Fish or noFish
    """
    print("Start gen training data, shuffle and normalize!")

    #train_data = train
    train_label = np_utils.to_categorical(train_label)
    if valide_label is not None:
        valide_label = np_utils.to_categorical(valide_label)
    # train_data, train_label, siamese_data_loader = siamese_train(siamese_train_data, siamese_train_label)
    kf = KFold(len(train_label), n_folds=nfolds, shuffle=True)

    num_fold = 0
    models = []
    for train_index, test_index in kf:
        # model = create_model(classes = 2)
        # model = create_siamese_net((train.shape)[1])

        if valide_data is None:
            X_train = train_data[train_index]
            Y_train = train_label[train_index]
        else:
            X_train = train_data
            Y_train = train_label
        print('Positive samples in train: %d' % np.sum(Y_train))
        print('Negative samples in train: %d' % (len(Y_train) - np.sum(Y_train)))

        if valide_data is None:
            X_valid = train_data[test_index]
            Y_valid = train_label[test_index]
        else:
            X_valid = valide_data
            Y_valid = valide_label

        print('Positive samples in valide: %d' % np.sum(Y_valid))
        print('Negative samples in valide: %d' % (len(Y_valid) - np.sum(Y_valid)))

        num_fold += 1
        print('Start KFold number {} from {}'.format(num_fold, nfolds))
        print('Split train: ', len(X_train), len(Y_train))
        print('Split valid: ', len(X_valid), len(Y_valid))

        if model_type == 'DNN':
            model = create_embedding_model(len(CONTINUOUS_INDICES), len(SPARSE_INDICES))
            X_train = gen_dnn_input(X_train)
            X_valid = gen_dnn_input(X_valid)
        elif model_type == 'LR':
            model = create_lr_model(X_train.shape[1])
        else:
            print('unknown keras model')
            exit(1)

        callbacks = [
            EarlyStopping(monitor='val_loss', patience=3, verbose=0),
        ]
        model.fit(X_train,
                Y_train, batch_size=BATCH_SIZE, epochs=DNN_EPOCHS,
                shuffle=True, verbose=2,
                validation_data=(X_valid, Y_valid)
                , callbacks=callbacks)

        model_name = 'keras' + strftime('_%Y_%m_%d_%H_%M_%S', gmtime())
        #model.save_weights(model_name)
        #siamese_features_array = gen_siamese_features(model, lgbm_train_data, siamese_train_data, siamese_train_label)
        models.append((model, 'k'))
        #if len(models) == 5:
        #    break

    return models #, siamese_features_array


def gen_rnn_input(input_text, max_num_words, max_len):
    """
    Gen rnn input sequence
    """
    tokenizer = Tokenizer(num_words = max_num_words)
    tokenizer.fit_on_texts(input_text)
    # Pad the data
    print("RNN: Convert text to indice sequence!")
    output_sequences = tokenizer.texts_to_sequences(input_text)
    output_sequences = pad_sequences(output_sequences, maxlen= max_len)
    return output_sequences


def create_rnn(input_sequence_size, max_num_words, embedding_dim, lstm_out):
    """
    """
    model = Sequential()
    model.add(Embedding(max_num_words, embedding_dim, input_length = input_sequence_size))
    model.add(LSTM(lstm_out, recurrent_dropout=0.2, dropout=0.2))
    model.add(Dense(9,activation='softmax'))
    model.compile(loss = 'categorical_crossentropy', optimizer='adam', metrics = ['categorical_crossentropy'])
    return model


def rnn_train(train_data, train_label, fold = 5):
    """
    """
    print("Start gen training data, shuffle and normalize!")

    train_target = np_utils.to_categorical(train_label)

    # train_data, train_target, siamese_data_loader = siamese_train(siamese_train_data, siamese_train_label)
    kf = KFold(len(train_target), n_folds=fold, shuffle=True)

    num_fold = 0
    models = []
    for train_index, test_index in kf:
        # model = create_model(classes = 2)
        model = create_rnn(MAX_SEQUENCE_LEN, MAX_NUM_WORDS, RNN_EMBEDDING_DIM, LSTM_OUT)
        # model = create_siamese_net((train.shape)[1])

        X_train = train_data[train_index]
        Y_train = train_target[train_index]
        print('Positive samples in train: %d' % np.sum(Y_train))
        print('Negative samples in train: %d' % (len(Y_train) - np.sum(Y_train)))

        X_valid = train_data[test_index]
        Y_valid = train_target[test_index]
        print('Positive samples in valide: %d' % np.sum(Y_valid))
        print('Negative samples in valide: %d' % (len(Y_valid) - np.sum(Y_valid)))

        num_fold += 1
        print('Start KFold number {} from {}'.format(num_fold, fold))
        print('Split train: ', len(X_train), len(Y_train))
        print('Split valid: ', len(X_valid), len(Y_valid))

        callbacks = [
            EarlyStopping(monitor='val_loss', patience=3, verbose=0),
        ]
        model.fit(X_train,
                Y_train, batch_size=BATCH_SIZE, epochs=DNN_EPOCHS,
                shuffle=True, verbose=2,
                validation_data=(X_valid, Y_valid)
                , callbacks=callbacks)

        model_name = 'keras' + strftime('_%Y_%m_%d_%H_%M_%S', gmtime())
        #model.save_weights(model_name)
        #siamese_features_array = gen_siamese_features(model, lgbm_train_data, siamese_train_data, siamese_train_label)
        models.append((model, 'k'))
        # break

    return models #, siamese_features_array


def model_eval(model, model_type, data_frame):
    """
    """
    if model_type == 'l':
        preds = model.predict(data_frame)
    elif model_type == 'k':
        preds = model.predict(data_frame, batch_size=BATCH_SIZE, verbose=2)
    elif model_type == 't':
        print("ToDO")
    elif model_type == 'x':
        preds = model.predict(xgb.DMatrix(data_frame), ntree_limit=model.best_ntree_limit+80)
    return preds


def models_eval(models, data):
    preds = None
    for (model, model_type) in models:
        pred = model_eval(model, model_type, data)
        if preds is None:
            preds = pred.copy()
        else:
            preds += pred
    preds /= len(models)
    return preds


def gen_sub(models, merge_features):
    """
    Evaluate single Type model
    """
    print('Start generate submission!')
    preds = models_eval(models, merge_features)
    submission = pd.DataFrame(preds, columns=['class'+str(c+1) for c in range(9)])
    submission['ID'] = pid
    sub_name = "submission" + strftime('_%Y_%m_%d_%H_%M_%S', gmtime()) + ".csv"
    print('Output to ' + sub_name)
    submission.to_csv(sub_name, index=False)


#def linear_combine(preds_array, preds, labels, remain_linear_sum, linear_array, min_loss, ind, min_linear_array):
#    """
#    """
#    if (len(preds_array) == ind):
#        loss = metrics.logloss(labels, preds)
#        if min_loss > loss:
#            min_linear_array = linear_array.copy()
#        return
#    if ind == len(preds_array) - 1:
#        linear_array[ind] = remain_linear_sum
#        preds += remain_linear_sum * preds_array[ind]
#        remain_linear_sum = 0
#    else:
#        for ll in range(0, remain_linear_sum * 10 + 1):
#            ll *= 0.1
#            linear_array[ind] = ll
#            preds += ll * preds_array[ind]

def linear_combine(preds_array, labels):
    """
    """
    min_loss = 10000
    opt_l = 0
    opt_r = 0
    for l in range(0, 11):
        l *= 0.1
        r = 1 - l
        preds = l * preds_array[0] + r * preds_array[1]
        loss = metrics.log_loss(labels, preds)
        if loss < min_loss:
            min_loss = loss
            opt_l = l
            opt_r = r
            print('min_loss: {} opt_l: {} opt_r: {}'.format(min_loss, opt_l, opt_r))
    # print('min_loss: {} opt_l: {} opt_r: {}'.format(min_loss, opt_l, opt_r))
    return opt_l, opt_r


if __name__ == "__main__":
    #rnn_input = gen_rnn_input(train_text, MAX_NUM_WORDS, MAX_SEQUENCE_LEN)
    #model_k = rnn_train(rnn_input, y, 10)
    #exit(0)
    #model_k = keras_train(TRAIN_DATA, TRAIN_LABEL, 2, VALIDE_DATA, VALIDE_LABEL, 'DNN')
    #keras_preds_train = models_eval(model_k, gen_dnn_input(train))
    #np.save("keras_preds" + \
    #        strftime('_%Y_%m_%d_%H_%M_%S', gmtime()) , keras_preds_train)
    # keras_preds_train = np.load('keras_preds_2017_09_24_16_28_23.npy')
    model_l = lgbm_train(TRAIN_DATA, TRAIN_LABEL, 2, VALIDE_DATA, VALIDE_LABEL)
    lgbm_preds_train = models_eval(model_l, train)
    model_x = xgbTrain(TRAIN_DATA, TRAIN_LABEL, 2, VALIDE_DATA, VALIDE_LABEL)
    xgb_preds_train = models_eval(model_x, train)
    # merge_train_data = np.concatenate((keras_preds_train, lgbm_preds_train), axis = 1)
    linear_combine(np.array([xgb_preds_train[TEST_INDEX], lgbm_preds_train[TEST_INDEX]]), VALIDE_LABEL)
    # model_k = keras_train(merge_train_data[TRAIN_INDEX], TRAIN_LABEL, 2, merge_train_data[TEST_INDEX], VALIDE_LABEL, 'LR')
    exit(0)
    print(train.shape)
    print(keras_preds_train.shape)
    merge_train_data = np.concatenate((train, keras_preds_train), axis = 1)
    model_l = lgbm_train(merge_train_data[TRAIN_INDEX], TRAIN_LABEL, 10, merge_train_data[TEST_INDEX], VALIDE_LABEL)
    #model_x = xgbTrain(merge_train_data, y, 5)#model_k)

    ## predict on test and sub
    #keras_preds_test = models_eval(model_k, gen_dnn_input(test))
    #merge_test_data = np.concatenate((test, keras_preds_test), axis = 1)
    #gen_sub(model_l, merge_test_data)
