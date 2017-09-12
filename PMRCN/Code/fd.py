
from sklearn import *
import sklearn
import pandas as pd
import numpy as np
import xgboost as xgb
import lightgbm as lgb
from time import gmtime, strftime
import numpy.random as rng

from sklearn.cross_validation import KFold
from keras.models import Sequential, Model
from keras.layers.core import Dense, Dropout, Flatten, Reshape
from keras.layers.normalization import BatchNormalization
from keras.layers.embeddings import Embedding
from keras.layers import Input, concatenate, merge
from keras.layers.convolutional import Convolution2D, MaxPooling2D, ZeroPadding2D, AveragePooling2D
from keras.optimizers import SGD, RMSprop, Adam
from keras.callbacks import EarlyStopping
from keras.utils import np_utils
from keras import backend as K
from sklearn.metrics import log_loss
from keras import __version__ as keras_version

HIDDEN_UNITS = [128, 32]
DNN_EPOCHS = 40
BATCH_SIZE = 5
DNN_BN = True
DROPOUT_RATE = 0

full_feature = True

data_folder = '../Data/'
train = pd.read_csv(data_folder + 'training_variants')
print train.dtypes
test = pd.read_csv(data_folder + 'test_variants')
trainx = pd.read_csv(data_folder + 'training_text', sep="\|\|", engine='python', header=None, skiprows=1, names=["ID","Text"])
print trainx.dtypes
testx = pd.read_csv(data_folder + 'test_text', sep="\|\|", engine='python', header=None, skiprows=1, names=["ID","Text"])

train = pd.merge(train, trainx, how='left', on='ID').fillna('')
#train = train.iloc[1:1000]
print train.dtypes
y = train['Class'].values
train = train.drop(['Class'], axis=1)

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
y = y - 1 #fix for zero bound array

CONTINUOUS_INDICES = []
SPARCE_INDICES = []

for i in range((train.shape)[1]):
    if (i >= 2 and i <= 113) or (i >= 3205 and i <= 3212):
        SPARCE_INDICES.append(i)
    else:
        CONTINUOUS_INDICES.append(i)
train = train[:, CONTINUOUS_INDICES]
test = test[:, CONTINUOUS_INDICES]

print train.shape
#train = train[:100]
#y = y[:100]

def xgbTrain(flod = 5):
    """
    """
    denom = 0
    fold = 5 #Change to 5, 1 for Kaggle Limits
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
        x1, x2, y1, y2 = model_selection.train_test_split(train, y, test_size=0.18, random_state=i)
        watchlist = [(xgb.DMatrix(x1, y1), 'train'), (xgb.DMatrix(x2, y2), 'valid')]
        model = xgb.train(params, xgb.DMatrix(x1, y1), 1000,  watchlist, verbose_eval=50, early_stopping_rounds=100)
        score1 = metrics.log_loss(y2, model.predict(xgb.DMatrix(x2), ntree_limit=model.best_ntree_limit), labels = list(range(9)))
        print(score1)

        models.append((model, 'x'))

    return models


def lgbm_train(fold = 5):
    """
    LGB Training
    """
    train_len = len(train)
    print("Over all training size:")
    print train_len
    train_data = train#[:train_len * 3 / 10]
    train_label = y#[:train_len * 3 / 10]
    valide_data = train[train_len * 9 / 10:]
    valide_label = y[train_len * 9 / 10:]

    models = []
    for i in range(fold):
        d_train = lgb.Dataset(train_data, train_label)
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
            'num_leaves': 60,
          #  'min_sum_hessian_in_leaf': 20,
            'max_depth': 10,
            'learning_rate': 0.02,
            'feature_fraction': 0.5,
            'verbose': 0,
          #   'valid_sets': [d_valide],
            'num_boost_round': 327,
            'feature_fraction_seed': i,
            # 'bagging_fraction': 0.9,
            # 'bagging_freq': 15,
            # 'bagging_seed': i,
            # 'early_stopping_round': 10
            # 'random_state': 10
            # 'verbose_eval': 20
            #'min_data_in_leaf': 665
        }

        # ROUNDS = 1
        print 'fold: %d th light GBM train :-)' % (i)
        # params['feature_fraction_seed'] = i
        bst = lgb.train(
                        params ,
                        d_train,
                        verbose_eval = False
                        # valid_sets = [d_valide]
                        #num_boost_round = 1
                        )
        #cv_result = lgb.cv(params, d_train, nfold=10)
        #pd.DataFrame(cv_result).to_csv('cv_result', index = False)
        #exit(0)
        # pred = model_eval(bst, 'l', test)
        #print pred.shape
        #print pred[0, :]
        models.append((bst, 'l'))

    return models


def create_model(input_len):
    model = Sequential()
    model.add(Dense(HIDDEN_UNITS[0], activation='relu', input_dim = input_len))
    if DNN_BN:
        model.add(BatchNormalization())
    if DROPOUT_RATE > 0:
        model.add(Dropout(DROPOUT_RATE))
    model.add(Dense(HIDDEN_UNITS[1], activation='relu'))
    if DNN_BN:
        model.add(BatchNormalization())
    if DROPOUT_RATE > 0:
        model.add(Dropout(DROPOUT_RATE))
    # model.add(Dropout(0.1))
    #model.add(Dense(32, activation='relu'))
    #model.add(Dropout(0.2))
    model.add(Dense(9, activation='softmax'))

    # optimizer = SGD(lr=1e-3, decay=1e-6, momentum=0.9, nesterov=True)
    optimizer = RMSprop(lr=1e-3, rho = 0.9, epsilon = 1e-8)
    model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics = ['accuracy'])

    return model


def create_embedding_model(CONTINUOUS_COLUMNS = 100):
    """
    """
   # aisle_id = Input(shape=(1,))
   # aisle_embedding = Embedding(135, 6, input_length = 1)(aisle_id)
   # aisle_embedding = Reshape((6,))(aisle_embedding)

    print "model input size: %d" % CONTINUOUS_COLUMNS
    dense_input = Input(shape=(CONTINUOUS_COLUMNS,))
    merge_input = dense_input #concatenate([dense_input, aisle_embedding, department_embedding], axis = 1)

    merge_len = CONTINUOUS_COLUMNS# + 6 + 4
    output = create_model(merge_len)(merge_input)

    model = Model([dense_input], output)
    optimizer = RMSprop(lr=1e-3, rho = 0.9, epsilon = 1e-8)
    # optimizer = SGD(lr=1e-2, decay=1e-6, momentum=0.9, nesterov=True)
    model.compile(optimizer = Adam(),
            loss='categorical_crossentropy', metrics = ['accuracy'])

    return model


def create_siamese_net(input_size):
    """
    """
    input_shape = (input_size, )
    left_input = Input(input_shape)
    right_input = Input(input_shape)
    #build model to use in each siamese 'leg'
    model = Sequential()
    model.add(Dense(HIDDEN_UNITS[0], activation='sigmoid', input_dim = input_size))
    if DNN_BN:
        model.add(BatchNormalization())
    if DROPOUT_RATE > 0:
        model.add(Dropout(DROPOUT_RATE))
    model.add(Dense(HIDDEN_UNITS[1], activation='sigmoid'))
    if DNN_BN:
        model.add(BatchNormalization())
    if DROPOUT_RATE > 0:
        model.add(Dropout(DROPOUT_RATE))
    #encode each of the two inputs into a vector with the convnet
    encoded_l = model(left_input)
    encoded_r = model(right_input)
    #merge two encoded inputs with the l1 distance between them
    L1_distance = lambda x: K.abs(x[0]-x[1])
    both = merge([encoded_l,encoded_r], mode = L1_distance, output_shape=lambda x: x[0])
    prediction = Dense(1,activation='sigmoid')(both)
    siamese_net = Model(input=[left_input,right_input],output=prediction)
    #optimizer = SGD(0.0004,momentum=0.6,nesterov=True,decay=0.0003)

    optimizer = Adam()
    #//TODO: get layerwise learning rates and momentum annealing scheme described in paperworking
    siamese_net.compile(loss="binary_crossentropy",optimizer=optimizer)

    siamese_net.count_params()
    return siamese_net

class Siamese_Loader:
    #For loading batches and testing tasks to a siamese net
    def __init__(self,Xtrain,Xval = None):
        self.Xval = Xval
        self.Xtrain = Xtrain
        self.n_classes, self.n_examples,self.feature_size = Xtrain.shape
        # self.n_val,self.n_ex_val,_,_ = Xval.shape

    def get_batch(self,n):
        #Create batch of pairs, half same class, half different class
        categories = rng.choice(self.n_classes,size=(n,),replace=False)
        pairs=[np.zeros((n, self.feature_size)) for i in range(2)]
        targets=np.zeros((n,))
        targets[n//2:] = 1
        for i in range(n):
            category = categories[i]
            idx_1 = rng.randint(0,self.n_examples)
            pairs[0][i,:] = self.Xtrain[category,idx_1] #.reshape(self.feature_size)
            idx_2 = rng.randint(0,self.n_examples)
            #pick images of same class for 1st half, different for 2nd
            category_2 = category if i >= n//2 else (category + rng.randint(1,self.n_classes)) % self.n_classes
            pairs[1][i,:] = self.Xtrain[category_2,idx_2] #.reshape(self.w,self.h,1)
        return pairs, targets

    def make_oneshot_task(self,N):
        #Create pairs of test image, support set for testing N way one-shot learning.
        categories = rng.choice(self.n_val,size=(N,),replace=False)
        indices = rng.randint(0,self.n_ex_val,size=(N,))
        true_category = categories[0]
        ex1, ex2 = rng.choice(self.n_examples,replace=False,size=(2,))
        test_image = np.asarray([self.Xval[true_category,ex1,:,:]]*N).reshape(N,self.w,self.h,1)
        support_set = self.Xval[categories,indices,:,:]
        support_set[0,:,:] = self.Xval[true_category,ex2]
        support_set = support_set.reshape(N,self.w,self.h,1)
        pairs = [test_image,support_set]
        targets = np.zeros((N,))
        targets[0] = 1
        return pairs, targets

    def test_oneshot(self,model,N,k,verbose=0):
        #Test average N way oneshot learning accuracy of a siamese neural net over k one-shot tasks
        n_correct = 0
        for i in range(k):
            inputs, targets = self.make_oneshot_task(N)
            probs = model.predict(inputs)
            if np.argmax(probs) == 0:
                n_correct+=1
        percent_correct = (100.0*n_correct / k)
        return percent_correct


def siamese_train():
    """
    """
    train_data = [[] for i in range(9)]
    i = 0
    for feature in train:
        train_data[y[i]].append(feature)
        # np.insert(train_data[y[i]], 0, feature, axis = 0)
        i += 1
    for i in range(9):
    #    train_data[i] = np.array(train_data[i])
        print len(train_data[i])
    print 'i = %d' % i
    train_data = np.array([np.array(xi) for xi in train_data])
    for i in range(9):
    #    train_data[i] = np.array(train_data[i])
        print type(train_data[i])
    print "train data shape before gen pair"
    print train_data.shape
    siamese_data_loader = Siamese_Loader(train_data)
    pairs, targets = siamese_data_loader.get_batch(100000)
    return pairs, targets


def keras_train(nfolds = 10):
    """
    Detect Fish or noFish
    """
    print "Start gen training data, shuffle and normalize!"

    #train_data = train
    #train_target = np_utils.to_categorical(y)

    train_data, train_target = siamese_train()
    kf = KFold(len(train_target), n_folds=nfolds, shuffle=True)

    num_fold = 0
    models = []
    for train_index, test_index in kf:
        # model = create_model(classes = 2)
        # model = create_embedding_model((train_data.shape)[1])
        model = create_siamese_net((train.shape)[1])

        X_train = train_data[train_index]
        Y_train = train_target[train_index]
        X_valid = train_data[test_index]
        Y_valid = train_target[test_index]

        num_fold += 1
        print('Start KFold number {} from {}'.format(num_fold, nfolds))
        print('Split train: ', len(X_train), len(Y_train))
        print('Split valid: ', len(X_valid), len(Y_valid))

        callbacks = [
            EarlyStopping(monitor='val_loss', patience=3, verbose=0),
        ]
        model.fit([X_train], Y_train, batch_size=BATCH_SIZE, epochs=DNN_EPOCHS,
                shuffle=True, verbose=2, validation_data=([X_valid], Y_valid)
                , callbacks=callbacks)

        models.append((model, 'k'))

    return models


def model_eval(model, model_type, data_frame):
    """
    """
    if model_type == 'l':
        preds = model.predict(data_frame)
    elif model_type == 'k':
        preds = model.predict(data_frame, batch_size=BATCH_SIZE, verbose=2)
    elif model_type == 't':
        print "ToDO"
    elif model_type == 'x':
        preds = model.predict(xgb.DMatrix(data_frame), ntree_limit=model.best_ntree_limit+80)

    return preds


def gen_sub(models):
    """
    Evaluate single Type model
    """
    print('Start generate submission!')
    preds = None
    for (model, model_type) in models:
        pred = model_eval(model, model_type, test)
        #print pred.shape
        #print pred[0, :]
        if preds is None:
            preds = pred.copy()
        else:
            preds += pred

    preds /= len(models)
    submission = pd.DataFrame(preds, columns=['class'+str(c+1) for c in range(9)])
    submission['ID'] = pid
    sub_name = "submission" + strftime('_%Y_%m_%d_%H_%M_%S', gmtime()) + ".csv"
    print('Output to ' + sub_name)
    submission.to_csv(sub_name, index=False)

if __name__ == "__main__":
    model_k = keras_train(10)
    # gen_sub(model_k, 'k', th, F1)
    # xgbTrain();
    #model_l = lgbm_train(10)#model_k)
    #model_x = xgbTrain(5)#model_k)
    # gen_sub(model_k) #model_k)
