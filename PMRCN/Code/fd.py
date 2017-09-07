
from sklearn import *
import sklearn
import pandas as pd
import numpy as np
import xgboost as xgb
import lightgbm as lgb

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
#    for i in range(56):
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
#print train.dtypes
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

def xgbTrain(flod = 5):
    """
    """
    denom = 0
    fold = 5 #Change to 5, 1 for Kaggle Limits
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
        #if score < 0.9:
        if denom != 0:
            pred = model.predict(xgb.DMatrix(test), ntree_limit=model.best_ntree_limit+80)
            preds += pred
        else:
            pred = model.predict(xgb.DMatrix(test), ntree_limit=model.best_ntree_limit+80)
            preds = pred.copy()
        denom += 1
        submission = pd.DataFrame(pred, columns=['class'+str(c+1) for c in range(9)])
        submission['ID'] = pid
        submission.to_csv('submission_xgb_fold_'  + str(i) + '.csv', index=False)

    preds /= denom
    submission = pd.DataFrame(preds, columns=['class'+str(c+1) for c in range(9)])
    submission['ID'] = pid
    submission.to_csv('submission_xgb.csv', index=False)


def lgbm_train(model_k = None):
    """
    LGB Training
    """
    train_len = len(train)
    print("Over all training size:")
    print train_len
    train_data = train[:train_len * 9 / 10]
    train_label = y[:train_len * 9 / 10]
    valide_data = train[train_len * 9 / 10:]
    valide_label = y[train_len * 9 / 10:]
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
        'num_leaves': 6,
      #  'min_sum_hessian_in_leaf': 20,
        'max_depth': 4,
        'learning_rate': 0.4,
        'feature_fraction': 1,
        'verbose': 1,
      #   'valid_sets': [d_valide],
        'num_boost_round': 50
    }

    # ROUNDS = 1

    print('light GBM train :-)')
    bst = lgb.train(params , d_train, valid_sets = [d_valide])#, num_boost_round = 1)

    return bst


def model_eval(model, model_type, train_data_frame):
    """
    """
    if model_type == 'l':
        preds = model.predict(train_data_frame)
    elif model_type == 'k':
        norm_slope = model[1]
        norm_intercept = model[2]
        data = train_data_frame * norm_slope + norm_intercept
        preds = keras_eval(model[0], data.values)
    elif model_type == 't':
        print "ToDO"

    return preds


def gen_sub(models, model_type):
    """
    Evaluate single Type model
    """
    preds = model_eval(models, model_type, test)

    submission = pd.DataFrame(preds, columns=['class'+str(c+1) for c in range(9)])
    submission['ID'] = pid
    submission.to_csv('submission_lgb.csv', index=False)

if __name__ == "__main__":
    # model_k, th, F1 = keras_train(10)
    # gen_sub(model_k, 'k', th, F1)
    xgbTrain();
    # model_l = lgbm_train()#model_k)
    # gen_sub(model_l, 'l') #model_k)
