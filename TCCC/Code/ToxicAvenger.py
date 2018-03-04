import numpy as np
import pandas as pd
import sklearn
from sklearn import feature_extraction, ensemble, decomposition, pipeline
from textblob import TextBlob
from nfold_train import nfold_train, models_eval
from time import gmtime, strftime
from RCNN_Keras import get_word2vec
from keras.preprocessing.text import Tokenizer, text_to_word_sequence
from keras.preprocessing.sequence import pad_sequences

zpolarity = {0:'zero',1:'one',2:'two',3:'three',4:'four',5:'five',6:'six',7:'seven',8:'eight',9:'nine',10:'ten'}
zsign = {-1:'negative',  0.: 'neutral', 1:'positive'}

data_dir = '../Data/'
train = pd.read_csv(data_dir + '/train.csv')
test = pd.read_csv(data_dir + '/test.csv')
# sub1 = pd.read_csv(data_dir + '/submission_ensemble.csv')
nrow = train.shape[0]
print("Train Size: {0}".format(nrow))
print("Test Size: {0}".format(test.shape[0]))

coly = [c for c in train.columns if c not in ['id','comment_text']]
print("Label columns: {0}".format(coly))
y = train[coly]
tid = test['id'].values

# train['polarity'] = train['comment_text'].map(lambda x: int(TextBlob(x).sentiment.polarity * 10))
# test['polarity'] = test['comment_text'].map(lambda x: int(TextBlob(x).sentiment.polarity * 10))

# train['comment_text'] = train.apply(lambda r: str(r['comment_text']) + ' polarity' +  zsign[np.sign(r['polarity'])] + zpolarity[np.abs(r['polarity'])], axis=1)
# test['comment_text'] = test.apply(lambda r: str(r['comment_text']) + ' polarity' +  zsign[np.sign(r['polarity'])] + zpolarity[np.abs(r['polarity'])], axis=1)

df = pd.concat([train['comment_text'], test['comment_text']], axis=0)
df = df.fillna("unknown")

print('Word2Vec...')
# get_word2vec(df)
# exit(0)

# print('Pipeline...')
# fp = pipeline.Pipeline([
#    ('union', pipeline.FeatureUnion(
#        n_jobs = -1,
#        transformer_list = [
#            ('pi1', pipeline.Pipeline([('count_comment_text', \
#                 feature_extraction.text.TfidfVectorizer(stop_words='english', analyzer=u'char', ngram_range=(2, 8), max_features=50000)), \
#                 ('tsvd1', decomposition.TruncatedSVD(n_components=128, n_iter=25, random_state=12))])
#                 ),
#            ('pi2', pipeline.Pipeline([('tfidf_Text', \
#                 feature_extraction.text.TfidfVectorizer(stop_words='english', ngram_range=(1, 3), max_features=50000)), \
#                 ('tsvd2', decomposition.TruncatedSVD(n_components=128, n_iter=25, random_state=12))])
#                 )
#        ])
#    )])

# data = fp.fit_transform(df)
# svd_name = "svd" + strftime('_%Y_%m_%d_%H_%M_%S', gmtime()) + ".npy"
# np.save(svd_name, data)
# data = np.load('svd_2018_03_01_16_33_00.npy')
data = df.values

# Text to sequence
print('Tokenizer...')
tokenizer = Tokenizer(num_words = 50000)
tokenizer.fit_on_texts(data)
data = tokenizer.texts_to_sequences(data)
data = pad_sequences(data, maxlen = 100)
svd_name = "token_sequence" + strftime('_%Y_%m_%d_%H_%M_%S', gmtime()) + ".npy"
np.save(svd_name, data)
# data = np.load('token_sequence_2018_03_04_15_50_25.npy')

train_data, train_label = data[:nrow], y
test_data = data[nrow:]

print("Training------")
multi_label_models = []
sub2 = pd.DataFrame(np.zeros((test.shape[0], len(coly))), columns = coly)
models, _, _, _ = nfold_train(train_data, train_label.values, fold = 10, model_types = ['rnn'])
# exit(0)
# for c in coly:
#     print("------Label: {0}".format(c))
#     label = train_label[c].values
#     models, _, _, _ = nfold_train(train_data, label, fold = 5, model_types = ['k']) #, valide_label = train_label)
#     multi_label_models.append(models)
#     sub2[c] = models_eval(models, test_data)
#model = ensemble.ExtraTreesClassifier(n_jobs=-1, random_state=3)
#model.fit(data[:nrow], y[:nrow])
# print(1- model.score(data[:nrow], y[:nrow]))
sub2[coly] = models_eval(models, test_data)
# sub2 = pd.DataFrame([[c[1] for c in sub2[row]] for row in range(len(sub2))]).T
# sub2.columns = coly
sub2['id'] = tid
for c in coly:
    sub2[c] = sub2[c].clip(0+1e12, 1-1e12)

''' #blend 1
sub2.columns = [x+'_' if x not in ['id'] else x for x in sub2.columns]
blend = pd.merge(sub1, sub2, how='left', on='id')
for c in coly:
    blend[c] = blend[c] * 0.8 + blend[c+'_'] * 0.2
    blend[c] = blend[c].clip(0+1e12, 1-1e12)
blend = blend[sub1.columns]

#blend 2
sub2 = blend[:]
sub2.columns = [x+'_' if x not in ['id'] else x for x in sub2.columns]
blend = pd.merge(sub1, sub2, how='left', on='id')
for c in coly:
    blend[c] = np.sqrt(blend[c] * blend[c+'_'])
    blend[c] = blend[c].clip(0+1e12, 1-1e12) '''
blend = sub2 #blend[sub2.columns]
sub_name = "sub" + strftime('_%Y_%m_%d_%H_%M_%S', gmtime()) + ".csv"
blend.to_csv(sub_name, index=False)
