import numpy as np
import pandas as pd
import sklearn
import tensorflow as tf
from sklearn import feature_extraction, ensemble, decomposition, pipeline
# from textblob import TextBlob
from nfold_train import nfold_train, models_eval
import time
from time import gmtime, strftime

from tensorflow.python.keras.preprocessing.text import Tokenizer, text_to_word_sequence
from tensorflow.python.keras.preprocessing.sequence import pad_sequences
from data_helper import data_helper
import shutil
import os
from contextlib import contextmanager
# from nltk.stem import PorterStemmer
# ps = PorterStemmer()
import gensim

zpolarity = {0:'zero',1:'one',2:'two',3:'three',4:'four',5:'five',6:'six',7:'seven',8:'eight',9:'nine',10:'ten'}
zsign = {-1:'negative',  0.: 'neutral', 1:'positive'}

flags = tf.app.flags
flags.DEFINE_string('input-training-data-path', "../../Data/", 'data dir override')
flags.DEFINE_string('output-model-path', ".", 'model dir override')
flags.DEFINE_string('model_type', "cnn", 'model type')
flags.DEFINE_integer('vocab_size', 300000, 'vocab size')
flags.DEFINE_integer('max_seq_len', 100, 'max sequence length')
flags.DEFINE_integer('nfold', 10, 'number of folds')
flags.DEFINE_integer('ensemble_nfold', 5, 'number of ensemble models')
flags.DEFINE_integer('emb_dim', 300, 'term embedding dim')
flags.DEFINE_string('rnn_unit', 0, 'RNN Units')
flags.DEFINE_integer('epochs', 1, 'number of Epochs')
flags.DEFINE_integer('batch_size', 128, 'Batch size')
flags.DEFINE_bool("load_wv_model", True, "Whether to load word2vec model")
flags.DEFINE_string('wv_model_type', "fast_text", 'word2vec model type')
flags.DEFINE_string('wv_model_file', "wiki.en.vec.indata", 'word2vec model file')
flags.DEFINE_bool("char_split", False, "Whether to split text into character")
flags.DEFINE_string('filter_size', 100, 'CNN filter size')
flags.DEFINE_bool("fix_wv_model", True, "Whether to fix word2vec model")
flags.DEFINE_integer('batch_interval', 1000, 'batch print interval')
flags.DEFINE_float("emb_dropout", 0, "embedding dropout")
flags.DEFINE_string('full_connect_hn', "64, 32", 'full connect hidden units')
flags.DEFINE_float("full_connect_dropout", 0, "full connect drop out")
flags.DEFINE_string('vdcnn_filters', "64, 128, 256", 'vdcnn filters')
flags.DEFINE_integer('vdcc_top_k', 1, 'vdcc top_k')
flags.DEFINE_bool("separate_label_layer", False, "Whether to separate label layer")
flags.DEFINE_bool("stem", False, "Whether to stem")
flags.DEFINE_bool("resnet_hn", False, "Whether to concatenate hn and rcnn")
flags.DEFINE_integer('letter_num', 3, 'letter number to aggregate')
FLAGS = flags.FLAGS

train = pd.read_csv(FLAGS.input_training_data_path + '/train.csv')
test = pd.read_csv(FLAGS.input_training_data_path +  '/test.csv')
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
# print(type(df))
# df_len = df.map(lambda x: len(x))
# print ("min: {0} max: {1} median: {2} mean: {3} std: {4}".format(df_len.min(), df_len.max(), df_len.median(), df_len.mean(), df_len.std()))
# exit(0)
df = df.fillna("unknown")

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
@contextmanager
def timer(name):
    """
    Taken from Konstantin Lopuhin https://www.kaggle.com/lopuhin
    in script named : Mercari Golf: 0.3875 CV in 75 LOC, 1900 s
    https://www.kaggle.com/lopuhin/mercari-golf-0-3875-cv-in-75-loc-1900-s
    """
    t0 = time.time()
    yield
    print('[{0}] done in {1} s'.format(name, time.time() - t0))


with timer("Performing stemming"):
    if FLAGS.stem:
        # stem_sentence = lambda s: " ".join(ps.stem(word) for word in s.strip().split())
        data = [gensim.parsing.stem_text(comment) for comment in data]
print('Tokenizer...')
if not FLAGS.char_split:
    tokenizer = Tokenizer(num_words = FLAGS.vocab_size)
    tokenizer.fit_on_texts(data)
    data = tokenizer.texts_to_sequences(data)
    data = pad_sequences(data, maxlen = FLAGS.max_seq_len)
else:
    tokenizer = None
    data_helper = data_helper(sequence_max_length = FLAGS.max_seq_len, \
            wv_model_path = FLAGS.input_training_data_path + FLAGS.wv_model_file, \
            letter_num = FLAGS.letter_num, emb_dim = FLAGS.emb_dim, load_wv_model = FLAGS.load_wv_model)
    data, emb_weight, FLAGS.vocab_size = data_helper.text_to_triletter_sequence(data)
# print(data[:2])
# exit(0)
# svd_name = "token_sequence" + strftime('_%Y_%m_%d_%H_%M_%S', gmtime()) + ".npy"
# np.save(svd_name, data)
# data = np.load('token_sequence_2018_03_04_15_50_25.npy')

train_data, train_label = data[:nrow], y.values
test_data = data[nrow:]

print("Training------")
multi_label_models = []
scores_text = []
sub2 = pd.DataFrame(np.zeros((test.shape[0], len(coly))), columns = coly)
models, _, _, _ = nfold_train(train_data, train_label, flags = FLAGS, model_types = [FLAGS.model_type], \
            tokenizer = tokenizer, scores = scores_text, emb_weight = emb_weight) #, valide_data = train_data, valide_label = train_label)
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
time_label = strftime('_%Y_%m_%d_%H_%M_%S', gmtime())
sub_name = "sub" + time_label + ".csv"
blend.to_csv(sub_name, index=False)

scores_text_frame = pd.DataFrame(scores_text, columns = ["score_text"])
score_text_file = "score_text" + time_label + ".csv"
scores_text_frame.to_csv(score_text_file, index=False)
scores = scores_text_frame["score_text"]
for i in range(FLAGS.epochs):
    scores_epoch = scores.loc[scores.str.startswith('epoch:{0}'.format(i + 1))].map(lambda s: float(s.split()[1]))
    print ("Epoch{0} mean:{1} std:{2} min:{3} max:{4} median:{5}".format(i + 1, \
        scores_epoch.mean(), scores_epoch.std(), scores_epoch.min(), scores_epoch.max(), scores_epoch.median()))
# score_file = "score" + time_label + ".csv"
# scores.to_csv(score_file, index=False)

# Move to hdfs
if not os.path.isdir(FLAGS.output_model_path):
    os.makedirs(FLAGS.output_model_path, exist_ok=True)
shutil.move(sub_name, FLAGS.output_model_path)
shutil.move(score_text_file, FLAGS.output_model_path)