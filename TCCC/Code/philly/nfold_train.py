from sklearn.model_selection import KFold
# from lgb import lgbm_train
# import xgboost as xgb
from functools import reduce
import numpy as np
# from keras_train import keras_train
# import gensim
# from RCNN_Keras import get_word2vec, RCNN_Model
# from RNN_Keras import RNN_Model
from CNN_Keras import CNN_Model, get_word2vec_embedding
from vdcnn import VDCNN_Model

# RNN_PARAMS
RCNN_HIDDEN_UNIT = [128, 64]


def nfold_train(train_data, train_label, model_types = None,
            stacking = False, valide_data = None, valide_label = None,
            test_data = None, train_weight = None, valide_weight = None, 
            flags = None ,tokenizer = None):
    """
    nfold Training
    """
    print("Over all training size:")
    print(train_data.shape)

    kf = KFold(n_splits=flags.nfold, shuffle=False)
    # wv_model = gensim.models.Word2Vec.load("wv_model_norm.gensim")

    stacking_data = None
    stacking_label = None
    test_preds = None
    num_fold = 0
    models = []
    embedding_weight = None 
    if flags.load_wv_model:
        embedding_weight = get_word2vec_embedding(location = flags.input_training_data_path + '/wiki.en.vec.indata', \
                 tokenizer = tokenizer, nb_words = flags.vocab_size, embed_size = flags.emb_dim, \
                 model_type = flags.wv_model_type)
    for train_index, test_index in kf.split(train_data):
        print('fold: %d th train :-)' % (num_fold))
        print('Train size: {} Valide size: {}'.format(train_index.shape[0], test_index.shape[0]))
        if valide_label is None:
            train_part = train_data[train_index]
            train_part_label = train_label[train_index]
            valide_part = train_data[test_index]
            valide_part_label = train_label[test_index]
            if train_weight is not None:
                train_part_weight = train_weight[train_index]
                valide_part_weight = train_weight[test_index]
        else:
            train_part = train_data
            train_part_label = train_label
            valide_part = valide_data
            valide_part_label = valide_label
            if train_weight is not None:
                train_part_weight, valide_part_weight = train_weight, valide_weight
        onefold_models = []
        for model_type in model_types:
            if model_type == 'k':
                pass
                # with tf.device('/cpu:0'):
                model = keras_train(train_part, train_part_label, valide_part, valide_part_label, num_fold)
                onefold_models.append((model, 'k'))
            elif model_type == 'x':
                pass
                # model = xgb_train(train_part, train_part_label, valide_part, valide_part_label, num_fold)
                # onefold_models.append((model, 'x'))
            elif model_type == 'l':
                model = lgbm_train(train_part, train_part_label, valide_part, valide_part_label, num_fold,
                        fold)
                onefold_models.append((model, 'l'))
            elif model_type == 'rcnn':
                # model = Create_RCNN(MAX_NUM_WORDS, RNN_EMBEDDING_DIM, 2, LSTM_UNIT, RCNN_HIDDEN_UNIT, wv_model)
                model = RCNN_Model(wv_model_file = 'wv_model_norm.gensim', num_classes = 2, context_vector_dim = LSTM_UNIT, \
                        hidden_dim = RCNN_HIDDEN_UNIT, max_len = MAX_SEQUENCE_LEN)
                model.train(train_part, train_part_label, valide_part, valide_part_label)
                print(model.model.summary())
                onefold_models.append((model, 'rcnn'))
            elif model_type == 'rnn':
                model = RNN_Model(max_token = MAX_NUM_WORDS, num_classes = 2, context_vector_dim = LSTM_UNIT, \
                        hidden_dim = RCNN_HIDDEN_UNIT, max_len = MAX_SEQUENCE_LEN, embedding_dim = RNN_EMBEDDING_DIM)
                model.train(train_part, train_part_label, valide_part, valide_part_label)
                # print(model.model.summary())
                onefold_models.append((model, 'rnn'))
            elif model_type == 'cnn':
                model = CNN_Model(max_token = flags.vocab_size, num_classes = 2, context_vector_dim = flags.rnn_unit, \
                        hidden_dim = [int(hn.strip()) for hn in flags.full_connect_hn.strip().split(',')], \
                        max_len = flags.max_seq_len, embedding_dim = flags.emb_dim, tokenizer = tokenizer, \
                        embedding_weight = embedding_weight, batch_size = flags.batch_size, epochs = flags.epochs, \
                        filter_size = flags.filter_size, fix_wv_model = flags.fix_wv_model, \
                        batch_interval = flags.batch_interval, emb_dropout = flags.emb_dropout, \
                        full_connect_dropout = flags.full_connect_dropout)
                if num_fold == 0:
                    print(model.model.summary())
                model.train(train_part, train_part_label, valide_part, valide_part_label)
                onefold_models.append((model, 'cnn'))
            elif model_type == 'vdcnn':
                model = VDCNN_Model(num_filters = [int(hn.strip()) for hn in flags.vdcnn_filters.strip().split(',')], \
                        sequence_max_length = flags.max_seq_len, top_k = flags.vdcc_top_k, embedding_size = flags.emb_dim, \
                        hidden_dim = [int(hn.strip()) for hn in flags.full_connect_hn.strip().split(',')], \
                        batch_size = flags.batch_size, dense_dropout = flags.full_connect_dropout, epochs = flags.epochs)
                if num_fold == 0:
                    print(model.model.summary())
                model.train(train_part, train_part_label, valide_part, valide_part_label)
                onefold_models.append((model, 'cnn'))
        if stacking:
            valide_pred = [model_eval(model[0], model[1], valide_part) for model in onefold_models]
            valide_pred = reduce((lambda x, y: np.c_[x, y]), valide_pred)
            test_pred = [model_eval(model[0], model[1], test_data) for model in onefold_models]
            test_pred = reduce((lambda x, y: np.c_[x, y]), test_pred)
            if stacking_data is None:
                stacking_data = np.c_[valide_part, valide_pred]
                stacking_label = valide_part_label
                test_preds = test_pred
            else:
                stacking_data = np.append(stacking_data, np.c_[valide_part, valide_pred], axis = 0)
                stacking_label = np.append(stacking_label, valide_part_label, axis = 0)
                test_preds += test_pred
            print('stacking_data shape: {0}'.format(stacking_data.shape))
            print('stacking_label shape: {0}'.format(stacking_label.shape))
        models.append(onefold_models[0])
        num_fold += 1
        if num_fold == flags.ensemble_nfold:
            break
    if stacking:
        test_preds /= fold
        test_data = np.c_[test_data, test_preds]
    return models, stacking_data, stacking_label, test_data


def model_eval(model, model_type, data_frame):
    """
    """
    if model_type == 'l':
        preds = model.predict(data_frame)
    elif model_type == 'k' or model_type == 'LR' or model_type == 'DNN' or model_type == 'rcnn' \
        or model_type == 'rnn' or model_type == 'cnn':
        preds = model.predict(data_frame, verbose = 2)
    elif model_type == 't':
        print("ToDO")
    elif model_type == 'x':
        preds = model.predict(xgb.DMatrix(data_frame), ntree_limit=model.best_ntree_limit)
    return preds #.reshape((data_frame.shape[0], -1))

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