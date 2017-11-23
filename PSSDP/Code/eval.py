import tensorflow as tf
import keras.backend as K
import keras
import numpy as np
from sklearn.metrics import roc_auc_score
from optimize_auc import rank_score
from keras.models import Sequential, Model
from scipy.special import expit as sigmoid

def gini(actual, pred, cmpcol = 0, sortcol = 1):
    assert( len(actual) == len(pred) )
    all = np.asarray(np.c_[ actual, pred, np.arange(len(actual)) ], dtype=np.float)
    all = all[ np.lexsort((all[:,2], -1*all[:,1])) ]
    totalLosses = all[:,0].sum()
    giniSum = all[:,0].cumsum().sum() / totalLosses

    giniSum -= (len(actual) + 1) / 2.
    return giniSum / len(actual)

def gini_normalized(a, p):
    return gini(a, p) / gini(a, a)

def rank_statistic_loss(y_true, y_pred):
    true_num = np.sum(y_true == 1)
    minor_class = 0
    major_class = 1
    if true_num * 2 < y_true.shape[0]:
        minor_class = 1
        major_class = 0

def min_pred(y_true, y_pred):
    return K.mean(y_pred, axis = -1)


def eval_gini(y_true, y_prob):
    print(type(y_true))
    print(type(y_prob))
    sess = tf.Session()
    with sess.as_default():
        y_true = y_true[np.argsort(y_prob.eval())]
    ntrue = 0
    gini = 0
    delta = 0
    n = len(y_true)
    for i in range(n-1, -1, -1):
        y_i = y_true[i]
        ntrue += y_i
        gini += y_i * delta
        delta += 1 - y_i
    gini = 1 - 2 * gini / (ntrue * (n - ntrue))
    return gini


def eval_auc(y_true, y_prob):
    score, up_opt = tf.metrics.auc(y_true, y_prob)
    K.get_session().run(tf.local_variables_initializer())
    with tf.control_dependencies([up_opt]):
        score = tf.identity(score)
    return score

class GiniWithEarlyStopping(keras.callbacks.Callback):
    def __init__(self, min_delta=0, patience=0, verbose=0, predict_batch_size=1024):
        #print("self vars: ",vars(self))  #uncomment and discover some things =)

        # FROM EARLY STOP
        super(GiniWithEarlyStopping, self).__init__()
        self.patience = patience
        self.verbose = verbose
        self.min_delta = min_delta
        self.wait = 0
        self.stopped_epoch = 0
        self.monitor_op = np.greater
        self.predict_batch_size=predict_batch_size

    def on_batch_begin(self, batch, logs={}):
        if(self.verbose > 2):
            if(batch!=0):
                print("")
            print("Hi! on_batch_begin() , batch=",batch,",logs:",logs)
            #print("self vars: ",vars(self))  #uncomment and discover some things =)

    def on_batch_end(self, batch, logs={}):
        if(self.verbose > 2):
            print("Hi! on_batch_end() , batch=",batch,",logs:",logs)
            #print("self vars: ",vars(self))  #uncomment and discover some things =)

    def on_train_begin(self, logs={}):
        if(self.verbose > 1):
            print("Hi! on_train_begin() ,logs:",logs)
            #print("self vars: ",vars(self))  #uncomment and discover some things =)

        # FROM EARLY STOP
        # Allow instances to be re-used
        self.wait = 0
        self.stopped_epoch = 0
        self.best = -np.Inf

    def on_train_end(self, logs={}):
        if(self.verbose > 1):
            print("Hi! on_train_end() ,logs:",logs)
            #print("self vars: ",vars(self))  #uncomment and discover some things =)

        # FROM EARLY STOP
        if self.stopped_epoch > 0 and self.verbose > 0:
            print('Epoch ',self.stopped_epoch,': GiniEarlyStopping')

    def on_epoch_begin(self, epoch, logs={}):
        if(self.verbose > 1):
            print("Hi! on_epoch_begin() , epoch=",epoch,",logs:",logs)
            #print("self vars: ",vars(self))  #uncomment and discover some things =)

    def on_epoch_end(self, epoch, logs={}):
        valide_data = self.validation_data[:-3]
        valide_label = self.validation_data[-3]
        if(self.validation_data):
            y_hat_val=self.model.predict(valide_data,batch_size=self.predict_batch_size)

        if(self.verbose > 1):
            print("Hi! on_epoch_end() , epoch=",epoch,",logs:",logs)
            #print("self vars: ",vars(self))  #uncomment and discover some things =)

        #i didn't found train data to check gini on train set (@TODO HERE)
        # from source code of Keras: https://github.com/fchollet/keras/blob/master/keras/engine/training.py#L1127
        # for cbk in callbacks:
        #     cbk.validation_data = val_ins
        # Probably we will need to change keras...
        #

            print("    GINI Callback:")
            if(self.validation_data):
                print('        validation_data.inputs       : ',np.shape(self.validation_data[0]))
                print('        validation_data.targets      : ',np.shape(valide_label))
                print("        roc_auc_score(y_real,y_hat)  : ",roc_auc_score(valide_label, y_hat_val ))
                print("        gini_normalized(y_real,y_hat): ",gini_normalized(valide_label, y_hat_val))
                print("        roc_auc_scores*2-1           : ",roc_auc_score(valide_label, y_hat_val)*2-1)

            print('    Logs (others metrics):',logs)
        # FROM EARLY STOP
        if(self.validation_data):
            if (self.verbose == 1):
                print("\n GINI Callback:",gini_normalized(valide_label, y_hat_val))
            current = gini_normalized(valide_label, y_hat_val)
            #roc_auc = roc_auc_score(valide_label, y_hat_val)
            #roc_auc_2_1 = 2 * roc_auc - 1
            # we can include an "gambiarra" (very usefull brazilian portuguese word)
            # to logs (scores) and use others callbacks too....
            #logs['gini_val']=current
            #logs['roc_auc_val']=roc_auc
            #logs['roc_auc_2_1_val']=roc_auc_2_1

            if self.monitor_op(current - self.min_delta, self.best):
                self.best = current
                self.wait = 0
            else:
                self.wait += 1
                if self.wait >= self.patience:
                    self.stopped_epoch = epoch
                    self.model.stop_training = True


class PairAUCEarlyStopping(keras.callbacks.Callback):
    def __init__(self, min_delta=0, patience=0, verbose=0, predict_batch_size=1024):
        #print("self vars: ",vars(self))  #uncomment and discover some things =)

        # FROM EARLY STOP
        super(PairAUCEarlyStopping, self).__init__()
        self.patience = patience
        self.verbose = verbose
        self.min_delta = min_delta
        self.wait = 0
        self.stopped_epoch = 0
        self.monitor_op = np.greater
        self.predict_batch_size=predict_batch_size

    def on_batch_begin(self, batch, logs={}):
        if(self.verbose > 2):
            if(batch!=0):
                print("")
            print("Hi! on_batch_begin() , batch=",batch,",logs:",logs)
            #print("self vars: ",vars(self))  #uncomment and discover some things =)

    def on_batch_end(self, batch, logs={}):
        if(self.verbose > 2):
            print("Hi! on_batch_end() , batch=",batch,",logs:",logs)
            #print("self vars: ",vars(self))  #uncomment and discover some things =)

    def on_train_begin(self, logs={}):
        if(self.verbose > 1):
            print("Hi! on_train_begin() ,logs:",logs)
            #print("self vars: ",vars(self))  #uncomment and discover some things =)

        # FROM EARLY STOP
        # Allow instances to be re-used
        self.wait = 0
        self.stopped_epoch = 0
        self.best = -np.Inf

    def on_train_end(self, logs={}):
        if(self.verbose > 1):
            print("Hi! on_train_end() ,logs:",logs)
            #print("self vars: ",vars(self))  #uncomment and discover some things =)

        # FROM EARLY STOP
        if self.stopped_epoch > 0 and self.verbose > 0:
            print('Epoch ',self.stopped_epoch,': GiniEarlyStopping')

    def on_epoch_begin(self, epoch, logs={}):
        if(self.verbose > 1):
            print("Hi! on_epoch_begin() , epoch=",epoch,",logs:",logs)
            #print("self vars: ",vars(self))  #uncomment and discover some things =)

    def on_epoch_end(self, epoch, logs={}):
        valide_data = self.validation_data[:-3]
        valide_label = self.validation_data[-3]
        if(self.validation_data):
            model = Model(inputs = self.model.input, outputs = self.model.get_layer('rank_scale_layer').output)
            y_hat_val=model.predict(valide_data,batch_size=self.predict_batch_size)

        if(self.verbose > 1):
            print("Hi! on_epoch_end() , epoch=",epoch,",logs:",logs)
            #print("self vars: ",vars(self))  #uncomment and discover some things =)

        #i didn't found train data to check gini on train set (@TODO HERE)
        # from source code of Keras: https://github.com/fchollet/keras/blob/master/keras/engine/training.py#L1127
        # for cbk in callbacks:
        #     cbk.validation_data = val_ins
        # Probably we will need to change keras...
        #

            print("    GINI Callback:")
            if(self.validation_data):
                print('        validation_data.inputs       : ',np.shape(self.validation_data[0]))
                print('        validation_data.targets      : ',np.shape(valide_label))
                print("        roc_auc_score(y_real,y_hat)  : ",roc_auc_score(valide_label, y_hat_val ))
                print("        gini_normalized(y_real,y_hat): ",gini_normalized(valide_label, y_hat_val))
                print("        roc_auc_scores*2-1           : ",roc_auc_score(valide_label, y_hat_val)*2-1)

            print('    Logs (others metrics):',logs)
        current = np.mean(np.heaviside(y_hat_val, 0.5))
        # FROM EARLY STOP
        if(self.validation_data):
            if (self.verbose == 1):
                rank_auc = current
                gini = 2 * rank_auc - 1
                current_sigmoid = np.mean(sigmoid(y_hat_val))
                gini_sigmoid = 2 * current_sigmoid - 1
                print("Validate Heaviside Rank Score: {}, rank_auc: {}, gini: {}".format(current, rank_auc, gini))
                print("Validate Sigmoid Rank Score: {}, rank_auc: {}, gini: {}".format(current_sigmoid, current_sigmoid, gini_sigmoid))

            if self.monitor_op(current - self.min_delta, self.best):
                self.best = current
                self.wait = 0
            else:
                self.wait += 1
                if self.wait >= self.patience:
                    self.stopped_epoch = epoch
                    self.model.stop_training = True
