# Author : Ding Fu & Paul-Antoine Nguyen

# This script considers all the products a user has ordered
#
# We train a model computing the probability of reorder on the "train" data
#
# For the submission, we keep the orders that have a probability of
# reorder higher than a threshold


import tempfile
import numpy as np
import pandas as pd
import lightgbm as lgb
import tensorflow as tf
from tensorflow.contrib.layers.python import ops

from sklearn.cross_validation import KFold
from keras.models import Sequential, Model
from keras.layers.core import Dense, Dropout, Flatten, Reshape
from keras.layers.normalization import BatchNormalization
from keras.layers.embeddings import Embedding
from keras.layers import Input, concatenate
from keras.layers.convolutional import Convolution2D, MaxPooling2D, ZeroPadding2D, AveragePooling2D
from keras.optimizers import SGD, RMSprop
from keras.callbacks import EarlyStopping
from keras.utils import np_utils
from sklearn.metrics import log_loss
from keras import __version__ as keras_version

IDIR = '../Data/'
LABEL_COLUMN = "label"

batch_size = 50
dnn_epoch = 2
hidden_units = [16, 8]
dropout_rate = 0
lgbm_epoch = 350

DNN_BN = False
add_UserXProduct = True
add_CrossFeature = True
small_test = False
debug = False

CATEGORICAL_COLUMNS = ['aisle_id', 'department_id']# , 'product_id']

CONTINUOUS_COLUMNS = [
        'user_total_orders', 'user_total_items', 'total_distinct_items',
       'user_average_days_between_orders', 'user_average_basket',
       'order_hour_of_day', 
        'days_since_prior_order']
        #'product_orders', 'product_reorders', 'product_reorder_rate',
        #'aisle_orders', 'aisle_reorders', 'aisle_reorder_rate',
        #'department_orders', 'department_reorders', 'department_reorder_rate'] # 'days_since_ratio']

CROSS_KEY_NAME = [
           'userXaisle', 
           'userXdepartment', 
          # 'userXproduct',
          #  'productXorder_dow',
          #  'productXdays_since_prior_order',
          #  'productXorder_hour_of_day',
           'aisleXorder_dow', 'aisleXdays_since_prior_order', 'aisleXorder_hour_of_day',
          #  'departmentXorder_dow', 'departmentXdays_since_prior_order', 'departmentXorder_hour_of_day',
            'userXorder_dow',
            'userXdays_since_prior_order',
            'userXorder_hour_of_day',
           'user', 'aisle', 'department', 'product']
KEY_GEN_FUNC = [
            lambda d, k: d[k[0]] * 150 + d[k[1]], 
                lambda d, k: d[k[0]] * 25 + d[k[1]],
                lambda d, k: d[k[0]] * 65536 + d[k[1]],
             lambda d, k: d[k[0]].astype(np.uint32) * 10 + d[k[1]], 
                lambda d, k: d[k[0]].astype(np.uint32) * 35 + d[k[1]], 
                lambda d, k: d[k[0]].astype(np.uint32) * 25 + d[k[1]],
             lambda d, k: d[k[0]].astype(np.uint16) * 10 + d[k[1]], 
                lambda d, k: d[k[0]].astype(np.uint16) * 35 + d[k[1]], 
                lambda d, k: d[k[0]].astype(np.uint16) * 25 + d[k[1]],
             lambda d, k: d[k[0]] * 10 + d[k[1]], 
                lambda d, k: d[k[0]] * 35 + d[k[1]], 
                lambda d, k: d[k[0]] * 25 + d[k[1]],
             lambda d, k: d[k[0]]
                ]

CROSS_FEATURE_CONFIG = pd.DataFrame(
        {'cross_key': 
            ['userXaisle', 'userXdepartment', 'userXproduct',
            'productXorder_dow', 'productXdays_since_prior_order', 'productXorder_hour_of_day',
            'aisleXorder_dow', 'aisleXdays_since_prior_order', 'aisleXorder_hour_of_day',
            'departmentXorder_dow', 'departmentXdays_since_prior_order', 'departmentXorder_hour_of_day',
            'userXorder_dow', 'userXdays_since_prior_order', 'userXorder_hour_of_day',
            'user', 'aisle', 'department', 'product'],
        'key_list':
            [['user_id', 'aisle_id'], ['user_id', 'department_id'], ['user_id', 'product_id'],
            ['product_id', 'order_dow'], ['product_id', 'days_since_prior_order'], ['product_id', 'order_hour_of_day'],
            ['aisle_id', 'order_dow'], ['aisle_id', 'days_since_prior_order'], ['aisle_id', 'order_hour_of_day'],
            ['department_id', 'order_dow'], ['department_id', 'days_since_prior_order'], ['department_id', 'order_hour_of_day'],
            ['user_id', 'order_dow'], ['user_id', 'days_since_prior_order'], ['user_id', 'order_hour_of_day'],
            ['user_id'], ['aisle_id'], ['department_id'], ['product_id']],
        'f':
            [   KEY_GEN_FUNC[0], KEY_GEN_FUNC[1], KEY_GEN_FUNC[2],
                KEY_GEN_FUNC[3], KEY_GEN_FUNC[4], KEY_GEN_FUNC[5],
                KEY_GEN_FUNC[6], KEY_GEN_FUNC[7], KEY_GEN_FUNC[8],
                KEY_GEN_FUNC[6], KEY_GEN_FUNC[7], KEY_GEN_FUNC[8],
                KEY_GEN_FUNC[9], KEY_GEN_FUNC[10], KEY_GEN_FUNC[11],
                KEY_GEN_FUNC[12], KEY_GEN_FUNC[12], KEY_GEN_FUNC[12], KEY_GEN_FUNC[12]]})

if add_CrossFeature:
    for cross_key in CROSS_KEY_NAME:
        CONTINUOUS_COLUMNS = [cross_key + '_orders', cross_key + '_reorders', cross_key + '_reorder_rate']
                                #cross_key + '_mean_cart_pos', cross_key + '_median_cart_pos']
                                #cross_key + '_mean_cart_pos', cross_key + '_median_cart_pos']

if add_UserXProduct:
    UP_Features = ['UP_orders', 'UP_orders_ratio',
       'UP_average_pos_in_cart', 'UP_reorder_rate', 'UP_orders_since_last',
       'UP_delta_hour_vs_last', 'UP_reorders'] # 'dow', 'UP_same_dow_as_last_order'
    CONTINUOUS_COLUMNS += UP_Features

dnn_features = CONTINUOUS_COLUMNS + CATEGORICAL_COLUMNS
lgbm_features = CONTINUOUS_COLUMNS + CATEGORICAL_COLUMNS# + ['DNN']

print('loading prior')
priors = pd.read_csv(IDIR + 'order_products__prior.csv', dtype={
            'order_id': np.int32,
            'product_id': np.uint16,
            'add_to_cart_order': np.int16,
            'reordered': np.int8})
#priors = priors[:10000]

print('loading train')
train = pd.read_csv(IDIR + 'order_products__train.csv', dtype={
            'order_id': np.int32,
            'product_id': np.uint16,
            'add_to_cart_order': np.int16,
            'reordered': np.int8})
# train = train[:8]
print('loading orders')
orders = pd.read_csv(IDIR + 'orders.csv', dtype={
        'order_id': np.int32,
        'user_id': np.int32,
        'eval_set': 'category',
        'order_number': np.int16,
        'order_dow': np.int8,
        'order_hour_of_day': np.int8,
        'days_since_prior_order': np.float32})

print('loading products')
products = pd.read_csv(IDIR + 'products.csv', dtype={
        'product_id': np.uint16,
        'product_name': np.str,
        #'aisle_id': np.str,
        #'department_id': np.str
        'aisle_id': np.uint8,
        'department_id': np.uint8
            },
        usecols=['product_id', 'aisle_id', 'department_id'])
#products.to_csv('products')
#exit(0)
print('priors {}: {}'.format(priors.shape, ', '.join(priors.columns)))
print('orders {}: {}'.format(orders.shape, ', '.join(orders.columns)))
print('train {}: {}'.format(train.shape, ', '.join(train.columns)))

###

#print('computing product f')
#prods = pd.DataFrame()
#prods['orders'] = priors.groupby(priors.product_id).size().astype(np.int32)
#prods['reorders'] = priors['reordered'].groupby(priors.product_id).sum().astype(np.float32)
#prods['reorder_rate'] = np.log(prods.reorders / prods.orders + 1).astype(np.float32)
#products = products.join(prods, on='product_id')
#products.set_index('product_id', drop=False, inplace=True)
#
#print ('computing aisle_id related features')
#aisles = products.groupby('aisle_id').agg({'orders': 'sum', 'reorders': 'sum'})
#aisles['reorder_rate'] = np.log(aisles['reorders'] / aisles['orders'] + 1)
#aisles = aisles.astype(
#    {'orders': np.uint32, 'reorders': np.uint32, 'reorder_rate': np.float32}, inplace=True)
#departments = products.groupby('department_id').agg({'orders': 'sum', 'reorders': 'sum'})
#departments['reorder_rate'] = np.log(departments['reorders'] / departments['orders'] + 1)
#departments = departments.astype(
#    {'orders': np.uint32, 'reorders': np.uint32, 'reorder_rate': np.float32}, inplace=True)
#Del prods

print('add order info to priors')
orders.set_index('order_id', inplace=True, drop=False)
priors = priors.join(orders, on='order_id', rsuffix='_')
priors.drop('order_id_', inplace=True, axis=1)

### user features


print('computing user f')
usr = pd.DataFrame()
usr['average_days_between_orders'] = orders.groupby('user_id')['days_since_prior_order'].mean().astype(np.float32)
usr['nb_orders'] = orders.groupby('user_id').size().astype(np.int16)

users = pd.DataFrame()
users['total_items'] = priors.groupby('user_id').size().astype(np.int16)
users['all_products'] = priors.groupby('user_id')['product_id'].apply(set)
users['total_distinct_items'] = (users.all_products.map(len)).astype(np.int16)

users = users.join(usr)
del usr
users['average_basket'] = (users.total_items / users.nb_orders).astype(np.float32)
print('user f', users.shape)

### userXproduct features
if add_UserXProduct:
    print('compute userXproduct f - this is long...')
    userXproduct = priors.copy()
    userXproduct['user_product'] = userXproduct.product_id + userXproduct.user_id * 100000
    userXproduct = userXproduct.sort_values('order_number')
    userXproduct = userXproduct \
        .groupby('user_product', sort=False) \
        .agg({'order_id': ['size', 'last'], 
                'add_to_cart_order': 'sum',
                'reordered': 'sum'})
    userXproduct.columns = ['nb_orders', 'last_order_id', 'sum_pos_in_cart', 'reorders']
    userXproduct = userXproduct.astype(
        {'nb_orders': np.int16, 'last_order_id': np.int32, 'sum_pos_in_cart': np.int16, 'reorders': np.int16}, 
        inplace=True)


def gen_cross_feature(raw_data, key_list, f, cross_key):
    """
    """
    raw_data[cross_key] = f(raw_data, key_list)
    #raw_data.to_csv('priors_after_cross')
    #exit(0)
    cross_feature = raw_data \
        .groupby(cross_key, sort=False) \
        .agg({'reordered': ['size', 'sum']}) 
               # 'add_to_cart_order': ['mean', 'median']})
               # 'order_dow': ['mean', 'median']})
    cross_feature.columns = ['orders', 'reorders'] #, 'mean_cart_pos', 'median_cart_pos']
    cross_feature['reorder_rate'] = cross_feature['reorders'] / cross_feature['orders']
    cross_feature = cross_feature.astype(
        {'orders': np.uint32, 'reorders': np.uint32, 'reorder_rate': np.float32}, inplace=True)
    # cross_feature.to_csv(cross_key)
    raw_data.drop(cross_key, inplace=True, axis=1)
    return cross_feature    

def gen_cross_features(config, data):
    """
    """
    cross_features = dict()   
    for row in config.itertuples():
        if row.cross_key in CROSS_KEY_NAME:
            print 'compute %s f - this is long...' % row.cross_key
            cross_features[row.cross_key] = gen_cross_feature(data, row.key_list, row.f, row.cross_key)
    return cross_features

products.set_index('product_id', inplace=True, drop=False)
priors = priors.join(products, on = 'product_id', rsuffix = '_')
priors.drop(['product_id_'], inplace = True, axis = 1)
# print priors.dtypes
# priors[['product_id', 'aisle_id', 'department_id']].to_csv('priors_part_before_cross')
#exit(0)
##print priors.dtypes
cross_features = gen_cross_features(CROSS_FEATURE_CONFIG, priors)
#exit(0)
del priors

### train / test orders ###
print('split orders : train, test')
test_orders = orders[orders.eval_set == 'test']
train_orders = orders[orders.eval_set == 'train']

train.set_index(['order_id', 'product_id'], inplace=True, drop=False)

### build list of candidate products to reorder, with features ###
train_index = set(train.index)


def join_cross_feature(data, cross_feature, config):
    """
    """
    cross_key = config.cross_key
    f = config.f
    key_list = config.key_list
    data[cross_key] = f(data, key_list)
    data[cross_key + '_orders'] = data[cross_key].map(cross_feature.orders)
    data[cross_key + '_reorders'] = data[cross_key].map(cross_feature.reorders)
    data[cross_key + '_reorder_rate'] = data[cross_key].map(cross_feature.reorder_rate)
  #  data[cross_key + '_mean_cart_pos'] = data[cross_key].map(cross_feature.mean_cart_pos)
  #  data[cross_key + '_median_cart_pos'] = data[cross_key].map(cross_feature.median_cart_pos)
    data.drop(cross_key, axis=1, inplace=True)


def features(selected_orders, labels_given=False):
    print('build candidate list')
    order_list = []
    product_list = []
    labels = []
    i=0
    for row in selected_orders.itertuples():
        i+=1
        if i%10000 == 0: 
            print('order row',i)
            if small_test:
                break
        order_id = row.order_id
        user_id = row.user_id
        user_products = users.all_products[user_id]
        product_list += user_products
        order_list += [order_id] * len(user_products)
        if labels_given:
            labels += [(order_id, product) in train_index for product in user_products]
        if debug:
            break
        
    df = pd.DataFrame({'order_id':order_list, 'product_id':product_list}, dtype=np.int32)
    labels = np.array(labels, dtype=np.int8)
    del order_list
    del product_list
    
    print('user related features')
    df['user_id'] = df.order_id.map(orders.user_id)
    df['user_total_orders'] = df.user_id.map(users.nb_orders)
    df['user_total_items'] = df.user_id.map(users.total_items)
    df['total_distinct_items'] = df.user_id.map(users.total_distinct_items)
    df['user_average_days_between_orders'] = df.user_id.map(users.average_days_between_orders)
    df['user_average_basket'] =  df.user_id.map(users.average_basket)
    
    print('order related features')
    df['order_dow'] = df.order_id.map(orders.order_dow)
    df['order_hour_of_day'] = df.order_id.map(orders.order_hour_of_day)
    df['days_since_prior_order'] = df.order_id.map(orders.days_since_prior_order)
    df['days_since_ratio'] = df.days_since_prior_order / df.user_average_days_between_orders
    
    print('product related features')
    df['aisle_id'] = df.product_id.map(products.aisle_id)
    df['department_id'] = df.product_id.map(products.department_id)
   # df['product_orders'] = df.product_id.map(products.orders).astype(np.int32)
   # df['product_reorders'] = df.product_id.map(products.reorders)
   # df['product_reorder_rate'] = df.product_id.map(products.reorder_rate)
   # 
   # print('aisle related features')
   # df['aisle_orders'] = df.aisle_id.map(aisles.orders)
   # df['aisle_reorders'] = df.aisle_id.map(aisles.reorders)  
   # df['aisle_reorder_rate'] = df.aisle_id.map(aisles.reorder_rate)
   # 
   # print('department related features')
   # df['department_orders'] = df.department_id.map(departments.orders)
   # df['department_reorders'] = df.department_id.map(departments.reorders)  
   # df['department_reorder_rate'] = df.department_id.map(departments.reorder_rate)
    
    for row in CROSS_FEATURE_CONFIG.itertuples():
        if row.cross_key in CROSS_KEY_NAME:
            print '%s related features' % row.cross_key
            join_cross_feature(df, cross_features[row.cross_key], row)
    
    if add_UserXProduct:
        print('user_X_product related features')
        df['z'] = df.user_id * 100000 + df.product_id
        df.drop(['user_id'], axis=1, inplace=True)
        df['UP_orders'] = df.z.map(userXproduct.nb_orders)
        df['UP_orders_ratio'] = (df.UP_orders / df.user_total_orders).astype(np.float32)
        df['UP_last_order_id'] = df.z.map(userXproduct.last_order_id)
        df['UP_average_pos_in_cart'] = (df.z.map(userXproduct.sum_pos_in_cart) / df.UP_orders).astype(np.float32)
        df['UP_reorders'] = df.z.map(userXproduct.reorders)
        df['UP_reorder_rate'] = np.log((df.UP_reorders / df.UP_orders) + 1).astype(np.float32)
        df['UP_orders_since_last'] = df.user_total_orders - df.UP_last_order_id.map(orders.order_number)
        df['UP_delta_hour_vs_last'] = abs(df.order_hour_of_day - df.UP_last_order_id.map(orders.order_hour_of_day)).map(lambda x: min(x, 24-x)).astype(np.int8)
        df.drop(['UP_last_order_id', 'z'], axis=1, inplace=True)
    #df['UP_same_dow_as_last_order'] = df.UP_last_order_id.map(orders.order_dow) == \
    #                                              df.order_id.map(orders.order_dow)
    if labels_given:
        df[LABEL_COLUMN] = labels
    print(df.dtypes)
    print(df.memory_usage())
    return (df, labels)

                      
def build_estimator(model_dir, model_type):
    """Build an estimator."""
    # Sparse base columns.
    aisle_id = tf.contrib.layers.sparse_column_with_hash_bucket(
      "aisle_id", hash_bucket_size=150)
    department_id = tf.contrib.layers.sparse_column_with_hash_bucket(
      "department_id", hash_bucket_size=25)

  # Continuous base columns.
    user_total_orders = tf.contrib.layers.real_valued_column("user_total_orders")
    user_total_items = tf.contrib.layers.real_valued_column("user_total_items")
    total_distinct_items = tf.contrib.layers.real_valued_column("total_distinct_items")
    user_average_days_between_orders = tf.contrib.layers.real_valued_column("user_average_days_between_orders")
    order_hour_of_day = tf.contrib.layers.real_valued_column("order_hour_of_day")
    days_since_prior_order = tf.contrib.layers.real_valued_column("days_since_prior_order")
    # days_since_ratio = tf.contrib.layers.real_valued_column("days_since_ratio")
    product_orders = tf.contrib.layers.real_valued_column("product_orders")
    product_reorders = tf.contrib.layers.real_valued_column("product_reorders")
    product_reorder_rate = tf.contrib.layers.real_valued_column("product_reorder_rate")
    
    if add_UserXProduct:
        UP_orders = tf.contrib.layers.real_valued_column("UP_orders")
        UP_orders_ratio = tf.contrib.layers.real_valued_column("UP_orders_ratio")
        UP_average_pos_in_cart = tf.contrib.layers.real_valued_column("UP_average_pos_in_cart")
        UP_reorder_rate = tf.contrib.layers.real_valued_column("UP_reorder_rate")
        UP_orders_since_last = tf.contrib.layers.real_valued_column("UP_orders_since_last")
        UP_delta_hour_vs_last = tf.contrib.layers.real_valued_column("UP_delta_hour_vs_last")

  # Transformations.

  # Wide columns and deep columns.
    wide_columns = [aisle_id, department_id,
                  tf.contrib.layers.crossed_column([aisle_id, department_id],
                                                   hash_bucket_size=int(3e3))]
    deep_columns = [
        tf.contrib.layers.embedding_column(aisle_id, dimension=8),
        tf.contrib.layers.embedding_column(department_id, dimension=5),
        user_total_orders,
        user_total_items,
        total_distinct_items,
        user_average_days_between_orders,
        order_hour_of_day,
        days_since_prior_order,
       # days_since_ratio,
        product_orders,
        product_reorders,
        product_reorder_rate]

    if add_UserXProduct:
        deep_columns += [
        UP_orders,
        UP_orders_ratio,
        UP_average_pos_in_cart,
        UP_reorder_rate,
        UP_orders_since_last,
        UP_delta_hour_vs_last]

    if model_type == "wide":
        m = tf.contrib.learn.LinearClassifier(model_dir=model_dir,
                                          feature_columns=wide_columns)
    elif model_type == "deep":
        m = tf.contrib.learn.DNNClassifier(model_dir=model_dir,
                                       feature_columns=deep_columns,
                                       hidden_units=[16, 8])
    else:
        m = tf.contrib.learn.DNNLinearCombinedClassifier(
        model_dir=model_dir,
        linear_feature_columns=wide_columns,
        dnn_feature_columns=deep_columns,
        dnn_hidden_units=[16, 8])
    return m

    
def normalize_tensor(x):
    """
    x: array
    """
    const_tensor = tf.constant(x.astype(np.float32))
    moment = tf.nn.moments(const_tensor, axes = [0])
    norm_tensor = (const_tensor - moment[0]) / tf.pow(moment[1], 0.5)
    return norm_tensor    


def normalize_tensor(x, mean, std):
    """
    x: array
    """
    norm_array = (x - mean) / std 
    norm_tensor = tf.constant(norm_array.values)
    return norm_tensor    


def input_fn(df, labels_given = False, mean = None, std = None):
    """Input builder function."""
    # Creates a dictionary mapping from each continuous feature column name (k) to
    # the values of that column stored in a constant Tensor.
    continuous_cols = {k: normalize_tensor(df[k], mean[k], std[k]) for k in CONTINUOUS_COLUMNS}
    # Creates a dictionary mapping from each categorical feature column name (k)
    # to the values of that column stored in a tf.SparseTensor.
    categorical_cols = {
        k: tf.SparseTensor(
          indices=[[i, 0] for i in range(df[k].size)],
          values=df[k].values,
          dense_shape=[df[k].size, 1])
        for k in CATEGORICAL_COLUMNS}
    # Merges the two dictionaries into one.
    feature_cols = dict(continuous_cols)
    feature_cols.update(categorical_cols)
    # Converts the label column into a constant Tensor.
    if labels_given:
        label = tf.constant(df[LABEL_COLUMN].values)
    else:
        label = None
    # Returns the feature columns and the label.
    return feature_cols, label


def F1_Score_by_threshold(labels, preds, threshold):
    """
    """
    tot_TP = 0.
    tot_FP = 0.
    tot_FN = 0.
    if len(labels) != len(preds):
        print "The length of labels are not equal preds!"
        exit(1)
    for i in xrange(len(labels)):
        # print "pred : %f label : %d th : %f" % (preds[i], labels[i], threshold)
        if preds[i] > threshold:
            if labels[i] == 1:
                tot_TP += 1
            else:
                tot_FP += 1
        elif labels[i] == 1:
            tot_FN += 1
    TP_FP = tot_TP + tot_FP
    TP_FN = tot_TP + tot_FN 
    if TP_FN == 0:
        mean_recall = 0
    else:
        mean_recall = tot_TP / TP_FN
    if TP_FP == 0:
        mean_precision = 0
    else:
        mean_precision = tot_TP / TP_FP
    
    if mean_recall == 0 or mean_precision == 0:
        F1_Score = 0
    else:
        F1_Score = 1. / (1. / mean_recall + 1. / mean_precision)
    print "threshold: %f" % threshold
    # print "TP: %f FP: %f FN: %f" % (tot_TP, tot_FP, tot_FN)
    print "mean_recall: %f mean_precision: %f F1_Score: %f" % (mean_recall, mean_precision, F1_Score)
    return F1_Score    


def create_model(input_len):
    model = Sequential()
    model.add(Dense(hidden_units[0], activation='sigmoid', input_dim = input_len))
    if DNN_BN:
        model.add(BatchNormalization()) 
    if dropout_rate > 0:
        model.add(Dropout(dropout_rate))
    model.add(Dense(hidden_units[1], activation='sigmoid'))
    if DNN_BN:
        model.add(BatchNormalization()) 
    if dropout_rate > 0:
        model.add(Dropout(dropout_rate))
    # model.add(Dropout(0.1))
    #model.add(Dense(32, activation='relu'))
    #model.add(Dropout(0.2))
    model.add(Dense(1, activation='sigmoid'))

    # optimizer = SGD(lr=1e-3, decay=1e-6, momentum=0.9, nesterov=True)
    optimizer = RMSprop(lr=1e-3, rho = 0.9, epsilon = 1e-8)
    model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics = ['accuracy'])

    return model

def create_embedding_model():
    """
    """
    aisle_id = Input(shape=(1,))
    aisle_embedding = Embedding(135, 6, input_length = 1)(aisle_id)
    aisle_embedding = Reshape((6,))(aisle_embedding)
    
    department_id = Input(shape=(1,))
    department_embedding = Embedding(22, 4, input_length = 1)(department_id)
    department_embedding = Reshape((4,))(department_embedding)
    
    #product_id = Input(shape=(1,))
    #product_embedding = Embedding(49969, 16, input_length = 1)(product_id)
    #product_embedding = Reshape((16,))(product_embedding)
    
    dense_input = Input(shape=(len(CONTINUOUS_COLUMNS),))
    merge_input = concatenate([dense_input, aisle_embedding, department_embedding], axis = 1)
    
    merge_len = len(CONTINUOUS_COLUMNS) + 6 + 4
    output = create_model(merge_len)(merge_input)

    model = Model([dense_input, aisle_id, department_id], output) 
    optimizer = RMSprop(lr=1e-3, rho = 0.9, epsilon = 1e-8)
    # optimizer = SGD(lr=1e-2, decay=1e-6, momentum=0.9, nesterov=True)
    model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics = ['accuracy'])

    return model


def merge_several_folds_mean(data, nfolds):
    a = np.array(data[0])
    for i in range(1, nfolds):
        a += np.array(data[i])
    a /= nfolds
    return a.tolist()


def find_best_th(labels, preds):
    """
    """    
    best_th = 0.
    max_F1 = 0.
    for th in [t * 0.01 for t in range(10, 30, 1)]:
        F1 = F1_Score_by_threshold(labels, preds, th)
        if F1 > max_F1:
            max_F1 = F1
            best_th = th
    print "Best_Threshold = %f  Max_F1 = %f" % (best_th, max_F1)
    return best_th, max_F1    


def keras_eval(models, data):
    """
    """
    preds = []
    for m in models:
        pred = m.predict(data, batch_size=batch_size, verbose=2)
        pred_array = [x[0] for x in pred]
        preds.append(pred_array)    
    avg_preds = merge_several_folds_mean(preds, len(models))
    
    return avg_preds


def keras_train(nfolds = 10):
    """
    Detect Fish or noFish
    """
    
    print "Start gen training data, shuffle and normalize!"
    df_train, labels = features(train_orders, labels_given=True)
    # df_train = df_train.sample(frac = 1).reset_index(drop = True)
    # labels = np_utils.to_categorical(labels, 2)
    
    df_train_part = df_train[dnn_features]
    train_target = df_train[LABEL_COLUMN].values
    # norm_min = df_train_part.min()
    # norm_max = df_train_part.max()
    norm_slope = 1. / df_train_part.std()
    norm_intercept = -1. * norm_slope * df_train_part.mean()
    norm_train = df_train_part * norm_slope + norm_intercept
    # norm_train.to_csv("norm_train", index = False)    

    train_data = norm_train[CONTINUOUS_COLUMNS].values
    aisle_ids = df_train_part['aisle_id'].values
    department_ids = df_train_part['department_id'].values
    # product_ids = df_train_part['product_id'].values
    
    # train_target = labels
    train_size = len(train_data)
    print "Training Data size : %d" % train_size
    df_test = train_data[train_size * 9 / 10 : ]
    aisle_id_test = aisle_ids[train_size * 9 / 10 : ]
    department_id_test = department_ids[train_size * 9 / 10 : ]
    # product_id_test = product_ids[train_size * 9 / 10 : ]
    df_test_label = train_target[train_size * 9 / 10 : ]
     
    yfull_train = dict()
    kf = KFold(len(labels), n_folds=nfolds, shuffle=True)
    num_fold = 0
    sum_score = 0
    models = []
    for train_index, test_index in kf:
        # model = create_model(classes = 2)
        model = create_embedding_model()
        
        X_train = train_data[train_index]
        Y_train = train_target[train_index]
        X_valid = train_data[test_index]
        Y_valid = train_target[test_index]
        aisle_id_train = aisle_ids[train_index]
        aisle_id_valide = aisle_ids[test_index]
        department_id_train = department_ids[train_index]
        department_id_valide = department_ids[test_index]
       # product_id_train = product_ids[train_index]
        #product_id_valide = product_ids[test_index]
        
       # print aisle_id_train
       # pd.DataFrame(X_train).to_csv("norm_train", index = False)
       # pd.DataFrame(Y_train).to_csv("train_labels", index = False)
       # pd.DataFrame(X_valid).to_csv("norm_valide", index = False)
       # pd.DataFrame(Y_valid).to_csv("valid_labels", index = False)
        
        num_fold += 1
        print('Start KFold number {} from {}'.format(num_fold, nfolds))
        print('Split train: ', len(X_train), len(Y_train))
        print('Split valid: ', len(X_valid), len(Y_valid))

        callbacks = [
            EarlyStopping(monitor='val_loss', patience=3, verbose=0),
        ]
        model.fit([X_train, aisle_id_train, department_id_train], Y_train, batch_size=batch_size, epochs=dnn_epoch,
                shuffle=True, verbose=2, validation_data=([X_valid, aisle_id_valide, department_id_valide], Y_valid)
                , callbacks=callbacks)
        # print model.get_weights()
        # predictions_valid = model.predict(X_valid, batch_size=batch_size, verbose=2)
        # predictions_train = model.predict(X_train, batch_size=batch_size, verbose=2)
        # pd.DataFrame(predictions_train).to_csv("re_train", index = False)
        # pd.DataFrame(predictions_valid).to_csv("re_valide", index = False)
        #score = log_loss(Y_valid, predictions_valid)
        #print('Score log_loss: ', score)
        #sum_score += score*len(test_index)

        models.append(model)
        if len(models) == 1:
            break
    
    avg_preds = keras_eval(models, [df_test, aisle_id_test, department_id_test])
    # pd.DataFrame(avg_preds).to_csv("tuneTh_pred", index = False)
    # pd.DataFrame(Y_valid).to_csv("tuneTh_labels", index = False)
        
    best_th, max_F1 = find_best_th(df_test_label, avg_preds)

    return ((models, norm_slope, norm_intercept), best_th, max_F1)


def tf_train(model_dir, model_type, train_steps):
    """Train and evaluate the model."""
    
    df_train, labels = features(train_orders, labels_given=True)
    #with open("train_label", "w") as label_file:
    #    for label in labels:
    #        print >> label_file, label
    df_train = df_train.sample(frac = 1).reset_index(drop = True)
    df_train_mean = df_train[CONTINUOUS_COLUMNS].mean(axis = 0)
    df_train_std = df_train[CONTINUOUS_COLUMNS].std(axis = 0)

    train_size = len(df_train)
    print "Training Data size : %d" % train_size
    df_train_part = df_train[ : train_size * 9 / 10]
    df_test = df_train[train_size * 9 / 10 : ]

    model_dir = tempfile.mkdtemp() if not model_dir else model_dir
    print("model directory = %s" % model_dir)
    
    m = build_estimator(model_dir, model_type)
    m.fit(input_fn=lambda: input_fn(df_train_part, True, df_train_mean, df_train_std), steps=train_steps)
    
    results = m.predict_proba(input_fn=lambda: input_fn(df_test, True, df_train_mean, df_train_std))
    df_test["pred"] = [p[1] for p in results]
    df_test.to_csv('re', index=False)
    
    best_th, max_F1 = find_best_th(df_test[LABEL_COLUMN].values, df_test['pred'].values)
    return m, best_th


def load_and_predict(model_dir, data = None):
    """
    load model from 
    """
    with tf.Session() as sess:    
        saver = tf.train.import_meta_graph(model_dir + './model.ckpt-16.meta')
        saver.restore(sess, tf.train.latest_checkpoint(model_dir))
        graph = tf.get_default_graph()
        prob = graph.get_tensor_by_name('predictions/probabilities:0')
        df_train, labels = features(train_orders, labels_given=True)
        sess.run(prob, input_fn(df_train)[0])
    

def lgbm_train(model_k = None):
    """
    LGB Training
    """
    # lgbm_features = f_to_use
    df_train, labels = features(train_orders, labels_given=True)
    
    if model_k != None:
        preds = model_eval(model_k, 'k', df_train[dnn_features])
        df_train['DNN'] = preds

    train_size = len(df_train)
    df_train_part = df_train[ : train_size * 9 / 10]
    df_test = df_train[train_size * 9 / 10 : ]
    d_train = lgb.Dataset(df_train_part[lgbm_features],
                          label=df_train_part[LABEL_COLUMN],
                          categorical_feature=['aisle_id', 'department_id'])  # , 'order_hour_of_day', 'dow'
    del df_train
    
    params = {
        'task': 'train',
        'boosting_type': 'gbdt',
        'objective': 'binary',
        'metric': {'binary_logloss'},
      #  'num_leaves': 256,
      #  'max_depth': 12,
      #  'feature_fraction': 0.9,
      #  'bagging_fraction': 0.95,
      #  'bagging_freq': 5,
        'num_leaves': 300,
        'min_sum_hessian_in_leaf': 20,
        'max_depth': 12,
        'learning_rate': 0.05,
        'feature_fraction': 0.6,
        'verbose': 1
    }
    
    ROUNDS = lgbm_epoch
    
    print('light GBM train :-)')
    bst = lgb.train(params, d_train, ROUNDS)
    # lgb.plot_importance(bst, figsize=(9,20))
    del d_train
    
    print('light GBM test')
    preds = bst.predict(df_test[lgbm_features])
        
    df_test['pred'] = preds
    best_th, max_F1 = find_best_th(df_test[LABEL_COLUMN].values, df_test['pred'].values)
    # pd.DataFrame(params).to_csv(str(max_F1) + str(lgbm_epoch)) 
    print str(params) + str(max_F1) + "_" + str(lgbm_epoch)
    return bst, best_th, max_F1


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


def gen_sub(models, model_type, best_th, F1, model_k = None):
    """
    Evaluate single Type model
    """
    df_valide, _ = features(test_orders)
    
    if model_type == 'k':
        valide_features = dnn_features
    elif model_type == 'l':
        valide_features = lgbm_features 
   
    if model_k != None:
        preds = model_eval(model_k, 'k', df_valide[dnn_features])
        df_valide['DNN'] = preds
    
    preds = model_eval(models, model_type, df_valide[valide_features]) 
    df_valide['pred'] = preds
    
    TRESHOLD = best_th  # guess, should be tuned with crossval on a subset of train data
    
    d = dict()
    for row in df_valide.itertuples():
        if row.pred > TRESHOLD:
            try:
                d[row.order_id] += ' ' + str(row.product_id)
            except:
                d[row.order_id] = str(row.product_id)
    
    for order in test_orders.order_id:
        if order not in d:
            d[order] = 'None'
    
    sub = pd.DataFrame.from_dict(d, orient='index')
    
    sub.reset_index(inplace=True)
    sub.columns = ['order_id', 'products']
    sub_name = 'sub_' + model_type + "_" + str(F1) + \
        "_stdNorm_sigmoid_" + str(hidden_units) + "_" + \
        str(dropout_rate) + "_" + str(dnn_epoch) + \
        "_" + str(lgbm_epoch) + "_log.csv"
    sub.to_csv(sub_name, index=False)
    

if __name__ == "__main__":
    model_k, th, F1 = keras_train(10)
    # gen_sub(model_k, 'k', th, F1)
    #model_l, th, F1 = lgbm_train()#model_k)
    #gen_sub(model_l, 'l', th, F1) #model_k)
