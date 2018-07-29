import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import os
# print(os.listdir("../input"))

import lightgbm as lgb
from sklearn.model_selection import *
from sklearn.metrics import mean_squared_error, make_scorer
from scipy.stats import mode, skew, kurtosis, entropy
from sklearn.ensemble import ExtraTreesRegressor

# import matplotlib.pyplot as plt
# import seaborn as sns

# import dask.dataframe as dd
# from dask.multiprocessing import get

from tqdm import tqdm, tqdm_notebook
tqdm.pandas(tqdm_notebook)
import concurrent.futures

import pickle

# Any results you write to the current directory are saved as output.
path = "../../Data/"
train = pd.read_csv(path + "train.csv", index_col = 'ID')
test = pd.read_csv(path + "test.csv", index_col = 'ID')

debug = False
if debug:
    train = train[:1000]
    test = test[:1000]

transact_cols = [f for f in train.columns if f not in ["ID", "target"]]
y = np.log1p(train["target"]).values

cols = ['f190486d6', '58e2e02e6', 'eeb9cd3aa', '9fd594eec', '6eef030c1',
       '15ace8c9f', 'fb0f5dbfe', '58e056e12', '20aa07010', '024c577b9',
       'd6bb78916', 'b43a7cfd5', '58232a6fb', '1702b5bf0', '324921c7b', 
       '62e59a501', '2ec5b290f', '241f0f867', 'fb49e4212',  '66ace2992',
       'f74e8f13d', '5c6487af1', '963a49cdc', '26fc93eb7', '1931ccfdd', 
       '703885424', '70feb1494', '491b9ee45', '23310aa6f', 'e176a204a',
       '6619d81fc', '1db387535', 'fc99f9426', '91f701ba2',  '0572565c2',
       '190db8488',  'adb64ff71', 'c47340d97', 'c5a231d81', '0ff32eb98'] 

from multiprocessing import Pool
CPU_CORES = 6
def _get_leak(df, cols, search_ind, lag=0):
    """ To get leak value, we do following:
       1. Get string of all values after removing first two time steps
       2. For all rows we shift the row by two steps and again make a string
       3. Just find rows where string from 2 matches string from 1
       4. Get 1st time step of row in 3 (Currently, there is additional condition to only fetch value if we got exactly one match in step 3)"""
    series_str = df[cols[lag+2:]].apply(lambda x: "_".join(x.round(2).astype(str)), axis=1)
    series_shifted_str = df.loc[search_ind, cols].shift(lag+2, axis=1)[cols[lag+2:]].apply(lambda x: "_".join(x.round(2).astype(str)), axis=1)
    target_rows = series_shifted_str.progress_apply(lambda x: np.where(x == series_str)[0])
    # print(target_rows)
    del series_str, series_shifted_str
    target_vals = target_rows.apply(lambda x: df.iloc[x[0]][cols[lag]] if len(x)==1 else 0)
    return target_vals

def get_all_leak(df, cols=None, nlags=15):
    """
    We just recursively fetch target value for different lags
    """
    df =  df.copy()
    with Pool(processes=CPU_CORES) as p:
        begin_ind = 0
        end_ind = 0
        leak_target = pd.Series(0, index = df.index)
        while begin_ind < nlags:
            end_ind = min(begin_ind + CPU_CORES, nlags)
            search_ind = (leak_target == 0)
            # print(search_ind)
            print('begin_ind: ', begin_ind, 'end_ind: ', end_ind, "search_ind_len: ", search_ind.sum())
            res = [p.apply_async(_get_leak, args=(df, cols, search_ind, i)) for i in range(begin_ind, end_ind)]
            for r in res:
                target_vals = r.get()
                leak_target[target_vals.index] = target_vals
                # print("leak_target", leak_target)
                # res = [r.get() for r in res]
            begin_ind = end_ind
    df['leak_target'] = leak_target
    # for i in range(nlags):
    #     print("Processing lag {}".format(i))
    #     df["leaked_target_"+str(i)] = _get_leak(df, cols, i)

    # MAX_WORKERS = 4
    # col_ind_begin = 0
    # col_len = nlags
    # while col_ind_begin < col_len:
    #     col_ind_end = min(col_ind_begin + MAX_WORKERS, col_len)
    #     with concurrent.futures.ThreadPoolExecutor(max_workers = MAX_WORKERS) as executor:
    #         future_predict = {executor.submit(_get_leak, df, cols, ind): ind for ind in range(col_ind_begin, col_ind_end)}
    #         for future in concurrent.futures.as_completed(future_predict):
    #             ind = future_predict[future]
    #             try:
    #                 df["leaked_target_"+str(ind)] = future.result()
    #             except Exception as exc:
    #                 print('%dth feature normalize generate an exception: %s' % (ind, exc))
    #     col_ind_begin = col_ind_end
    #     if col_ind_begin % 1 == 0:
    #         print('Gen %d normalized features' % col_ind_begin)
    return df

test["target"] = train["target"].mean()

# all_df = pd.concat([train[["ID", "target"] + cols], test[["ID", "target"]+ cols]]).reset_index(drop=True)
all_df = pd.concat([train[["target"] + cols], test[["target"]+ cols]]) #.reset_index(drop=True)
all_df.head()

NLAGS = 30 #Increasing this might help push score a bit
all_df = get_all_leak(all_df, cols=cols, nlags=NLAGS)

with open(path + 'leak_target.pickle', 'wb+') as handle:
    pickle.dump(all_df[['target', 'leak_target']], handle, protocol=pickle.HIGHEST_PROTOCOL)
exit(0)

leaky_cols = ["leaked_target_"+str(i) for i in range(NLAGS)]
train = train.join(all_df.set_index("ID")[leaky_cols], on="ID", how="left")
test = test.join(all_df.set_index("ID")[leaky_cols], on="ID", how="left")

train[["target"]+leaky_cols].head(10)

train["nonzero_mean"] = train[transact_cols].apply(lambda x: np.expm1(np.log1p(x[x!=0]).mean()), axis=1)
test["nonzero_mean"] = test[transact_cols].apply(lambda x: np.expm1(np.log1p(x[x!=0]).mean()), axis=1)

#We start with 1st lag target and recusrsively fill zero's
train["compiled_leak"] = 0
test["compiled_leak"] = 0
for i in range(NLAGS):
    train.loc[train["compiled_leak"] == 0, "compiled_leak"] = train.loc[train["compiled_leak"] == 0, "leaked_target_"+str(i)]
    test.loc[test["compiled_leak"] == 0, "compiled_leak"] = test.loc[test["compiled_leak"] == 0, "leaked_target_"+str(i)]
    
print("Leak values found in train and test ", sum(train["compiled_leak"] > 0), sum(test["compiled_leak"] > 0))
print("% of correct leaks values in train ", sum(train["compiled_leak"] == train["target"])/sum(train["compiled_leak"] > 0))

# train.loc[train["compiled_leak"] == 0, "compiled_leak"] = train.loc[train["compiled_leak"] == 0, "nonzero_mean"]
# test.loc[test["compiled_leak"] == 0, "compiled_leak"] = test.loc[test["compiled_leak"] == 0, "nonzero_mean"]

from sklearn.metrics import mean_squared_error
np.sqrt(mean_squared_error(y, np.log1p(train["compiled_leak"]).fillna(14.49)))

#submission
sub = test[["ID"]]
sub["target"] = test["compiled_leak"]
sub.to_csv(path + "baseline_submission_with_leaks.csv", index=False)