import lightgbm as lgb
import pandas as pd

def lgbm_train(train_part, train_part_label, valide_part, valide_part_label, fold_seed,
        fold = 5, train_weight = None, valide_weight = None):
    """
    LGBM Training
    """
    print("-----LGBM training-----")

    d_train = lgb.Dataset(train_part, train_part_label, weight = train_weight)#, init_score = train_part[:, -1])
    d_valide = lgb.Dataset(valide_part, valide_part_label, weight = valide_weight)#, init_score = valide_part[:, -1])
    params = {
            'task': 'train',
            'boosting_type': 'gbdt', #'gbdt',
            'objective': 'binary',
            'metric': {'auc', 'binary_logloss'},
            'num_leaves': 15, #60, #40, # 60,
           # 'min_sum_hessian_in_leaf': 10,
            'max_depth': 5,#12, #6, # 10,
            'learning_rate': 0.03, # 0.025,
            'feature_fraction': 0.5,#0.35, # 0.6
            'verbose': 0,
          #   'valid_sets': [d_valide],
            'num_boost_round': 1500, #361,
            'feature_fraction_seed': fold_seed,
            #'drop_rate': 0.05,
            # 'bagging_fraction': 0.8,
            # 'bagging_freq': 20,
            # 'bagging_seed': fold_seed,
             'early_stopping_round': 150,
            # 'random_state': 10
            # 'verbose_eval': 20
            #'min_data_in_leaf': 665
        }

    bst = lgb.train(
                    params ,
                    d_train,
                    verbose_eval = 50,
                    valid_sets = [d_train, d_valide],
                    # feature_name=['f' + str(i + 1) for i in range(train_part.shape[1])],
                    #feval = gini_lgbm
                    #num_boost_round = 1
                    )
    #pred = model_eval(bst, 'l', valide_part)
    #print(pred[:10])
    #print(valide_part_label[:10])
    #print(valide_part[:10, -1])
    # exit(0)
    #feature_imp = bst.feature_importance(importance_type = 'gain')
    #print (feature_name[np.argsort(feature_imp)])
    # exit(0)
    # cv_result = lgb.cv(params, d_train, nfold=fold) #, feval = gini_lgbm)
    # pd.DataFrame(cv_result).to_csv('cv_result', index = False)
    # exit(0)
    return bst