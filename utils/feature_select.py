import numpy as np
import pandas as pd
import warnings
import matplotlib.pyplot as plt
from lightgbm import LGBMClassifier, LGBMRegressor
from pandas.core.common import SettingWithCopyWarning
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import KFold, StratifiedKFold
from util import rmse, kfold_lightgbm

warnings.simplefilter(action='ignore', category=SettingWithCopyWarning)
warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.filterwarnings('ignore')


FEATS_EXCLUDED = ['first_active_month', 'target', 'card_id', 'outliers',
                  'oof_preds', 'origin', ]


def check_score_improve_or_reduce(train_df, num_folds, stratified=True, mode="improve", debug=False, name="", need_result=False, fold_random_state=42):

    print("Starting LightGBM. Train shape: {}, test shape: {}".format(
        train_df.shape))
#     for col in sorted(train_df.columns):
#         print(col,":",train_df[col].dtypes)
    # Cross validation model
    if stratified:
        folds = StratifiedKFold(
            n_splits=num_folds, shuffle=True, random_state=fold_random_state)
    else:
        folds = KFold(n_splits=num_folds, shuffle=True,
                      random_state=fold_random_state)

    # Create arrays and dataframes to store results
    feats = [f for f in train_df.columns if f not in FEATS_EXCLUDED]
    feats = [f for f in feats if len(train_df[f].unique()) > 1]
    oof_preds = np.zeros(len(train_df))

    cv_scores = []
    scores = [[], []]
    # k-fold
    if debug:
        feats = feats[:2]
    for feat_num in range(1, len(feats)+1):
        if mode == "improve":
            feat = feats[:feat_num]
            new_feat = feats[feat_num-1]
        elif mode == "reduce":
            feat = feats[feat_num-1:]
            new_feat = feats[feat_num-1]
        else:
            raise ValueError
        cv_score = 0
        for n_fold, (train_idx, valid_idx) in enumerate(folds.split(train_df[feats], train_df['target'])):
            trn_x, trn_y = train_df[feat].iloc[train_idx], train_df['target'].iloc[train_idx]
            val_x, val_y = train_df[feat].iloc[valid_idx], train_df['target'].iloc[valid_idx]
            params = {
                'objective': 'regression_l2',
                'boosting_type': 'gbdt',
                'n_jobs': 4, 'max_depth': 7,
                'n_estimators': 2000,
                'subsample_freq': 2,
                'subsample_for_bin': 200000,
                'min_data_per_group': 100,
                'max_cat_to_onehot': 4,
                'cat_l2': 10.0,
                'cat_smooth': 10.0,
                'max_cat_threshold': 32,
                'metric_freq': 10,
                'verbosity': -1,
                'metric': 'rmse',
                'num_iterations': 10000,
                'colsample_bytree': 0.5,
                'learning_rate': 0.0061033234451294376,
                'min_child_samples': 80,
                'min_child_weight': 100.0,
                'min_split_gain': 1e-06,
                'num_leaves': 47,
                'reg_alpha': 10.0,
                'reg_lambda': 10.0,
                'subsample': 0.9}
            # set data structure
            clf = LGBMRegressor(**params)
            clf.fit(
                trn_x, trn_y,
                eval_set=[(trn_x, trn_y), (val_x, val_y)],
                eval_metric="rmse",
                verbose=0,
                early_stopping_rounds=100,
            )

        # params optimized by optuna

            oof_preds[valid_idx] = clf.predict(
                val_x, num_iteration=clf.best_iteration_)
            score = rmse(val_y, oof_preds[valid_idx])
            scores[n_fold].append(score)

            print('no {}-fold loss: {:.6f}'.format(n_fold + 1,
                                                   score))
        # params optimized by optuna

            cv_score += score/folds.n_splits

        print("column #:{} new column :'{}' cross validation score:{:.6f}".format(
            len(feat), new_feat, cv_score))
        cv_scores.append(cv_score)
    result = {"cv_score": cv_scores,
              "cv1_score": scores[0], "cv2_score": scores[1]}
    return pd.DataFrame(result, index=feats)


def get_diff(result_dic, name="reduce"):
    t_col = [col for col in result_dic.columns if "diff" not in col]
    for col in t_col:
        print(result_dic[col].values[1:].shape)
        if name == "reduce":
            new = np.append(
                (result_dic[col].values[1:] - result_dic[col].values[:-1]), np.zeros(1))
        elif name == "improve":
            new = np.append(
                np.zeros(1), -(result_dic[col].values[1:] - result_dic[col].values[:-1]))
        else:
            raise NotImplementedError
        print(new.shape)
        result_dic[col+"_diff"] = new
    return result_dic


def get_no_influence_column(improve_result, reduce_result):
    mask_total = improve_result["cv_score_diff"] < 0
    mask_1 = improve_result["cv1_score_diff"] < 0
    mask_2 = improve_result["cv2_score_diff"] < 0
    total_no_improve_col = improve_result.loc[mask_total].index
    one_no_improve_col = improve_result.loc[(mask_1) | (mask_2)].index
    all_no_improve_col = improve_result.loc[(mask_1) & (mask_2)].index

    mask_total = reduce_result["cv_score_diff"] < 0
    mask_1 = reduce_result["cv1_score_diff"] < 0
    mask_2 = reduce_result["cv2_score_diff"] < 0

    total_no_reduce_col = reduce_result.loc[mask_total].index
    one_no_reduce_col = reduce_result.loc[(mask_1) | (mask_2)].index
    all_no_reduce_col = reduce_result.loc[(mask_1) & (mask_2)].index

    result_dict = {"no_improve": {
        "total": total_no_improve_col,
        "one": one_no_improve_col,
        "all": all_no_improve_col,
    },
        "no_reduce": {
        "total": total_no_reduce_col,
        "one": one_no_reduce_col,
        "all": all_no_reduce_col
    },
        "no_both": {
        "total": list(set(total_no_improve_col).intersection(set(total_no_reduce_col))),
        "one": list(set(one_no_improve_col).intersection(set(one_no_reduce_col))),
        "all": list(set(all_no_improve_col).intersection(set(all_no_reduce_col))),
    },
        "no_either": {
        "total": list(set(total_no_improve_col) | (set(total_no_reduce_col))),
        "one": list(set(one_no_improve_col) | (set(one_no_reduce_col))),
        "all": list(set(all_no_improve_col) | (set(all_no_reduce_col))),

    }
    }

    return result_dict

# result_dict = get_no_influence_column(improve_result, reduce_result)


def delete_no_use_col_to_kfold(col_information, key, train_df, test_df):
    print("-------- start {} ---------".format(key))
    if isinstance(key, str):
        delete_col = col_information[key]
        name = key
    else:
        key1, key2 = key
        name = key1 + "_" + key2
        delete_col = col_information[key1][key2]
    _, _, _, _, cv_score = kfold_lightgbm(train_df.drop(delete_col, axis=1), test_df.drop(
        delete_col, axis=1), num_folds=5, stratified=False, need_result=True)

    return {name: cv_score}


def check_datashift(train_df, test_df, num_folds, samplesize=50000, stratified=True, debug=False, name="", need_result=False, fold_random_state=42):

    print("Starting LightGBM. Train shape: {}, test shape: {}".format(
        train_df.shape, test_df.shape))
#     for col in sorted(train_df.columns):
#         print(col,":",train_df[col].dtypes)
    # Cross validation model
    if stratified:
        folds = StratifiedKFold(
            n_splits=num_folds, shuffle=True, random_state=fold_random_state)
    else:
        folds = KFold(n_splits=num_folds, shuffle=True,
                      random_state=fold_random_state)

    # Create arrays and dataframes to store results
    feats = [f for f in train_df.columns if f not in FEATS_EXCLUDED]
    mini_train = train_df.sample(test_df.shape[0], random_state=100)
    mini_test = test_df

    mini_train["origin"] = True
    mini_test["origin"] = False
    x_train = mini_train[feats+["origin"]].append(mini_test[feats+["origin"]])
    x_train = x_train.sample(frac=1)
    print("x train shape:", x_train.shape, len(feats))
    oof_preds = np.zeros(len(x_train))

    cv_scores = []
    # k-fold
    if debug:
        feats = feats[:2]
    for feat in feats:
        cv_score = 0
        for n_fold, (train_idx, valid_idx) in enumerate(folds.split(x_train[feats], x_train['origin'])):
            trn_x, trn_y = x_train[[
                feat]].iloc[train_idx], x_train['origin'].iloc[train_idx]
            val_x, val_y = x_train[[
                feat]].iloc[valid_idx], x_train['origin'].iloc[valid_idx]
            params = {
                'objective': 'binary',
                'min_data': 1
            }
            # set data structure
            try:
                clf = LGBMClassifier(**params)
                clf.fit(
                    trn_x, trn_y,
                    eval_set=[(trn_x, trn_y), (val_x, val_y)],
                    eval_metric="auc",
                    verbose=0,
                    early_stopping_rounds=100,
                )

            # params optimized by optuna

                oof_preds[valid_idx] = clf.predict_proba(
                    val_x, num_iteration=clf.best_iteration_)[:, 1]

                score = roc_auc_score(val_y, oof_preds[valid_idx])
                cv_score += score/folds.n_splits
            except:
                print("column:{} cause LightGBMError".format(feat))
#             print('Fold %2d RMSE : %.6f' % (n_fold + 1, score))

        print("column '{}' cross validation score:{:.6f}".format(feat, cv_score))
        cv_scores.append(cv_score)
    return pd.DataFrame(cv_scores, index=feats, columns=["roc_auc"])


def check_dist(df, col):

    t = df[col][df[col].replace({np.inf: np.nan}).replace(
        {-np.inf: np.nan}).notnull()]
#     t = (t - t.mean())/t.std()
    plt.figure()
    plt.hist(t)
    plt.title("normal_"+col)
    try:
        plt.figure()
        plt.hist(np.sqrt(t))
        plt.title("sqrt_"+col)
    except ValueError:
        pass
    try:
        plt.figure()
        plt.hist(np.sqrt(np.log(t)))
        plt.title("log_sqrt_"+col)
    except ValueError:
        pass
    try:
        plt.figure()
        plt.hist(np.log(np.log(t)))
        plt.title("log_"+col)
    except ValueError:
        pass
    try:
        plt.figure()
        plt.hist(np.power(np.exp(np.exp(t)), 4))
        plt.title("exp_"+col)
    except ValueError:
        pass


# for col in train_df.columns:
#     print(f"###### {col} #######")
#     if train_df[col].nunique() < 2:
#         continue
#     check_dist(train_df, col)
