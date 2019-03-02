import numpy as np
import pandas as pd
import warnings
import os
import datetime
from lightgbm import LGBMClassifier
from xgboost import XGBClassifier
from pandas.core.common import SettingWithCopyWarning
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import KFold, StratifiedKFold
from util import save_status_intermediate, kfold_lightgbm

warnings.simplefilter(action='ignore', category=SettingWithCopyWarning)
warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.filterwarnings('ignore')


FEATS_EXCLUDED = ['first_active_month', 'target', 'card_id', 'outliers',
                  'oof_preds', 'origin', ]


def predict_outlier_eval_mode(params, train_df, clf_name, weights, stratified=False, num_folds=5, debug=False, fold_random_state=1989):

    FOLDs = StratifiedKFold(
        n_splits=num_folds, shuffle=True, random_state=fold_random_state)

    feats = [f for f in train_df.columns if f not in FEATS_EXCLUDED]
    if debug:
        early_stopping_rounds = 10
        feats = feats[:2]
    else:
        early_stopping_rounds = 200
#     print("Starting get outlier by {}. Train shape: {}, test shape: {}".format(clf_name, train_df[feats].shape, test_df[feats].shape))
#     w = train_df["outliers"].value_counts()
#     weights = {i : np.sum(w) / w[i] for i in w.index}
    cv_score = 0
    for fold_, (trn_idx, val_idx) in enumerate(FOLDs.split(train_df[feats], train_df["outliers"])):
        trn_x, trn_y = train_df[feats].iloc[trn_idx], train_df["outliers"].iloc[trn_idx]
        val_x, val_y = train_df[feats].iloc[val_idx], train_df["outliers"].iloc[val_idx]
        if clf_name == "lgb":
            clf = LGBMClassifier(**params)
            clf.fit(
                trn_x, trn_y,
                eval_set=[(trn_x, trn_y), (val_x, val_y)],
                eval_metric=params["metric"],
                verbose=100,
                early_stopping_rounds=early_stopping_rounds,
                #                 sample_weight=trn_y.map(weights)
            )
        else:
            clf = XGBClassifier(**params)
            clf.fit(
                trn_x, trn_y,
                eval_set=[(trn_x, trn_y), (val_x, val_y)],
                #                 eval_metric=xgb_logloss,
                verbose=100,
                early_stopping_rounds=200,
                #                 sample_weight=trn_y.map(weights)
            )

        score = clf.best_score_["valid_1"]["auc"]
        print('no {}-fold loss: {:.6f}'.format(fold_ + 1,
                                               score))
        cv_score += score/num_folds

    return cv_score


def kfold_lightgbm_outlier(train_df, test_df, num_folds, folder_name, stratified=False, debug=False, name="outlier_proba", need_result=False, fold_random_state=42):
    assert "outliers" in train_df.columns
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
    oof_preds = np.zeros(train_df.shape[0])
    sub_preds = np.zeros(test_df.shape[0])
    feats = [f for f in train_df.columns if f not in FEATS_EXCLUDED]
    feats = [f for f in feats if len(train_df[f].unique()) > 1]
    print("### use feature number:", len(feats))
    clfs = []
    w = train_df["outliers"].value_counts()
    weights = {i: np.sum(w) / w[i] for i in w.index}

    if debug:
        feats = feats[:3]

    # k-fold
    cv_score = 0
    for n_fold, (train_idx, valid_idx) in enumerate(folds.split(train_df[feats], train_df['target'])):
        trn_x, trn_y = train_df[feats].iloc[train_idx], train_df['outliers'].iloc[train_idx]
        val_x, val_y = train_df[feats].iloc[valid_idx], train_df['outliers'].iloc[valid_idx]
        params = {'objective': 'binary',
                  'boosting_type': 'gbdt',
                  'n_jobs': 4, 'max_depth': 7,
                  'subsample_freq': 2,
                  'subsample_for_bin': 200000,
                  'min_data_per_group': 100,
                  'num_iterations': 2000,
                  'max_cat_to_onehot': 4,
                  'cat_l2': 10.0,
                  'cat_smooth': 10.0,
                  'max_cat_threshold': 32,
                  'metric_freq': 10,
                  'verbosity': 100,
                  'metric': 'auc',
                  'colsample_bytree': 0.5,
                  'learning_rate': 0.0061033234451294376,
                  'min_child_samples': 80,
                  'min_child_weight': 100.0,
                  'min_split_gain': 1e-06,
                  'num_leaves': 47,
                  'reg_alpha': 10.0,
                  'n_estimators': 1000,
                  'reg_lambda': 10.0,
                  'subsample': 0.9}
        # set data structure
        clf = LGBMClassifier(**params)
        clf.fit(
            trn_x, trn_y,
            eval_set=[(trn_x, trn_y), (val_x, val_y)],
            eval_metric=params["metric"],
            early_stopping_rounds=200,
            verbose=100,
            sample_weight=trn_y.map(weights)
        )
        clfs.append(clf)

        # params optimized by optuna

        oof_preds[valid_idx] = clf.predict_proba(
            val_x, num_iteration=clf.best_iteration_)[:, 1]
        sub_preds += clf.predict_proba(test_df[feats], num_iteration=clf.best_iteration_)[
            :, 1] / folds.n_splits

#         fold_importance_df = pd.DataFrame()
#         fold_importance_df["feature"] = feats
#         fold_importance_df["importance"] = clf.feature_importances_
#         fold_importance_df["fold"] = n_fold + 1
#         feature_importance_df = pd.concat([feature_importance_df, fold_importance_df], axis=0)
        score = roc_auc_score(val_y, oof_preds[valid_idx])
        cv_score += score/folds.n_splits
        print('Fold %2d RMSE : %.6f' % (n_fold + 1, score))

    print("cross validation score:{:.6f}".format(cv_score))

#     if not debug:
    # save submission file
    train_df[name] = oof_preds
    folder_name, sub_preds = save_status_intermediate(train_df[list(set(
        feats+[name]))], test_df, sub_preds, clfs, "lgb", folder_name, feature_name=name)

#         save_to_log("lgb",num_folds, stratified, fold_random_state, folder_name, cv_score, params)
#     else:
#         display_importances(feature_importance_df)
    print("#### finsh !!! ####")
    if need_result:
        return sub_preds


def get_outlier_adjusting(train_df, test_df, threshould_dict, best_score_path, non_outlier_path=None, test_outlier_path=None, num_folds=5, outlier_num_folds=None, clf_name="lgb", debug=False):
    if threshould_dict["number"]:
        add_name = "number"
    else:
        add_name = "proba"
    if debug:
        num_folds = 2
    if non_outlier_path is None:
        if "outlier_proba" in train_df.columns:
            non_outlier_df = train_df[train_df["outliers"] == 0].drop(
                "outlier_proba", axis=1)
        else:
            non_outlier_df = train_df[train_df["outliers"] == 0]
        non_outlier_clfs, non_outlier_preds, folder_name, non_outlier_path, cv_score = kfold_lightgbm(
            non_outlier_df, test_df, num_folds, name="non_outlier_{}".format(add_name), need_result=True, debug=debug)
    else:
        print("read csv from non outlier data ...")
        folder_name = os.path.dirname(non_outlier_path)
        non_outlier_preds = pd.read_csv(non_outlier_path)
        non_outlier_df = pd.read_csv(folder_name+"/train_feat.csv")
        non_outlier_df.index = train_df[train_df.outliers == 0].index
        print("finish")
    if outlier_num_folds is None:
        outlier_num_folds = num_folds

    if test_outlier_path is None:
        if "outlier_proba" in test_df.columns:
            test_outlier_pred = test_df[["card_id", "outlier_proba"]]
        else:
            #             params, clf_class = make_model("lgb", method="clf")
            test_outlier_pred = kfold_lightgbm_outlier(
                train_df, test_df, folder_name=folder_name, num_folds=outlier_num_folds, need_result=True, debug=debug)

            test_outlier_path = os.path.join(
                folder_name, "test_outlier_pred.csv")
            test_outlier_pred.to_csv(test_outlier_path, index=False)
    else:
        print("read csv from previous predicted outlier data ...")

        test_outlier_pred = pd.read_csv(test_outlier_path)["outlier_proba"]
        print("finish")

    if threshould_dict["number"]:
        raise NotImplementedError
        print("get outlier top {}".format(threshould_dict["thres_number"]))
        test_outlier_id = test_outlier_pred.sort_values(
            by='outlier_proba', ascending=False).head(threshould_dict["thres_number"])['card_id']
    else:
        print("get outlier by probability > {}".format(
            threshould_dict["thres_proba"]))

        test_outlier_id = test_outlier_pred[test_outlier_pred["outlier_proba"]
                                            > threshould_dict["thres_proba"]]["card_id"]
        train_non_outlier_id = train_df.loc[train_df.outlier_proba <=
                                            threshould_dict["thres_proba"]].index
        outlier_num = len(test_outlier_id)
        print("outlier number {}, percent:{:.3f}".format(
            outlier_num, 100*outlier_num/len(test_outlier_pred)))
    best_score_df = pd.read_csv(best_score_path)
    most_likely_liers = best_score_df[best_score_df.card_id.isin(
        test_outlier_id)]
    least_likely_liers = non_outlier_df[non_outlier_df.index.isin(
        train_non_outlier_id)].index
    non_outlier_preds.loc[non_outlier_preds.card_id.isin(
        test_outlier_id), "target"] = most_likely_liers["target"]
    train_df["oof_preds_with_non_outlier"] = train_df["oof_preds"]
    train_df.loc[least_likely_liers,
                 "oof_preds_with_non_outlier"] = non_outlier_df.oof_preds
    train_df[["oof_preds_with_non_outlier"]].to_csv(
        folder_name + "/oof_preds_with_non_outlier.csv")

    best_cv_score = best_score_path.split("=")[1][:8]
    non_outlier_score = non_outlier_path.split("=")[1][:8]
    save_path = os.path.join(folder_name, "cv={}_with_{}_{}>{}.csv".format(
        best_cv_score, non_outlier_score, add_name, threshould_dict["thres_"+add_name]))
    if os.path.exists(save_path) is False:
        non_outlier_preds.to_csv(save_path, index=False)

    train_params = {
        "best_cv_path": [os.path.abspath(best_score_path)],
        "non_outlier_path": [os.path.abspath(non_outlier_path)],
        "subm_path": [os.path.abspath(save_path)],
        "folder_name": [os.path.abspath(folder_name)],
        "threshould_type": [add_name],
        "threshould": [threshould_dict["thres_"+add_name]],
        "non_outlier_num_folds": [num_folds],
        "outlier_num_folds": [outlier_num_folds],
        "non_outlier_cv_score": [float(non_outlier_score)],
        "best_cv_score": [float(best_cv_score)],
        "timestamp": [datetime.datetime.now().strftime('%Y-%m-%d-%H-%M-%S')],
    }
    fit_result_path = "fit_outlier_devision_result.csv"

    save_to_log_universal(fit_result_path, train_params)
    return train_df


def save_to_log_universal(path, train_params):

    fit_result_csv = pd.DataFrame(train_params)
    if os.path.exists(path):
        fit_result_csv.to_csv(path, header=False, mode="a", index=False)
    else:
        fit_result_csv.to_csv(path, header=True, mode="a", index=False)
