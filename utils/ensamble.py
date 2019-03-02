import numpy as np
import os
import pandas as pd
import warnings
import glob
from xgboost import XGBRegressor
from pandas.core.common import SettingWithCopyWarning
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import KFold, StratifiedKFold

warnings.simplefilter(action='ignore', category=SettingWithCopyWarning)
warnings.simplefilter(action='ignore', category=FutureWarning)


def rmse(y_true, y_pred):
    return np.sqrt(mean_squared_error(y_true, y_pred))


FEATS_EXCLUDED = ["target", "card_id"]

target_df = [
    "./subm_cv=3.646972_clf=lgb/",
    "./subm_cv=3.647059_clf=lgb/",
    "./non_outlier_probasubm_cv=1.555885_clf=lgb/"
]


def check_outlier_cand(cands, target, name=None):
    pred_tests = [glob.glob(cand + "subm_cv=*")[0] for cand in cands]
    oof_preds_train_paths = [
        glob.glob(cand + "train_feat*")[0] for cand in cands]
    oof_preds_train = []
    t_cols = []
    for path in oof_preds_train_paths:
        print(path)
        try:
            oof_preds_train.append(pd.read_csv(path)["oof_preds"])
            t_cols.append("oof_preds")
        except KeyError:
            oof_preds_train.append(pd.read_csv(
                path)["oof_preds_with_non_outlier"])
            t_cols.append("oof_preds_with_non_outlier")

#     oof_preds_train = [pd.read_csv(path)[["oof_preds_with_non_outlier"]] for path in oof_preds_train]
    for i in range(len(cands)-1):
        one = oof_preds_train[i]
        for j in range(i+1, len(cands)):
            two = oof_preds_train[j]
            print("check difference between {} {}".format(cands[i], cands[j]))
            if len(one) < 200000 or len(two) < 200000:
                raise ValueError
            mean_df = check_difference(one, two, target)
    if len(cands) < 3:
        save_path = name
#         print(mean_df)
        if name is None:
            raise NotImplementedError
        pd.DataFrame(mean_df).to_csv(save_path, index=False)
    pred_tests = [pd.read_csv(path)["target"] for path in pred_tests]
    for i in range(len(cands) - 1):
        one = pred_tests[i]
        for j in range(i+1, len(cands)):
            two = pred_tests[j]
            score = np.corrcoef(one, two)[0, 1]
            print(f"corr coef between {cands[i]} {cands[j]} = {score}")


def check_cand(cands, target):
    oof_preds_train = [glob.glob(cand + "train_feat*")[0] for cand in cands]
    oof_preds_train = [pd.read_csv(path)[["oof_preds"]]
                       for path in oof_preds_train]
    for i in range(len(cands)-1):
        one = oof_preds_train[i]
        for j in range(i+1, len(cands)):
            two = oof_preds_train[j]
            print("check difference between {} {}".format(cands[i], cands[j]))
            check_difference(one.oof_preds, two.oof_preds, target)


def check_difference(one_df, second_df, target, name=None):
    diff1 = one_df - target
    diff2 = second_df - target
    diff = pd.DataFrame({"diff1": diff1.values, "diff2": diff2.values})

    diff["same"] = diff["diff1"] * diff["diff2"] > 0
    print("different direction rate:", 1-diff.same.mean())

    mean_df = one_df*0.5 + second_df * 0.5
    corr_coef = np.corrcoef(one_df, second_df)[1, 0]
    if name is None or name == "oof_preds":
        print("one:", rmse(one_df, target))
        print("second:", rmse(second_df, target))

        score = rmse(mean_df, target)
        print("score:", score)
    print(corr_coef)

    return mean_df


def naive_concat_test(cands):
    assert len(cands) == 2
    pred_tests = [glob.glob(cand + "cv=*")[0] for cand in cands]
    pred_tests = [pd.read_csv(path) for path in pred_tests]
    ensamble_subm = pd.DataFrame(pred_tests[0]["card_id"])
    ensamble_subm["target"] = pred_tests[0]["target"] * \
        0.5 + pred_tests[1]["target"] * 0.5
    ensamble_subm.to_csv(
        "./non_out_cv=1.554_cv=1.556_oof_with_ensamble.csv", index=False)

# naive_concat_test(cands)


def get_concat_data(cands):
    train_org = pd.read_csv("../train.csv")[["card_id", "target"]]
    pred_tests = [glob.glob(cand + "subm_cv=*")[0] for cand in cands]
    oof_preds_train_paths = [
        glob.glob(cand + "train_feat*")[0] for cand in cands]
    train_df = None
    test_df = None
    for path_id, path in enumerate(oof_preds_train_paths):
        print(path)
        if train_df is None:
            try:
                train_df = pd.read_csv(path)[["oof_preds"]]
            except KeyError:
                train_df = pd.read_csv(path)[["oof_preds_with_non_outlier"]]
            train_df.columns = ["oof_preds_0"]
            train_df["card_id"] = train_org["card_id"]
        else:
            try:
                new_df = pd.read_csv(path)[["oof_preds"]]
            except KeyError:
                new_df = pd.read_csv(path)[["oof_preds_with_non_outlier"]]
            new_df.columns = [f"oof_preds_{path_id}"]
            new_df["card_id"] = train_org["card_id"]
#             print(train_df.head())
#             print(new_df.head())

            train_df = train_df.merge(new_df, on="card_id")
            print("train df shape:", train_df.shape)

    train_df["target"] = train_org.target
    for path_id, path in enumerate(pred_tests):
        print(path)
        if test_df is None:
            test_df = pd.read_csv(path)[["card_id", "target"]]
            test_df.columns = ["card_id", "oof_preds_0"]
        else:
            new_df = pd.read_csv(path)[["card_id", "target"]]
            new_df.columns = ["card_id", f"oof_preds_{path_id}"]
            test_df = test_df.merge(new_df, on="card_id", how="left")
#             print(test_df.head())
#             print(new_df.head())
            print("test df shape:", test_df.shape)
    return train_df, test_df


def stacked_kfold_xgboost(train, test, stratified=False, num_folds=5, debug=False, name="", need_result=False, fold_random_state=42):
    print("Starting XGBoost. Train shape: {}, test shape: {}".format(
        train.shape, test.shape))

    params = {
        'gpu_id': 0,
        # 'n_gpus': 2,
        'objective': 'reg:linear',
        'eval_metric': 'rmse',
        'silent': True,
        'booster': 'gbtree',
        'n_jobs': 6,
        'n_estimators': 20000,
        'tree_method': 'gpu_hist',
        'grow_policy': 'lossguide',
        'max_depth': 5,
        'seed': 538,
        'colsample_bylevel': 0.9,
        'colsample_bytree': 0.8,
        'gamma': 0.0001,
        'learning_rate': 0.006150886706231842,
        'max_bin': 128,
        'max_leaves': 47,
        'min_child_weight': 40,
        'reg_alpha': 10.0,
        'reg_lambda': 10.0,
        'subsample': 0.9}

    train_df = train.copy()
    test_df = test.copy()

    FOLDs = KFold(n_splits=num_folds, shuffle=True,
                  random_state=fold_random_state)

    oof_xgb = np.zeros(len(train_df))
    sub_preds = np.zeros(len(test_df))
    feats = [f for f in train_df.columns if f not in FEATS_EXCLUDED]
    feats = [f for f in feats if len(
        train_df[f].unique()) > 1 and "stacked" not in f]
    for col in sorted(feats):
        print(col)
    print("### use feature number:", len(feats))

    for fold_, (trn_idx, val_idx) in enumerate(FOLDs.split(train_df[feats], train_df["target"])):
        trn_x, trn_y = train_df[feats].iloc[trn_idx], train_df["target"].iloc[trn_idx]
        val_x, val_y = train_df[feats].iloc[val_idx], train_df["target"].iloc[val_idx]
        clf = XGBRegressor(**params)
        clf.fit(
            trn_x, trn_y,
            eval_set=[(trn_x, trn_y), (val_x, val_y)],
            verbose=100,
            early_stopping_rounds=100,
        )
        oof_xgb[val_idx] = clf.predict(val_x, ntree_limit=clf.best_ntree_limit)
        sub_preds += clf.predict(test_df[feats],
                                 ntree_limit=clf.best_ntree_limit) / FOLDs.n_splits

#         fold_importance_df = pd.DataFrame()
#         fold_importance_df["feature"] = feats
#         fold_importance_df["importance"] = clf.feature_importances_
#         fold_importance_df["fold"] =num_folds + 1
#         feature_importance_df = pd.concat([feature_importance_df, fold_importance_df], axis=0)
        score = rmse(val_y, oof_xgb[val_idx])

        print('no {}-fold loss: {:.6f}'.format(fold_ + 1,
                                               score))

    cv_score = rmse(oof_xgb, train_df["target"])
    print("cross validation score:{:.6f}".format(cv_score))

    if not debug:
        folder_name = "./stacking"
        if os.path.exists(folder_name) is False:
            os.mkdir(folder_name)

        # save submission file
        subm_path = os.path.join(
            folder_name, subm_cv="{:.6f}.csv".format(cv_score))
        train_df["stacked_oof_preds"] = oof_xgb
        train_df.to_csv(
            "{}/cv={:.6f}_oof_xgb.csv".format(folder_name, cv_score), index=False)
        test_df["target"] = sub_preds
        test_df[["card_id", "target"]].to_csv(
            subm_path, index=False)

    if need_result:
        return sub_preds, folder_name, subm_path


def stacked_kfold_sklearn(train, test, num_folds, clf, name, stratified=False, debug=False, need_result=False, fold_random_state=42):
    train_df = train.copy()
    test_df = test.copy()
    print("Starting LightGBM. Train shape: {}, test shape: {}".format(
        train_df.shape, test_df.shape))

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
    feats = [f for f in feats if len(
        train_df[f].unique()) > 1 and "stacked" not in f]

    # k-fold
    cv_score = 0
    for n_fold, (train_idx, valid_idx) in enumerate(folds.split(train_df[feats], train_df['target'])):
        trn = train_df[feats+["target"]].iloc[train_idx]
        val = train_df[feats+["target"]].iloc[valid_idx]

        print("start train fold {}".format(n_fold))
        # set data structure
#         clf = LGBMRegressor(**params)
        clf.fit(trn[feats], trn["target"])

        # params optimized by optuna

        oof_preds[valid_idx] = clf.predict(val[feats])
        sub_preds += clf.predict(test_df[feats]) / folds.n_splits

        score = rmse(val["target"], oof_preds[valid_idx])
        cv_score += score/folds.n_splits
        print('Fold %2d RMSE : %.6f' % (n_fold + 1, score))

    print("total cv score:{:.6f}".format(cv_score))
    if not debug:
        folder_name = "./stacking"
        if os.path.exists(folder_name) is False:
            os.mkdir(folder_name)

        # save submission file
        train_df["stacked_oof_preds"] = oof_preds
        train_df.to_csv(
            "{}/cv={:.6f}_oof_{}.csv".format(folder_name, cv_score, name), index=False)
        test_df["target"] = sub_preds
        test_df[["card_id", "target"]].to_csv(
            "{}/subm_cv={:.6f}.csv".format(folder_name, cv_score), index=False)

    if need_result:
        return sub_preds, folder_name,
