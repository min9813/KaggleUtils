import time
import warnings
import optuna
from lightgbm import LGBMRegressor
from xgboost import XGBRegressor
from pandas.core.common import SettingWithCopyWarning
from sklearn.model_selection import KFold, StratifiedKFold

warnings.simplefilter(action='ignore', category=SettingWithCopyWarning)
warnings.simplefilter(action='ignore', category=FutureWarning)
FEATS_EXCLUDED = ['first_active_month', 'target', 'card_id', 'outliers',
                  'oof_preds', 'origin', ]


class OptunaObjective(object):

    def __init__(self, train_df, is_contain_categorical, sample_weight=None, num_folds=2):
        self.sample_weight = sample_weight
        self.train_df = train_df
        self.study = None
        self.num_folds = num_folds
        self.is_contain_categorical = is_contain_categorical
        self.debug = None
        self.trial_number = 0
        self.total_trial_number = 0
        self.start_time = None
        self.prev = None

    def outlier_pred_objective(self, trial):
        # def outlier_pred_objective(trial):
        drop_rate = trial.suggest_uniform('drop_rate', 0, 1.0)
        feature_fraction = trial.suggest_uniform('feature_fraction', 0, 1.0)
        learning_rate = trial.suggest_uniform('learning_rate', 0, 1.0)
        subsample = trial.suggest_uniform('subsample', 0.8, 1.0)
        num_leaves = trial.suggest_int('num_leaves', 5, 1000)
        n_estimators = trial.suggest_int('n_estimators', 100, 5000)
        min_data_in_leaf = trial.suggest_int('min_data_in_leaf', 5, 10000)
        min_child_samples = trial.suggest_int('min_child_samples', 5, 500)
        min_child_weight = trial.suggest_int('min_child_weight', 5, 500)
        boosting_types = trial.suggest_categorical(
            'boosting_type', ['gbdt', 'dart'])
        reg_alpha = trial.suggest_uniform("reg_alpha", 0, 20)
        reg_lambda = trial.suggest_uniform("reg_lambda", 0, 20)

        params = {
            "objective": "binary",
            "metric": "auc",
            "n_jobs": 6,
            'subsample_freq': 2,
            'subsample_for_bin': 200000,
            'max_cat_to_onehot': 4,
            'cat_l2': 10.0,
            'cat_smooth': 10.0,
            "drop_rate": drop_rate,
            "feature_fraction": feature_fraction,
            "learning_rate": learning_rate,
            "subsample": subsample,
            "num_leaves": num_leaves,
            "n_estimators": n_estimators,
            "min_data_in_leaf": min_data_in_leaf,
            "min_chihld_samples": min_child_samples,
            "min_child_weight": min_child_weight,
            "boosting_type": boosting_types,
            "reg_alpha": reg_alpha,
            "reg_lambda": reg_lambda
        }

        if self.debug:
            cv_score = predict_outlier_eval_mode(
                params, train_df, num_folds=self.num_folds, clf_name="lgb", weights=weights, stratified=True)
        else:
            cv_score = predict_outlier_eval_mode(
                params, train_df, num_folds=self.num_folds, clf_name="lgb", weights=weights, stratified=True)

        self.trial_number += 1
        print("### trial no {} finish; total:{}, progress:{:.2f}%".format(self.trial_number,
                                                                          self.total_trial_number, float(self.trial_number)/self.total_trial_number * 100))
        now = time.time()
        tmp_time = now - self.prev
        self.prev = now
        mean_iter_time = (now - self.start_time)/self.trial_number
        remain_time = mean_iter_time * \
            (self.total_trial_number - self.trial_number)

        print(
            f"##### this iteration spent {tmp_time}s, mean {mean_iter_time}s/iter, estimate remain time = {remain_time}s")
        return - cv_score

    def xgb_target_pred_objective(self, trial):
        # def outlier_pred_objective(trial):
        #         drop_rate = trial.suggest_loguniform('drop_rate', 0, 1.0)
        #         if self.is_contain_categorical:
        #             cat_l2 = trial.suggest_uniform("cat_l2",0,20)
        #         else:
        #             cat_l2 = 10

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
            'max_depth': trial.suggest_int("max_depth", 5, 15),
            'seed': 538,
            #             'gamma':trial.suggest_uniform("gamma",0,20)
            #             'colsample_bylevel': 0.9,
            'colsample_bytree': trial.suggest_uniform("colsample_bytree", 0.4, 1),
            'gamma': 0.0001,
            'learning_rate': trial.suggest_loguniform("learning_rate", 1e-3, 0.6),
            'max_bin': 400,
            #             'max_leaves': 47,
            'min_child_weight': trial.suggest_int('min_child_weight', 5, 500),
            'reg_alpha': trial.suggest_loguniform("reg_alpha", 1e-3, 20),
            'reg_lambda': trial.suggest_loguniform("reg_lambda", 1e-3, 20),
            'subsample': trial.suggest_uniform("subsample", 0.4, 1)}

        if self.debug:
            cv_score = predict_target_eval_mode(
                params, self.train_df, num_folds=self.num_folds, clf_name="xgb", stratified=False)
        else:
            cv_score = predict_target_eval_mode(
                params, self.train_df, num_folds=self.num_folds, clf_name="xgb", stratified=False)
        self.trial_number += 1
        print("### trial no {} finish; total:{}, progress:{:.2f}%".format(self.trial_number,
                                                                          self.total_trial_number, float(self.trial_number)/self.total_trial_number * 100))
        now = time.time()
        tmp_time = now - self.prev
        self.prev = now
        mean_iter_time = (now - self.start_time)/self.trial_number
        remain_time = mean_iter_time * \
            (self.total_trial_number - self.trial_number)

        print(
            f"##### this iteration spent {tmp_time}s, mean {mean_iter_time}s/iter, estimate remain time = {remain_time}s")
        return cv_score

    def run(self, debug=True, trials=500, target_value="target"):
        if debug:
            trials = 2
            self.num_folds = 2
        else:
            trials = trials
        self.total_trial_number = trials
        self.debug = True
        study = optuna.create_study()
        self.start_time = time.time()
        self.prev = self.start_time
        if target_value == "target":
            study.optimize(self.xgb_target_pred_objective, n_trials=trials)
        elif target_value == "outlier":
            study.optimize(self.outlier_pred_objective, n_trials=trials)
        else:
            raise NotImplementedError

        print('Number of finished trials: {}'.format(len(study.trials)))

        print('Best trial:')
        trial = study.best_trial

        print('  Value: {}'.format(trial.value))

        print('  Params: ')
        for key, value in trial.params.items():
            print('    {}: {}'.format(key, value))

        self.study = study


def predict_target_eval_mode(params, train_df, clf_name, stratified=False, num_folds=5, debug=False, fold_random_state=1989):

    if stratified:
        folds = StratifiedKFold(
            n_splits=num_folds, shuffle=True, random_state=fold_random_state)
    else:
        folds = KFold(n_splits=num_folds, shuffle=True,
                      random_state=fold_random_state)


#     oof_xgb = np.zeros(len(train_df))
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
    for fold_, (trn_idx, val_idx) in enumerate(folds.split(train_df[feats], train_df["target"])):
        trn_x, trn_y = train_df[feats].iloc[trn_idx], train_df["target"].iloc[trn_idx]
        val_x, val_y = train_df[feats].iloc[val_idx], train_df["target"].iloc[val_idx]
        if clf_name == "lgb":
            clf = LGBMRegressor(**params)
            clf.fit(
                trn_x, trn_y,
                eval_set=[(trn_x, trn_y), (val_x, val_y)],
                eval_metric=params["metric"],
                verbose=100,
                early_stopping_rounds=early_stopping_rounds,
                #                 sample_weight=trn_y.map(weights)
            )
        else:
            clf = XGBRegressor(**params)
            clf.fit(
                trn_x, trn_y,
                eval_set=[(trn_x, trn_y), (val_x, val_y)],
                #                 eval_metric=xgb_logloss,
                verbose=100,
                early_stopping_rounds=200,
                #                 sample_weight=trn_y.map(weights)
            )
#         if clf_name == "xgb":
#             oof_xgb[val_idx] = clf.predict_proba(val_x, ntree_limit=clf.best_ntree_limit)[:,1]
#         elif clf_name == "lgb":
#             oof_xgb[val_idx] = clf.predict_proba(val_x, num_iteration=clf.best_iteration_)[:,1]
        if clf_name == "xgb":
            score = clf.best_score
        elif clf_name == "lgb":
            score = clf.best_score_["valid_1"]["rmse"]
        print('no {}-fold loss: {:.6f}'.format(fold_ + 1,
                                               score))

        cv_score += score*1./num_folds

    return cv_score
