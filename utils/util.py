import datetime
import gc
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import seaborn as sns
import time
import warnings
import pickle
from lightgbm import LGBMRegressor
from xgboost import XGBRegressor
from numba import jit
from contextlib import contextmanager
from pandas.core.common import SettingWithCopyWarning
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import KFold, StratifiedKFold

warnings.simplefilter(action='ignore', category=SettingWithCopyWarning)
warnings.simplefilter(action='ignore', category=FutureWarning)
# sys.path.append("/home/minteiko/developer/project/fm/")

# import xdeepfm

sns.set(style='white', context='notebook', palette='deep')
warnings.filterwarnings('ignore')
sns.set_style('white')

FEATS_EXCLUDED = ['first_active_month', 'target', 'card_id', 'outliers', 'oof_preds', 'origin',
                  'hist_purchase_date_max', 'hist_purchase_date_min', 'hist_card_id_size',
                  'new_purchase_date_max', 'new_purchase_date_min', 'new_card_id_size',
                  'new_no_authorize_card_id_size', 'new_authorize_card_id_size',
                  'hist_no_authorize_card_id_size', 'hist_authorize_card_id_size',
                  'OOF_PRED', 'month_0', 'new_category_3_min_mean', 'new_authorized_flag_merchant_mean_std',
                  'new_authorized_flag_mean_mean', 'new_category_3_max_mean',
                  'hist_category_3_min_mean', 'hist_category_3_max_mean', 'new_authorized_flag_merchant_mean_mean',
                  'hist_no_authorize_purchase_date_max', 'hist_no_authorize_purchase_date_min',
                  'hist_authorize_purchase_date_max', 'hist_authorize_purchase_date_min',
                  'new_no_authorize_purchase_date_max', 'new_no_authorize_purchase_date_min',
                  'new_authorize_purchase_date_max', 'new_authorize_purchase_date_min',
                  'new_category_1_N_purchase_date_max', 'new_category_1_N_purchase_date_min',
                  'new_category_1_Y_purchase_date_max', 'new_category_1_Y_purchase_date_min',
                  'hist_category_1_N_purchase_date_max', 'hist_category_1_N_purchase_date_min',
                  'hist_category_1_Y_purchase_date_max', 'hist_category_1_Y_purchase_date_min']

DATE_BASE = datetime.datetime(2019, 1, 26)


@contextmanager
def timer(title):
    t0 = time.time()
    yield
    print("{} - done in {:.0f}s".format(title, time.time() - t0))

# rmse


def data_overview(data):
    print("shape:", data.shape)
    print("---------main statiscal value---------")
    print(data.describe())
    print("---percent of NAN---")
    print(data.isnull().mean())
    print("---length of unique value-----")
    for col in data.columns:
        length = len(data[col].unique())
        print("-----------------------------------------")
        print("{}: #{}".format(col, length))
        if length < 30:
            value_count = data[col].value_counts()
            print("----value count----")
            print(value_count)

    return data.head()


def data_plot(data):
    data_length = data.shape[0]

    for col in data.columns:
        length = len(data[col].unique())

        try:
            if length < 30:
                value_count = data.loc[data[col].isnull(
                ) is False, col].value_counts()

                plt.figure()
                plt.bar(data[col].unique(), value_count)
                plt.title(col)
            elif data[col].dtype != "object" and "id" not in col:
                plt.figure()
                data[col].hist(bins=int(np.sqrt(data_length)))
                plt.title(col)
                plt.figure()
                data[col].hist()
                plt.title(col+" large")
        except ValueError:
            print("something wrong in {}".format(col))
        except TypeError:
            print("something wrong in {}".format(col))


def rmse(y_true, y_pred):
    return np.sqrt(mean_squared_error(y_true, y_pred))

# One-hot encoding for categorical columns with get_dummies


def one_hot_encoder(df, nan_as_category=True):
    original_columns = df.columns.tolist()

    categorical_columns = list(filter(lambda c: c in ['object'], df.dtypes))
    df = pd.get_dummies(df, columns=categorical_columns,
                        dummy_na=nan_as_category)

    new_columns = list(filter(lambda c: c not in original_columns, df.columns))
    return df, new_columns

# Display/plot feature importance


def display_importances(feature_importance_df_, save_path=None):
    cols = feature_importance_df_[["feature", "importance"]].groupby(
        "feature").mean().sort_values(by="importance", ascending=False)
    if save_path is not None:
        cols.to_csv(save_path, index=True)
    cols = cols[:40].index
    best_features = feature_importance_df_.loc[feature_importance_df_.feature.isin(
        cols)]

    plt.figure(figsize=(8, 10))
    sns.barplot(x="importance", y="feature", data=best_features.sort_values(
        by="importance", ascending=False))
    plt.title('LightGBM Features (avg over folds)')
    plt.tight_layout()
    plt.savefig('lgbm_importances.png')

# reduce memory


def reduce_mem_usage(df, verbose=True):
    numerics = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']
    start_mem = df.memory_usage().sum() / 1024**2
    for col in df.columns:
        col_type = df[col].dtypes
        if col_type in numerics:
            c_min = df[col].min()
            c_max = df[col].max()
            if str(col_type)[:3] == 'int':
                if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                    df[col] = df[col].astype(np.int8)
                elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                    df[col] = df[col].astype(np.int16)
                elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                    df[col] = df[col].astype(np.int32)
                elif c_min > np.iinfo(np.int64).min and c_max < np.iinfo(np.int64).max:
                    df[col] = df[col].astype(np.int64)
            else:
                if c_min > np.finfo(np.float16).min and c_max < np.finfo(np.float16).max:
                    df[col] = df[col].astype(np.float16)
                elif c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
                    df[col] = df[col].astype(np.float32)
                else:
                    df[col] = df[col].astype(np.float64)

    end_mem = df.memory_usage().sum() / 1024**2
    print('Memory usage after optimization is: {:.2f} MB'.format(end_mem))
    print('Decreased by {:.1f}%'.format(
        100 * (start_mem - end_mem) / start_mem))

    return df


def process_date(df):
    df['purchase_date'] = pd.to_datetime(df['purchase_date'])
    df['month'] = df['purchase_date'].dt.month
    df['day'] = df['purchase_date'].dt.day
    df['hour'] = df['purchase_date'].dt.hour
    df['weekofyear'] = df['purchase_date'].dt.weekofyear
    df['weekday'] = df['purchase_date'].dt.weekday
    df['weekend'] = (df['purchase_date'].dt.weekday >= 5).astype(int)
    return df


def dist_holiday(df, col_name, date_holiday, date_ref, period=100):
    df[col_name] = np.maximum(np.minimum(
        (pd.to_datetime(date_holiday) - df[date_ref]).dt.days, period), 0)

# preprocessing train & test


def process_main_df(df, embed_mode="numerical"):

    # datetime features
    df['quarter'] = df['first_active_month'].dt.quarter
    df['elapsed_time'] = (DATE_BASE - df['first_active_month']).dt.days

    if embed_mode == "numerical":
        feature_cols = ['feature_1', 'feature_2', 'feature_3']
    elif embed_mode == "onehot":
        feature_cols = [col for col in df.columns if "feature" in col]

    for f in feature_cols:
        df['days_' + f] = df['elapsed_time'] * df[f]
        df['days_' + f + '_ratio'] = df[f] / df['elapsed_time']

    # one hot encoding
    df, cols = one_hot_encoder(df, nan_as_category=False)

    if embed_mode == "numerical":
        df_feats = df.reindex(columns=feature_cols)
        df['features_sum'] = df_feats.sum(axis=1)
        df['features_mean'] = df_feats.mean(axis=1)
        df['features_max'] = df_feats.max(axis=1)
        df['features_min'] = df_feats.min(axis=1)
        df['features_var'] = df_feats.std(axis=1)
        df['features_prod'] = df_feats.product(axis=1)
    elif embed_mode == "onehot":
        pass

    return df


# preprocessing train & test
def train_test(num_rows=None, embed_mode="numerical"):

    def read_csv(filename):
        df = pd.read_csv(
            filename, index_col=['card_id'], parse_dates=['first_active_month'], nrows=num_rows)
        return df

    # load csv
    train_df = read_csv('../train.csv')
    test_df = read_csv('../test.csv')
    print("samples: train {}, test: {}".format(train_df.shape, test_df.shape))

    # outlier
    train_df['outliers'] = 0
    train_df.loc[train_df['target'] < -30., 'outliers'] = 1
    feature_cols = ['feature_1', 'feature_2', 'feature_3']

    if embed_mode == "numerical":
        train_df = reduce_mem_usage(process_main_df(train_df))
        test_df = reduce_mem_usage(process_main_df(test_df))
        for f in feature_cols:
            order_label = train_df.groupby([f])['outliers'].mean()
            train_df[f] = train_df[f].map(order_label)
            test_df[f] = test_df[f].map(order_label)
    elif embed_mode == "onehot":
        train_df = pd.get_dummies(train_df, columns=feature_cols)
        test_df = pd.get_dummies(test_df, columns=feature_cols)
        train_df = reduce_mem_usage(process_main_df(train_df, embed_mode))
        test_df = reduce_mem_usage(process_main_df(test_df, embed_mode))

    return train_df, test_df

# preprocessing historical transactions


def mode(x):
    return x.value_counts().index[0]


def historical_transactions(num_rows=None, merchant_df=None, embed_mode="numerical"):
    """
    preprocessing historical transactions
    """
    na_dict = {
        'category_2': 1.,
        'category_3': 'A',
        'merchant_id': 'M_ID_00a6ca8a8a',
    }

    holidays = [
        ('Christmas_Day_2017', '2017-12-25'),  # Christmas: December 25 2017
        ('Mothers_Day_2017', '2017-06-04'),  # Mothers Day: May 14 2017
        ('fathers_day_2017', '2017-08-13'),  # fathers day: August 13 2017
        ('Children_day_2017', '2017-10-12'),  # Childrens day: October 12 2017
        # Valentine's Day : 12th June, 2017
        ('Valentine_Day_2017', '2017-06-12'),
        # Black Friday: 24th November 2017
        ('Black_Friday_2017', '2017-11-24'),
        ('Mothers_Day_2018', '2018-05-13'),
    ]

    # agg
    aggs = dict()
    col_unique = ['subsector_id', 'merchant_id', 'merchant_category_id']
    aggs.update({col: ['nunique'] for col in col_unique})

    col_seas = ['month', 'hour', 'weekofyear', 'weekday', 'day']
    aggs.update({col: ['nunique', 'mean', 'min', 'max'] for col in col_seas})

    aggs_specific = {
        'purchase_amount': ['sum', 'max', 'min', 'mean', 'var', 'skew'],
        'installments': ['sum', 'max', 'mean', 'var', 'skew'],
        'purchase_date': ['max', 'min'],
        'month_lag': ['max', 'min', 'mean', 'var', 'skew'],
        'month_diff': ['max', 'min', 'mean', 'var', 'skew'],
        'authorized_flag': ['mean'],
        'weekend': ['mean'],  # overwrite
        'weekday': ['mean'],  # overwrite
        'day': ['nunique', 'mean', 'min'],  # overwrite
        'category_1': ['mean'],
        'category_2': ['mean'],
        'card_id': ['size', 'count'],
        'price': ['sum', 'mean', 'max', 'min', 'var'],
        'Christmas_Day_2017': ['mean', 'sum'],
        'Mothers_Day_2017': ['mean', 'sum'],
        'fathers_day_2017': ['mean', 'sum'],
        'Children_day_2017': ['mean', 'sum'],
        'Valentine_Day_2017': ['mean', 'sum'],
        'Black_Friday_2017': ['mean', 'sum'],
        'Mothers_Day_2018': ['mean', 'sum'],
        'duration': ['mean', 'min', 'max', 'var', 'skew'],
        'amount_month_ratio': ['mean', 'min', 'max', 'var', 'skew'],
        'amount_month_lag_ratio': ['mean', 'min', 'max', 'var', 'skew'],

    }
    aggs.update(aggs_specific)

#     authorize_aggs = {
#         'purchase_amount': ['sum', 'max', 'min', 'mean', 'var', 'skew'],
#         'installments': ['sum', 'max', 'mean', 'var', 'skew'],
#         'purchase_date': ['max', 'min'],
#         'month_lag': ['max', 'min', 'mean', 'var', 'skew'],
#         'month_diff': ['max', 'min', 'mean', 'var', 'skew'],
# #         'authorized_flag': ['mean'],
#         'weekend': ['mean'], # overwrite
#         'weekday': ['mean'], # overwrite
#         'day': ['nunique', 'mean', 'min'], # overwrite
#         'category_1': ['mean'],
#         'category_2': ['mean'],
#         'card_id': ['size', 'count'],
#         'price': ['sum', 'mean', 'max', 'min', 'var'],
# #         'Christmas_Day_2017': ['mean', 'sum'],
# #         'Mothers_Day_2017': ['mean', 'sum'],
# #         'fathers_day_2017': ['mean', 'sum'],
# #         'Children_day_2017': ['mean', 'sum'],
# #         'Valentine_Day_2017': ['mean', 'sum'],
# #         'Black_Friday_2017': ['mean', 'sum'],
# #         'Mothers_Day_2018': ['mean', 'sum'],
#         'duration': ['mean', 'min', 'max', 'var', 'skew'],
#         'amount_month_ratio': ['mean', 'min', 'max', 'var', 'skew'],

#     }
#     authorize_aggs.update({col: ['nunique'] for col in col_unique})
    # starting to process
    # load csv
    df = pd.read_csv('../historical_transactions.csv', nrows=num_rows)
    print('read historical_transactions {}'.format(df.shape))

    # fillna
    df.fillna(na_dict, inplace=True)
    df['installments'].replace({
        -1: np.nan, 999: np.nan}, inplace=True)
    # trim
    df['purchase_amount'] = df['purchase_amount'].map(lambda x: min(x, 0.8))
    df, aggs = process_date_before_agg(
        df, aggs, holidays, embed_mode=embed_mode)
    # trim
#     df['purchase_amount'] = df['purchase_amount'].apply(lambda x: min(x, 0.8))

#     # Y/N to 1/0
#     for col in ['category_2', 'category_3','authorized_flag']:
#         df['purchase_'+col+'_mean'] = df.groupby([col])['purchase_amount'].transform('mean')
#         df['purchase_'+col+'_min'] = df.groupby([col])['purchase_amount'].transform('min')
#         df['purchase_'+col+'_max'] = df.groupby([col])['purchase_amount'].transform('max')
#         df['purchase_'+col+'_sum'] = df.groupby([col])['purchase_amount'].transform('sum')
#         aggs['purchase_'+col + '_mean'] = ['mean']
#     df['authorized_flag'] = df['authorized_flag'].replace({'Y': 1, 'N': 0}).astype(int).astype(np.int16)
#     df['category_1'] = df['category_1'].replace({'Y': 1, 'N': 0}).astype(int).astype(np.int16)
#     if embed_mode == "numerical":
#         df['category_3'] = df['category_3'].replace({'A': 0, 'B': 1, 'C': 2}).astype(int).astype(np.int16)
#         aggs["category_3"] = ["mean"]
#     elif embed_mode == "onehot":
#         df = pd.get_dummies(df, columns=["category_3"])
#         for col in df.columns:
#             if "category_3" in col:
#                 aggs[col] = ["mean"]

#     # additional features
#     df['price'] = df['purchase_amount'] / df['installments']

#     # datetime features
#     df = process_date(df)

#     # holidays
#     for d_name, d_day in holidays:
#         dist_holiday(df, d_name, d_day, 'purchase_date')

#     df['month_diff'] = (DATE_BASE - df['purchase_date']).dt.days // 30
#     df['month_diff'] += df['month_lag']

#     # additional features
#     df['duration'] = df['purchase_amount'] * df['month_diff']
#     df['amount_month_ratio'] = df['purchase_amount'] / df['month_diff']
#     df['amount_month_lag_ratio'] = df["purchase_amount"] / (df['month_lag'] - 1)

    # reduce memory usage
    df = reduce_mem_usage(df)
#     authorize_aggs = {
#         'purchase_amount': ['sum', 'max', 'min', 'mean', 'var', 'skew'],
#         'installments': ['sum', 'max', 'mean', 'var', 'skew'],
#         'purchase_date': ['max', 'min'],
#         'month_lag': ['max', 'min', 'mean', 'var', 'skew'],
#         'month_diff': ['max', 'min', 'mean', 'var', 'skew'],
# #         'authorized_flag': ['mean'],
#         'weekend': ['mean'], # overwrite
#         'weekday': ['mean'], # overwrite
#         'day': ['nunique', 'mean', 'min'], # overwrite
#         'category_1': ['mean'],
#         'category_2': ['mean'],
#         'category_3':['mean'],
#         'card_id': ['size', 'count'],
#         'price': ['sum', 'mean', 'max', 'min', 'var'],
# #         'Christmas_Day_2017': ['mean', 'sum'],
# #         'Mothers_Day_2017': ['mean', 'sum'],
# #         'fathers_day_2017': ['mean', 'sum'],
# #         'Children_day_2017': ['mean', 'sum'],
# #         'Valentine_Day_2017': ['mean', 'sum'],
# #         'Black_Friday_2017': ['mean', 'sum'],
# #         'Mothers_Day_2018': ['mean', 'sum'],
#         'duration': ['mean', 'min', 'max', 'var', 'skew'],
#         'amount_month_ratio': ['mean', 'min', 'max', 'var', 'skew'],
#     }
#     no_authorize_df = df.loc[df.authorized_flag==0]
#     no_authorize_df = no_authorize_df.groupby("card_id").agg(authorize_aggs)
#     print("finish no authorize df agg")
#     no_authorize_df.columns = ["no_authorize_"+col[0]+"_"+col[1] for col in no_authorize_df.columns]
#     no_authorize_df = reduce_mem_usage(no_authorize_df)


#     authorize_df = df.loc[df.authorized_flag==1]
#     authorize_df = authorize_df.groupby("card_id").agg(authorize_aggs)
#     print("finish authorize df agg")
#     authorize_df.columns = ["authorize_"+col[0]+"_"+col[1] for col in authorize_df.columns]
#     authorize_df = reduce_mem_usage(authorize_df)

#     no_authorize_df = df.loc[df.category_1==0]
#     no_authorize_df = no_authorize_df.groupby("card_id").agg(authorize_aggs)
#     print("finish no authorize df agg")
#     no_authorize_df.columns = ["category_1_N_"+col[0]+"_"+col[1] for col in no_authorize_df.columns]
#     no_authorize_df = reduce_mem_usage(no_authorize_df)


#     authorize_df = df.loc[df.authorized_flag==1]
#     authorize_df = authorize_df.groupby("card_id").agg(authorize_aggs)
#     print("finish authorize df agg")
#     authorize_df.columns = ["category_1_Y_"+col[0]+"_"+col[1] for col in authorize_df.columns]
#     authorize_df = reduce_mem_usage(authorize_df)

    if merchant_df is not None:
        for m_col in merchant_df.columns:
            if "merchant_id" == m_col:
                continue
            aggs[m_col] = ["mean", "std"]
        df = df.merge(merchant_df, on="merchant_id", how="left")

    print("finish category embedding")

    cm_g, cm_agg_df = agg_by_merchant_card(df)

    print("finish card_merchant aggregation")

    df = df.reset_index().groupby('card_id').agg(aggs)
    print("finish card  aggregation")

    # change column name
    df.columns = pd.Index([e[0] + "_" + e[1] for e in df.columns.tolist()])

    df = df.merge(cm_agg_df, left_index=True, right_index=True)
    del cm_agg_df
    gc.collect()
#     df = df.merge(no_authorize_df, left_index=True, right_index=True)
#     del no_authorize_df
#     gc.collect()
#     df = df.merge(authorize_df, left_index=True, right_index=True)
#     del authorize_df
#     gc.collect()

#     print(df.columns)

    df = process_after_agg(df)
#     df['hist_CLV'] = df['hist_card_id_count'] * df['hist_purchase_amount_sum'] / df['hist_month_diff_mean']

#     df['hist_purchase_date_diff'] = (df['hist_purchase_date_max'] - df['hist_purchase_date_min']).dt.days
#     df['hist_purchase_date_average'] = df['hist_purchase_date_diff'] / df['hist_card_id_size']
#     df['hist_purchase_date_uptonow'] = (DATE_BASE - df['hist_purchase_date_max']).dt.days
#     df['hist_purchase_date_uptomin'] = (DATE_BASE - df['hist_purchase_date_min']).dt.days
#     for col in ["","category_1_N_","category_1_Y_"]:
#         df['hist_{}purchase_date_diff'.format(col)] = (df['hist_{}purchase_date_max'.format(col)] - df['hist_{}purchase_date_min'.format(col)]).dt.days
#         df['hist_{}purchase_date_average'.format(col)] = df['hist_{}purchase_date_diff'.format(col)] / df['hist_{}card_id_size'.format(col)]
#         df['hist_{}purchase_date_uptonow'.format(col)] = (DATE_BASE - df['hist_{}purchase_date_max'.format(col)]).dt.days
#         df['hist_{}purchase_date_uptomin'.format(col)] = (DATE_BASE - df['hist_{}purchase_date_min'.format(col)]).dt.days

    df.columns = ['hist_' + c for c in df.columns]

    # reduce memory usage
    df = reduce_mem_usage(df)

    return df, cm_g


# preprocessing new_merchant_transactions
def new_merchant_transactions(num_rows=None, merchant_df=None, embed_mode="numerical"):
    """
    preprocessing new_merchant_transactions
    """
    na_dict = {
        'category_2': 1.,
        'category_3': 'A',
        'merchant_id': 'M_ID_00a6ca8a8a',
    }

    holidays = [
        ('Christmas_Day_2017', '2017-12-25'),  # Christmas: December 25 2017
        # ('Mothers_Day_2017', '2017-06-04'),  # Mothers Day: May 14 2017
        # ('fathers_day_2017', '2017-08-13'),  # fathers day: August 13 2017
        ('Children_day_2017', '2017-10-12'),  # Childrens day: October 12 2017
        # ('Valentine_Day_2017', '2017-06-12'),  # Valentine's Day : 12th June, 2017
        # Black Friday: 24th November 2017
        ('Black_Friday_2017', '2017-11-24'),
        ('Mothers_Day_2018', '2018-05-13'),
    ]

    aggs = dict()
    col_unique = ['subsector_id', 'merchant_id', 'merchant_category_id']
    aggs.update({col: ['nunique'] for col in col_unique})

    col_seas = ['month', 'hour', 'weekofyear', 'weekday', 'day']
    aggs.update({col: ['nunique', 'mean', 'min', 'max'] for col in col_seas})

    aggs_specific = {
        'purchase_amount': ['sum', 'max', 'min', 'mean', 'var', 'skew'],
        'installments': ['sum', 'max', 'mean', 'var', 'skew'],
        'purchase_date': ['max', 'min'],
        'month_lag': ['max', 'min', 'mean', 'var', 'skew'],
        'month_diff': ['mean', 'var', 'skew'],
        'weekend': ['mean'],
        'month': ['mean', 'min', 'max'],
        'weekday': ['mean', 'min', 'max'],
        'category_1': ['mean'],
        'category_2': ['mean'],
        'card_id': ['size', 'count'],
        'price': ['mean', 'max', 'min', 'var'],
        'Christmas_Day_2017': ['mean', 'sum'],
        'Children_day_2017': ['mean', 'sum'],
        'Black_Friday_2017': ['mean', 'sum'],
        'Mothers_Day_2018': ['mean', 'sum'],
        'duration': ['mean', 'min', 'max', 'var', 'skew'],
        'amount_month_ratio': ['mean', 'min', 'max', 'var', 'skew'],
        'amount_month_lag_ratio': ['mean', 'min', 'max', 'var', 'skew'],

    }
    aggs.update(aggs_specific)

    # load csv
    df = pd.read_csv('../new_merchant_transactions.csv', nrows=num_rows)
    print('read new_merchant_transactions {}'.format(df.shape))

    # fillna
    df.fillna(na_dict, inplace=True)
    df['installments'].replace({
        -1: np.nan, 999: np.nan}, inplace=True)

    # trim
    df['purchase_amount'] = df['purchase_amount'].apply(lambda x: min(x, 0.8))

    # Y/N to 1/0
#     for col in ['category_2', 'category_3','authorized_flag']:
#         df["purchase_"+col+'_mean'] = df.groupby([col])['purchase_amount'].transform('mean')
#         df["purchase_"+col+'_min'] = df.groupby([col])['purchase_amount'].transform('min')
#         df["purchase_"+col+'_max'] = df.groupby([col])['purchase_amount'].transform('max')
#         df["purchase_"+col+'_sum'] = df.groupby([col])['purchase_amount'].transform('sum')
#         aggs["purchase_"+col + '_mean'] = ['mean']
#     df['authorized_flag'] = df['authorized_flag'].replace({'Y': 1, 'N': 0}).astype(int).astype(np.int16)
#     df['category_1'] = df['category_1'].replace({'Y': 1, 'N': 0}).astype(int).astype(np.int16)
#     if embed_mode == "numerical":
#         df['category_3'] = df['category_3'].replace({'A': 0, 'B': 1, 'C': 2}).astype(int).astype(np.int16)
#         aggs["category_3"] = ["mean"]
#     elif embed_mode == "onehot":
#         df = pd.get_dummies(df, columns=["category_3"])
#         for col in df.columns:
#             if "category_3" in col:
#                 aggs[col] = ["mean"]

#     # additional features
#     df['price'] = df['purchase_amount'] / df['installments']

#     # datetime features
#     df = process_date(df)
#     for d_name, d_day in holidays:
#         dist_holiday(df, d_name, d_day, 'purchase_date')

#     df['month_diff'] = (DATE_BASE - df['purchase_date']).dt.days // 30
#     df['month_diff'] += df['month_lag']

#     # additional features
#     df['duration'] = df['purchase_amount'] * df['month_diff']
#     df['amount_month_ratio'] = df['purchase_amount'] / df['month_diff']
#     df['amount_month_lag_ratio'] = df["purchase_amount"] / df['month_lag']
    df, aggs = process_date_before_agg(
        df, aggs, holidays, embed_mode=embed_mode, name="new")

#     authorize_aggs = {
#         'purchase_amount': ['sum', 'max', 'min', 'mean', 'var', 'skew'],
#         'installments': ['sum', 'max', 'mean', 'var', 'skew'],
#         'purchase_date': ['max', 'min'],
#         'month_lag': ['max', 'min', 'mean', 'var', 'skew'],
#         'month_diff': ['max', 'min', 'mean', 'var', 'skew'],
# #         'authorized_flag': ['mean'],
#         'weekend': ['mean'], # overwrite
#         'weekday': ['mean'], # overwrite
#         'day': ['nunique', 'mean', 'min'], # overwrite
#         'category_1': ['mean'],
#         'category_2': ['mean'],
#         'category_3':['mean'],
#         'card_id': ['size', 'count'],
#         'price': ['sum', 'mean', 'max', 'min', 'var'],
# #         'Christmas_Day_2017': ['mean', 'sum'],
# #         'Mothers_Day_2017': ['mean', 'sum'],
# #         'fathers_day_2017': ['mean', 'sum'],
# #         'Children_day_2017': ['mean', 'sum'],
# #         'Valentine_Day_2017': ['mean', 'sum'],
# #         'Black_Friday_2017': ['mean', 'sum'],
# #         'Mothers_Day_2018': ['mean', 'sum'],
#         'duration': ['mean', 'min', 'max', 'var', 'skew'],
#         'amount_month_ratio': ['mean', 'min', 'max', 'var', 'skew'],
#     }
#     authorize_aggs.update({col: ['nunique'] for col in col_unique})

#     no_authorize_df = df.loc[df.category_1==0]
#     no_authorize_df = no_authorize_df.groupby("card_id").agg(authorize_aggs)
#     print("finish no authorize df agg")
#     no_authorize_df.columns = ["category_1_N_"+col[0]+"_"+col[1] for col in no_authorize_df.columns]
#     no_authorize_df = reduce_mem_usage(no_authorize_df)


#     authorize_df = df.loc[df.authorized_flag==1]
#     authorize_df = authorize_df.groupby("card_id").agg(authorize_aggs)
#     print("finish authorize df agg")
#     authorize_df.columns = ["category_1_Y_"+col[0]+"_"+col[1] for col in authorize_df.columns]
#     authorize_df = reduce_mem_usage(authorize_df)
    # reduce memory usage
    df = reduce_mem_usage(df)

    if merchant_df is not None:
        for m_col in merchant_df.columns:
            if "merchant_id" == m_col:
                continue
            aggs[m_col] = ["mean", "std"]
        df = df.merge(merchant_df, on="merchant_id", how="left")

    cm_g, cm_agg_df = agg_by_merchant_card(df)

    print("finish category agg")

    df = df.reset_index().groupby('card_id').agg(aggs)
    print("finish card agg")

    # change column name
    df.columns = pd.Index([e[0] + "_" + e[1] for e in df.columns.tolist()])

    df = df.merge(cm_agg_df, left_index=True, right_index=True)
    del cm_agg_df
    gc.collect()
#     df = df.merge(no_authorize_df, left_index=True, right_index=True)
#     del no_authorize_df
#     gc.collect()
#     df = df.merge(authorize_df, left_index=True, right_index=True)
#     del authorize_df
#     gc.collect()
    df = process_after_agg(df)

    df.columns = ['new_' + c for c in df.columns]

#     df['new_CLV'] = df['new_card_id_count'] * df['new_purchase_amount_sum'] / df['new_month_diff_mean']

#     df['new_purchase_date_diff'] = (df['new_purchase_date_max'] - df['new_purchase_date_min']).dt.days
#     df['new_purchase_date_average'] = df['new_purchase_date_diff'] / df['new_card_id_size']
#     df['new_purchase_date_uptonow'] = (DATE_BASE - df['new_purchase_date_max']).dt.days
#     df['new_purchase_date_uptomin'] = (DATE_BASE - df['new_purchase_date_min']).dt.days
#     for col in ["","category_1_N_","category_1_Y_"]:
#         df['new_{}purchase_date_diff'.format(col)] = (df['new_{}purchase_date_max'.format(col)] - df['new_{}purchase_date_min'.format(col)]).dt.days
#         df['new_{}purchase_date_average'.format(col)] = df['new_{}purchase_date_diff'.format(col)] / df['new_{}card_id_size'.format(col)]
#         df['new_{}purchase_date_uptonow'.format(col)] = (DATE_BASE - df['new_{}purchase_date_max'.format(col)]).dt.days
#         df['new_{}purchase_date_uptomin'.format(col)] = (DATE_BASE - df['new_{}purchase_date_min'.format(col)]).dt.days

    # reduce memory usage
    df = reduce_mem_usage(df)

    return df, cm_g


def additional_features_nojit(df):
    df['hist_first_buy'] = (df['hist_purchase_date_min'] -
                            df['first_active_month']).dt.days
    df['hist_last_buy'] = (df['hist_purchase_date_max'] -
                           df['first_active_month']).dt.days

    df['new_first_buy'] = (df['new_purchase_date_min'] -
                           df['first_active_month']).dt.days
    df['new_last_buy'] = (df['new_purchase_date_max'] -
                          df['first_active_month']).dt.days

    return df
# additional features


@jit
def additional_features(df):

    df['hist_first_buy'] = (pd.to_datetime(
        df['hist_purchase_date_min']*1e9) - df['first_active_month']).dt.days
    df['hist_last_buy'] = (pd.to_datetime(
        df['hist_purchase_date_max']*1e9) - df['first_active_month']).dt.days

    df['new_first_buy'] = (pd.to_datetime(
        df['new_purchase_date_min']*1e9) - df['first_active_month']).dt.days
    df['new_last_buy'] = (pd.to_datetime(
        df['new_purchase_date_max']*1e9) - df['first_active_month']).dt.days

#     date_features = [
#         'hist_purchase_date_max', 'hist_purchase_date_min', 'new_purchase_date_max', 'new_purchase_date_min']
#     for f in date_features:
#         df[f] = df[f].astype(np.int64) * 1e-9

    #
    df['card_id_total'] = df['new_card_id_size'] + df['hist_card_id_size']
    df['card_id_cnt_total'] = df['new_card_id_count'] + df['hist_card_id_count']
    df['card_id_cnt_ratio'] = df['new_card_id_count'] / df['hist_card_id_count']

    df['purchase_amount_total'] = df['new_purchase_amount_sum'] + \
        df['hist_purchase_amount_sum']
    df['purchase_amount_mean'] = df['new_purchase_amount_mean'] + \
        df['hist_purchase_amount_mean']
    df['purchase_amount_max'] = df['new_purchase_amount_max'] + \
        df['hist_purchase_amount_max']
    df['purchase_amount_min'] = df['new_purchase_amount_min'] + \
        df['hist_purchase_amount_min']
    df['purchase_amount_ratio'] = df['new_purchase_amount_sum'] / \
        df['hist_purchase_amount_sum']

    df['installments_total'] = df['new_installments_sum'] + \
        df['hist_installments_sum']
    df['installments_mean'] = df['new_installments_mean'] + \
        df['hist_installments_mean']
    df['installments_max'] = df['new_installments_max'] + \
        df['hist_installments_max']
    df['installments_ratio'] = df['new_installments_sum'] / \
        df['hist_installments_sum']

    df['price_total'] = df['purchase_amount_total'] / df['installments_total']
    df['price_mean'] = df['purchase_amount_mean'] / df['installments_mean']
    df['price_max'] = df['purchase_amount_max'] / df['installments_max']

    #
    df['month_diff_mean'] = df['new_month_diff_mean'] + df['hist_month_diff_mean']
    df['month_diff_ratio'] = df['new_month_diff_mean'] / \
        df['hist_month_diff_mean']

    df['month_lag_mean'] = df['new_month_lag_mean'] + df['hist_month_lag_mean']
    df['month_lag_max'] = df['new_month_lag_max'] + df['hist_month_lag_max']
    df['month_lag_min'] = df['new_month_lag_min'] + df['hist_month_lag_min']
    df['category_1_mean'] = df['new_category_1_mean'] + df['hist_category_1_mean']

    df['duration_mean'] = df['new_duration_mean'] + df['hist_duration_mean']
    df['duration_min'] = df['new_duration_min'] + df['hist_duration_min']
    df['duration_max'] = df['new_duration_max'] + df['hist_duration_max']

    df['amount_month_ratio_mean'] = df['new_amount_month_ratio_mean'] + \
        df['hist_amount_month_ratio_mean']
    df['amount_month_ratio_min'] = df['new_amount_month_ratio_min'] + \
        df['hist_amount_month_ratio_min']
    df['amount_month_ratio_max'] = df['new_amount_month_ratio_max'] + \
        df['hist_amount_month_ratio_max']

    df['CLV_ratio'] = df['new_CLV'] / df['hist_CLV']
    df['CLV_sq'] = df['new_CLV'] * df['hist_CLV']

    df = reduce_mem_usage(df)
#     date_feature = [col for col in df.columns if "purchase_date" in col and ("max" in col) or ("min" in col)]
#     for f in date_feature:
#         df[f] = df[f].astype(np.int64) * 1e-9
    return df

# LightGBM GBDT with KFold or Stratified KFold


def kfold_lightgbm(train_df, test_df, num_folds, stratified=True, debug=False, name="", need_result=False, fold_random_state=42):

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
    feature_importance_df = pd.DataFrame()
    feats = [f for f in train_df.columns if f not in FEATS_EXCLUDED]
    feats = [f for f in feats if len(train_df[f].unique()) > 1]
    for col in sorted(feats):
        print(col)
    print("### use feature number:", len(feats))
    clfs = []
    # k-fold
    cv_score = 0
    for n_fold, (train_idx, valid_idx) in enumerate(folds.split(train_df[feats], train_df['target'])):
        trn_x, trn_y = train_df[feats].iloc[train_idx], train_df['target'].iloc[train_idx]
        val_x, val_y = train_df[feats].iloc[valid_idx], train_df['target'].iloc[valid_idx]
        params = {
            'objective': 'regression_l2',
            'boosting_type': 'gbdt',
            'n_jobs': 8, 'max_depth': 7,
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
            verbose=100,
            early_stopping_rounds=500,
        )
        clfs.append(clf)

        # params optimized by optuna

        oof_preds[valid_idx] = clf.predict(
            val_x, num_iteration=clf.best_iteration_)
        sub_preds += clf.predict(test_df[feats],
                                 num_iteration=clf.best_iteration_) / folds.n_splits

        fold_importance_df = pd.DataFrame()
        fold_importance_df["feature"] = feats
        fold_importance_df["importance"] = clf.feature_importances_
        fold_importance_df["fold"] = n_fold + 1
        feature_importance_df = pd.concat(
            [feature_importance_df, fold_importance_df], axis=0)
        score = rmse(val_y, oof_preds[valid_idx])
        cv_score += score/folds.n_splits
        print('Fold %2d RMSE : %.6f' % (n_fold + 1, score))

    print("cross validation score:{:.6f}".format(cv_score))

    if not debug:
        # save submission file
        train_df["oof_preds"] = oof_preds
        folder_name, sub_preds, subm_path = save_status(
            train_df[feats+["oof_preds"]], test_df, clfs, cv_score, sub_preds, feature_importance_df, "lgb", name=name)

        save_to_log("lgb", num_folds, stratified,
                    fold_random_state, folder_name, cv_score, params)
    else:
        display_importances(feature_importance_df)

    if need_result:
        return clfs, sub_preds, folder_name, subm_path, cv_score


def kfold_xgboost(train_df, test_df, stratified=True, num_folds=5, debug=False, name="", need_result=False, fold_random_state=42):
    print("Starting XGBoost. Train shape: {}, test shape: {}".format(
        train_df.shape, test_df.shape))

    params = {
        'gpu_id': 0,
        # 'n_gpus': 2,
        'objective': 'reg:linear',
        'eval_metric': 'rmse',
        'silent': True,
        'booster': 'gbtree',
        'n_jobs': 8,
        'n_estimators': 20000,
        'tree_method': 'gpu_hist',
        'grow_policy': 'lossguide',
        'max_depth': 12,
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

    FOLDs = KFold(n_splits=num_folds, shuffle=True,
                  random_state=fold_random_state)

    oof_xgb = np.zeros(len(train_df))
    sub_preds = np.zeros(len(test_df))
    feature_importance_df = pd.DataFrame()
    feats = [f for f in train_df.columns if f not in FEATS_EXCLUDED]
    feats = [f for f in train_df.columns if len(train_df[f].unique()) > 1]
    for col in sorted(feats):
        print(col)
    print("### use feature number:", len(feats))
    clfs = []

    for fold_, (trn_idx, val_idx) in enumerate(FOLDs.split(train_df[feats], train_df["target"])):
        trn_x, trn_y = train_df[feats].iloc[trn_idx], train_df["target"].iloc[trn_idx]
        val_x, val_y = train_df[feats].iloc[val_idx], train_df["target"].iloc[val_idx]
        clf = XGBRegressor(**params)
        clf.fit(
            trn_x, trn_y,
            eval_set=[(trn_x, trn_y), (val_x, val_y)],
            verbose=100,
            early_stopping_rounds=50,
        )
        oof_xgb[val_idx] = clf.predict(val_x, ntree_limit=clf.best_ntree_limit)
        sub_preds += clf.predict(test_df[feats],
                                 ntree_limit=clf.best_ntree_limit) / FOLDs.n_splits

        fold_importance_df = pd.DataFrame()
        fold_importance_df["feature"] = feats
        fold_importance_df["importance"] = clf.feature_importances_
        fold_importance_df["fold"] = num_folds + 1
        feature_importance_df = pd.concat(
            [feature_importance_df, fold_importance_df], axis=0)
        score = rmse(val_y, oof_xgb[val_idx])

        print('no {}-fold loss: {:.6f}'.format(fold_ + 1,
                                               score))
        clfs.append(clf)

    cv_score = rmse(oof_xgb, train_df["target"])
    print("cross validation score:{:.6f}".format(cv_score))

    if not debug:
        # save submission file
        train_df["oof_preds"] = oof_preds

        folder_name, sub_preds, subm_path = save_status(
            train_df[feats+["outliers", "target", "oof_preds"]], test_df, clfs, cv_score, sub_preds, feature_importance_df, "xgb", name=name)
        save_to_log("xgb", num_folds, stratified,
                    fold_random_state, folder_name, cv_score, params)
    else:
        display_importances(feature_importance_df)

    if need_result:
        return clfs, sub_preds, folder_name, subm_path


def save_status(train_df, test_df, clfs, cv_score, sub_preds, feature_importance_df, clf_name, name=""):
    # save submission file
    submission_file_name = 'subm_cv={:.6f}_clf={}_{}.csv'.format(cv_score,
                                                                 clf_name, datetime.datetime.now().strftime('%Y-%m-%d-%H-%M'))
    folder_name = "{}subm_cv={:.6f}_clf={}".format(name, cv_score, clf_name)

    subm_path = os.path.join(folder_name, submission_file_name)
    train_path = os.path.join(folder_name, "train_feat.csv")
    test_path = os.path.join(folder_name, "test_feat.csv")
    if os.path.exists(folder_name) is False:
        os.mkdir(folder_name)
        print("save train test files")
        train_df.to_csv(train_path, index=False)

        test_df.loc[:, 'target'] = sub_preds
        test_df = test_df.reset_index()
        test_df[['card_id', 'target']].to_csv(subm_path, index=False)
        test_df.to_csv(test_path, index=False)
        # display importances
    else:
        test_df = test_df.reset_index()

    clf_path = os.path.join(
        folder_name, "{}_cv={}.pickle".format(clf_name, cv_score))
    with open(clf_path, "wb") as pkl:
        pickle.dump(clfs, pkl)
    feat_importance_path = os.path.join(folder_name, "feature_importance.csv")
    display_importances(feature_importance_df, feat_importance_path)

    return folder_name, test_df[['card_id', 'target']], subm_path


def save_to_log(clf, num_folds, stratified, fold_random_state, path, cv_score, params):
    fit_result_csv = {
        "clf": [clf],
        "cv_score": [cv_score],
        "timestamp": [datetime.datetime.now().strftime('%Y-%m-%d-%H-%M-%S')],
        "num_folds": [num_folds],
        "stratified": [stratified],
        "fold_random_state": [fold_random_state],
        "datapath": [path],
        "params": [params]
    }

    fit_result_csv = pd.DataFrame(fit_result_csv)
    if os.path.exists("fit_result.csv"):
        fit_result_csv.to_csv(
            "fit_result.csv", header=False, mode="a", index=False)
    else:
        fit_result_csv.to_csv(
            "fit_result.csv", header=True, mode="a", index=False)


def only_for_kfold(train_df, test_df, clf="lgb", num_folds=5, stratified=False, debug=False):
    if clf == "lgb":
        kfold_lightgbm(train_df, test_df, num_folds=num_folds,
                       stratified=stratified, debug=debug)
    elif clf == "xgb":
        kfold_xgboost(train_df, test_df, num_folds=num_folds)


def main(embed_mode="numerical", debug=False):
    num_rows = 100000 if debug else None
#     with timer("merchant"):
#         merchant_df = process_merchant(num_rows)

    with timer("historical transactions"):
        hist_df, hist_cm_g = historical_transactions(
            num_rows, embed_mode=embed_mode)

    with timer("new merchants"):
        new_merchant_df, new_cm_g = new_merchant_transactions(
            num_rows, embed_mode=embed_mode)
    with timer("additional merchant card id features"):
        cm_g = card_merchant_additional_feature(hist_cm_g, new_cm_g)

    with timer("additional features"):
        df = pd.concat([new_merchant_df, hist_df], axis=1)
        new_index = new_merchant_df.index
#         print(df.head().index)
#         print(hist_df.head().index)
#         print(mask)

        del new_merchant_df, hist_df
        gc.collect()
        df = pd.concat([df, cm_g], axis=1)
        del cm_g
        gc.collect()
        train_df, test_df = train_test(num_rows, embed_mode=embed_mode)
#         print(train_df.dtypes)
#         for k in df.dtypes.index:
#             print(k,df.dtypes.values[df.dtypes.index==k])
        train_df = train_df.join(df, how='left', on='card_id')
        test_df = test_df.join(df, how='left', on='card_id')
        del df
        gc.collect()

        train_df = additional_features(train_df)
        train_df["no_new_data"] = ~(train_df.index.isin(new_index))
#         train_df["flag_only_authorize"] = train_df["hist_authorized_flag_mean"] == 1
        test_df = additional_features(test_df)
        test_df["no_new_data"] = ~(test_df.index.isin(new_index))
#         test_df["flag_only_authorize"] = test_df["hist_authorized_flag_mean"] == 1
    if debug is False:
        with timer("save tmp df"):
            test_df.to_csv("./tmp_test_df.csv", index=False)
            train_df.to_csv("./tmp_train_df.csv", index=False)

    with timer("Run LightGBM with kfold"):
        kfold_lightgbm(train_df, test_df, num_folds=5,
                       stratified=False, debug=debug)

    return train_df, test_df


def process_merchant(num_rows, embed_mode="numerical"):
    df = pd.read_csv("../merchants.csv", nrows=num_rows)

    df["category_1"] = df["category_1"].replace({"N": 0, "Y": 1})
    df["category_4"] = df["category_4"].replace({"N": 0, "Y": 1})

    if embed_mode == "numerical":
        df["most_recent_sales_range"] = df["most_recent_sales_range"].replace(
            {"E": 4, "D": 3, "C": 2, "B": 1, "A": 0})
        df["most_recent_purchases_range"] = df["most_recent_purchases_range"].replace(
            {"E": 4, "D": 3, "C": 2, "B": 1, "A": 0})
    elif embed_mode == "onehot":
        df["category_2"] = df["category_2"].astype(int)
        df = pd.get_dummies(df, columns=[
                            "most_recent_sales_range", "most_recent_purchases_range", "category_2"])

    inf_col = ["avg_purchases_lag3",
               "avg_purchases_lag6", "avg_purchases_lag12"]
    sale_col = ["avg_sales_lag3", "avg_sales_lag6", "avg_sales_lag12"]
    for col_id, col in enumerate(inf_col):
        df[col] = df[col].replace({np.inf: np.nan})
        df[sale_col[col_id]+"ratio"] = df[sale_col[col_id]]/df[col]

    df.drop(["merchant_group_id", "merchant_category_id",
             "subsector_id", "state_id"], axis=1, inplace=True)

    df.columns = ["merchant_id"] + \
        ["merchant_org_"+col for col in df.columns[1:]]

    df = reduce_mem_usage(df)

    return df


@jit
def card_merchant_additional_jit_part(new_cm_g, hist_cm_g):
    df = None
    for col in new_cm_g.columns:
        if df is None:
            df = pd.DataFrame(new_cm_g[col]+hist_cm_g[col])
            df = df.rename(columns={col: col+"_sum"})
            df[col+"_ratio"] = new_cm_g[col]/hist_cm_g[col]
        else:
            df[col+"_sum"] = new_cm_g[col]+hist_cm_g[col]
            df[col+"_ratio"] = new_cm_g[col]/hist_cm_g[col]

    return df, new_cm_g, hist_cm_g


def card_merchant_additional_feature(new_cm_g, hist_cm_g):
    print("-----------------------")
    print(new_cm_g.columns)
    print("-----------------------")
    print(hist_cm_g.columns)
#     new_cm_g.columns = ["new_"+col for col in new_cm_g.columns]
#     hist_cm_g.columns = ["hist_"+col for col in hist_cm_g.columns]
    df, new_cm_g, hist_cm_g = card_merchant_additional_jit_part(
        new_cm_g, hist_cm_g)

    del new_cm_g, hist_cm_g
    gc.collect()

    print("finish making additinal card-merchant feature")

    aggs = ["mean", "std"]
    df = df.groupby("card_id").agg(aggs)
    df.columns = [col[0]+"_"+col[1] for col in df.columns]
    df = reduce_mem_usage(df)
    return df


@jit
def process_after_agg(df, size_name="card_id_size"):
    if size_name == "card_id_size":
        count_name = "card_id_count"
    else:
        count_name = size_name
    df['CLV'] = df[count_name] * df['purchase_amount_sum'] / df['month_diff_mean']

    df['purchase_date_diff'] = (
        df['purchase_date_max'] - df['purchase_date_min']).dt.days
    df['purchase_date_average'] = df['purchase_date_diff'] / df[size_name]
    df['purchase_date_uptonow'] = (DATE_BASE - df['purchase_date_max']).dt.days
    df['purchase_date_uptomin'] = (DATE_BASE - df['purchase_date_min']).dt.days

    for f in ["purchase_date_max", "purchase_date_min"]:
        df[f] = df[f].astype(np.int64) * 1e-9

    return df


@jit
def process_date_before_agg(df, aggs, holidays, embed_mode="numerical", name="hist"):

    # Y/N to 1/0
    for col in ['category_2', 'category_3', 'authorized_flag']:
        df['purchase_'+col +
            '_mean'] = df.groupby([col])['purchase_amount'].transform('mean')
        df['purchase_'+col +
            '_min'] = df.groupby([col])['purchase_amount'].transform('min')
        df['purchase_'+col +
            '_max'] = df.groupby([col])['purchase_amount'].transform('max')
        df['purchase_'+col +
            '_sum'] = df.groupby([col])['purchase_amount'].transform('sum')
        aggs['purchase_'+col + '_mean'] = ['mean']
    df['authorized_flag'] = df['authorized_flag'].replace(
        {'Y': 1, 'N': 0}).astype(int).astype(np.int16)
    df['category_1'] = df['category_1'].replace(
        {'Y': 1, 'N': 0}).astype(int).astype(np.int16)
    if embed_mode == "numerical":
        df['category_3'] = df['category_3'].replace(
            {'A': 0, 'B': 1, 'C': 2}).astype(int).astype(np.int16)
        aggs["category_3"] = ["mean"]
    elif embed_mode == "onehot":
        df = pd.get_dummies(df, columns=["category_3"])
        for col in df.columns:
            if "category_3" in col:
                aggs[col] = ["mean"]

    # additional features
    df['price'] = df['purchase_amount'] / df['installments']

    # datetime features
    df = process_date(df)

    # holidays
    for d_name, d_day in holidays:
        dist_holiday(df, d_name, d_day, 'purchase_date')

    df['month_diff'] = (DATE_BASE - df['purchase_date']).dt.days // 30
    df['month_diff'] += df['month_lag']

    # additional features
    df['duration'] = df['purchase_amount'] * df['month_diff']
    df['amount_month_ratio'] = df['purchase_amount'] / df['month_diff']
    if name == "hist":
        df['amount_month_lag_ratio'] = df["purchase_amount"] / \
            (df['month_lag'] - 1)
    elif name == "new":
        df['amount_month_lag_ratio'] = df["purchase_amount"] / \
            (df['month_lag'])
    else:
        raise NotImplementedError

    return df, aggs


def agg_by_merchant_card(df):
    #     df['month_diff'] = (DATE_BASE - df['purchase_date']).dt.days // 30
    #     df['month_diff'] += df['month_lag']
    #     if df["authorized_flag"].dtype=="object":
    #         df["authorized_flag"] = df["authorized_flag"].replace({"N":0,"Y":1})
    merchant_aggs = {
        "purchase_amount": ["mean", "max", "min", "nunique", "sum"],
        "purchase_date": ["max", "min"],
        "authorized_flag": ["mean"],
        "month_diff": ["mean", "max", "min"],
        "duration": ["mean", "max", "min"],
        "amount_month_ratio": ["mean", "max", "min", "var"],
        "amount_month_lag_ratio": ["mean", "max", "min", "var"],
        "subsector_id": ["count"]

    }

    cm_d_aggs = {
        "subsector_id_count": ["mean", "max", "std"]
    }
#     df['duration'] = df['purchase_amount'] * df['month_diff']
#     df['amount_month_ratio'] = df['purchase_amount'] / df['month_diff']
    df = df.groupby(["card_id", "merchant_id"]).agg(merchant_aggs)
    new_columns = []
    for col in df.columns:
        new_col = col[0]+"_"+col[1]
        new_columns.append(new_col)
#         if "purchase_date" in col:
#             continue
        cm_d_aggs[new_col] = ["mean", "std"]
    df.columns = new_columns
#     df['purchase_date_merchant_diff'] = (df['purchase_date_merchant_max'] - df['purchase_date_merchant_min']).dt.days
#     df['purchase_date_merchant_average'] = df['purchase_date_merchant_diff'] / df['subsector_id_merchant_count']
#     df['purchase_date_merchant_uptonow'] = (DATE_BASE - df['purchase_date_merchant_max']).dt.days
#     df['purchase_date_merchant_uptomin'] = (DATE_BASE - df['purchase_date_merchant_min']).dt.days
    df = process_after_agg(df, "subsector_id_count")

#     for f in ["purchase_date_merchant_max","purchase_date_merchant_min"]:
#         df[f] = df[f].astype(np.int64) * 1e-9

    cm_g_agg = df.groupby("card_id").agg(cm_d_aggs)
    cm_g_agg.columns = [col[0]+"_merchant_card_"+col[1]
                        for col in cm_g_agg.columns]
#     df = reduce_mem_usage(df)
    return df, cm_g_agg


def save_status_intermediate(train_df, test_df, prediction, clfs, clf_name, folder_name, feature_name):
    # save submission file

    train_path = os.path.join(
        folder_name, f"train_add_{feature_name}_clf={clf_name}.csv")
    test_path = os.path.join(
        folder_name, f"test_add_{feature_name}_clf={clf_name}.csv")
    if os.path.exists(folder_name) is False:
        #         os.mkdir(folder_name)
        #         print("save train test files")
        #         train_df.to_csv(train_path, index=True)

        #         test_df.loc[:,feature_name] = prediction
        #         test_df = test_df.reset_index()
        #         test_df[['card_id', 'target']].to_csv(subm_path, index=False)
        #         test_df.to_csv(test_path, index=True)
        print("something wrong")
        return train_df, prediction
        # display importances
    else:
        train_df.to_csv(train_path, index=True)
        test_df[feature_name] = prediction
        if "card_id" not in test_df.columns:
            test_df = test_df.reset_index()
        test_df.to_csv(test_path, index=False)
#     clf_path = os.path.join(folder_name, "{}_cv={}.pickle".format(clf_name,cv_score))
#     with open(clf_path, "wb") as pkl:
#         pickle.dump(clfs,pkl)

    return folder_name, test_df[['card_id', feature_name]]
