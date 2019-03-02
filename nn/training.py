import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import os
import sys
import matplotlib.pyplot as plt
import datetime
from torch import optim
from torch.utils.data import DataLoader
from sklearn.metrics import mean_squared_error, log_loss
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import Imputer
from pandas.core.common import SettingWithCopyWarning
from model import Net
import warnings
import pickle


warnings.simplefilter(action='ignore', category=SettingWithCopyWarning)
warnings.simplefilter(action='ignore', category=FutureWarning)

sys.path.append("/home/minteiko/developer/project/fm/")

import xdeepfm

FEATS_EXCLUDED = ['first_active_month', 'target', 'card_id', 'outliers',
                  'hist_purchase_date_max', 'hist_purchase_date_min',
                  'hist_card_id_size',
                  'new_purchase_date_max', 'new_purchase_date_min',
                  'new_card_id_size',
                  'OOF_PRED', 'month_0']


def rmse(y_true, y_pred):
    return np.sqrt(mean_squared_error(y_true, y_pred))


def make_model(params, field_num, input_size, output_size=1, device="cpu"):
    model = xdeepfm.xDeepFM(
        input_size, output_size, field_num, params["embed_dim"], params["dnn_params"], params["cin_params"], params["activation"], device=device)

    return model


def train_nn(model, criterion, train_data, valid_data, optimizer, max_iter, log_interval=100, test_data=None):
    model.train()
    valid_loss = []
    train_loss = []
    min_val_loss = np.inf
    best_model = None
    best_outputs = None
    for epoch in range(max_iter):
        for batch_idx, x_batch in enumerate(train_data):
            x_batch, y_batch = x_batch[:, :-1].cuda(), x_batch[:, -1].cuda()
#             x_batch = x_batch.type(torch.LongTensor).cuda()
            optimizer.zero_grad()
            output = model(x_batch)
            loss = criterion(output.squeeze(), y_batch)
            loss.backward()
            optimizer.step()
        print('Train Epoch: {} \t MSE Loss: {:.6f}'.format(
            epoch, np.sqrt(loss.item())))
        train_loss.append(loss.item())
        val_loss, outputs = evaluator(model, criterion, valid_data)
        if min_val_loss > val_loss:
            min_val_loss = val_loss
            best_model = type(model)(
                model.input_dim, model.output_dim)  # get a new instance
            # copy weights and stuff
            best_model.load_state_dict(model.state_dict())
            best_outputs = np.concatenate(outputs)
            best_iter = epoch
        valid_loss.append(val_loss)
    print("best result epoch:{}, validation loss:{}".format(
        best_iter, min_val_loss))
    return best_model, train_loss, valid_loss, best_outputs, min_val_loss


def evaluator(model, criterion, valid_data):
    model.eval()
    with torch.no_grad():
        loss = 0
        outputs = []
        for batch_idx, x_batch in enumerate(valid_data):
            x_batch, y_batch = x_batch[:, :-1].cuda(), x_batch[:, -1].cuda()
#             x_batch = x_batch.type(torch.LongTensor).cuda()
            output = model(x_batch)
            # print(batch_idx)
            # print(output.size(), y_batch.size())
            cur_loss = criterion(output.squeeze(), y_batch)
            loss += cur_loss
            outputs.append(output.squeeze().cpu().numpy())
        loss = np.sqrt(loss.item()/(batch_idx+1))
        print('Valid MSE Loss: {:.6f}'.format(
            loss,))
    model.train()
    return loss, outputs


def kfold_nn(train_df, test_df, stratified=False, num_folds=5, debug=False, name="", need_result=False, fold_random_state=42):
    print("Starting Neural Net. Train shape: {}, test shape: {}".format(
        train_df.shape, test_df.shape))
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    np.random.seed(0)
    torch.manual_seed(0)

    batchsize = 1024

    FOLDs = KFold(n_splits=num_folds, shuffle=True,
                  random_state=fold_random_state)

    oof_xgb = np.zeros(len(train_df))
    sub_preds = np.zeros(len(test_df))
    feats = [f for f in train_df.columns if f not in FEATS_EXCLUDED]
    feats = [f for f in feats if len(
        train_df[f].unique()) > 1 and len(test_df[f].unique()) > 1]
    for col in sorted(feats):
        print(col)
    print("### use feature number:", len(feats))

    params = {
        "input_dim": len(feats),
        "output_dim": 1

    }

    criterion = nn.MSELoss()
    folder_name = "nn_subm_cv_normed/"
    test_pickle_name = f"{folder_name}/test_feats.pickle"
    if os.path.exists(test_pickle_name):
        with open(test_pickle_name, "rb") as pkl:
            test_df[feats] = pickle.load(pkl)
    else:
        test_df[feats] = preprocess_for_nn(test_df[feats])
        if os.path.exists(folder_name) is False:
            os.mkdir(folder_name)
        with open(test_pickle_name, "wb") as pkl:
            pickle.dump(test_df[feats], pkl)

    test_iter = DataLoader(torch.FloatTensor(
        test_df[feats].values), batch_size=batchsize, shuffle=False)

    losses = {
        "train": [],
        "valid": []
    }
    transformed_folder_path = "./transformed_normed"
    if os.path.exists(transformed_folder_path) is False:
        os.mkdir(transformed_folder_path)

    for fold_, (trn_idx, val_idx) in enumerate(FOLDs.split(train_df[feats], train_df["target"])):
        train_path = os.path.join(
            transformed_folder_path, "trn_total{}_fold:{}.csv".format(num_folds, fold_+1))
        valid_path = os.path.join(
            transformed_folder_path, "val_total{}_fold:{}.csv".format(num_folds, fold_+1))

        flag_data = os.path.exists(train_path)
        if flag_data:
            print("load file from pickle ...")
            with open(train_path, "rb") as pkl:
                trn = pickle.load(pkl)
            with open(valid_path, "rb") as pkl:
                val = pickle.load(pkl)
            print("finish!")
        else:
            print("make transformed data and save it ...")
            trn = train_df[feats+["target"]].iloc[trn_idx]
            val = train_df[feats+["target"]].iloc[val_idx]
            trn[feats], val[feats] = preprocess_for_nn(trn[feats], val[feats])
            if os.path.exists(transformed_folder_path) is False:
                os.mkdir(transformed_folder_path)
            with open(train_path, "wb") as pkl:
                pickle.dump(trn, pkl)
            with open(valid_path, "wb") as pkl:
                pickle.dump(val, pkl)
            print("finish!")

#         trn = train_df[feats+["target"]].iloc[trn_idx]
#         val = train_df[feats+["target"]].iloc[val_idx]
#         trn[feats], val[feats] = preprocess_for_nn(trn[feats], val[feats])

        print("start training fold no {}".format(fold_))
        net = Net(**params).cuda()
        net.apply(init_weights)
        train_iter = DataLoader(torch.FloatTensor(
            trn.values), batch_size=batchsize, shuffle=True)
        valid_iter = DataLoader(torch.FloatTensor(
            val.values), batch_size=batchsize, shuffle=False)
        valid_iter_2 = DataLoader(torch.FloatTensor(
            trn.values), batch_size=batchsize, shuffle=False)
        optimizer = optim.Adam(net.parameters(), lr=1e-3,
                               weight_decay=1e-4)
        net, t_loss, v_loss, outputs, min_val_loss = train_nn(net,
                                                              criterion=criterion,
                                                              train_data=train_iter,
                                                              valid_data=valid_iter,
                                                              optimizer=optimizer,
                                                              max_iter=20,
                                                              valid_data2=valid_iter_2,
                                                              )
        net.cuda()

        oof_xgb[val_idx] = outputs
        sub_preds += predict(net, test_iter)
        print("check eval")
        evaluator(net, criterion, valid_iter, valid_iter_2)

        score = rmse(val["target"], oof_xgb[val_idx])

        print('no {}-fold loss: {:.6f}'.format(fold_ + 1,
                                               score))
        losses["train"].append(t_loss)
        losses["valid"].append(v_loss)

#         raise NotImplementedError

    sub_preds /= num_folds

    cv_score = np.sqrt(mean_squared_error(oof_xgb, train_df["target"]))
    print("cross validation score:{:.6f}".format(cv_score))
    train_df["oof_preds"] = oof_xgb
    save_folder_name = "./{}/cv={:.6f}".format(folder_name, cv_score)
    if os.path.exists(save_folder_name) is False:
        os.mkdir(save_folder_name)
    train_df.to_csv(save_folder_name+"/train_feat.csv", index=False)
    subm_path = "{}/subm_cv={:.6f}.csv".format(save_folder_name, cv_score)
    test_df["target"] = sub_preds
    test_df[["card_id", "target"]].to_csv(subm_path, index=False)


def predict(model, test_iter):
    model.eval()
    with torch.no_grad():
        predict = []
        for batch_idx, x_batch in enumerate(test_iter):
            x_batch = x_batch.cuda()
#             x_batch = x_batch.type(torch.LongTensor).cuda()
            output = model(x_batch)
            predict.append(output.squeeze().cpu().numpy())
    return np.concatenate(predict)


def preprocess_for_nn(trn, val=None):
    print("transforming data ...")
    inf_columns = trn.columns[(trn.max() == np.inf) | (trn.min() == -np.inf)]
    for col in inf_columns:
        trn[col] = trn[col].replace({np.inf: np.nan})
        trn[col] = trn[col].replace({-np.inf: np.nan})
#         print(col ,trn[col].nunique())


#     print("train shape:", trn.shape)
#     print(trn.isnull().mean())
    ss = StandardScaler()
    imp = Imputer(strategy="median")
    trn = imp.fit_transform(trn)
    trn = ss.fit_transform(trn)

    if val is None:
        #         print("after shape:", trn.shape)
        return trn
    else:
        print("transforming valid data ...")

        inf_columns = val.columns[(val.max() == np.inf)
                                  | (val.min() == -np.inf)]
        for col in inf_columns:
            val[col] = val[col].replace({np.inf: np.nan})
            val[col] = val[col].replace({-np.inf: np.nan})
#             print(col, val[col].nunique())
#         print(val.isnull().mean())

        imp = Imputer(strategy="median")
#         for col in val.columns:
#             print(col, val[col].nunique())
        val = imp.fit_transform(val)
        val = ss.fit_transform(val)
#         print(trn.shape)
#         print(val.shape)

        return trn, val


def init_weights(m):
    try:
        if m.weight.dim() > 1:
            nn.init.xavier_normal_(m.weight)
            m.bias.data.fill_(0.01)
    except AttributeError:
        pass
