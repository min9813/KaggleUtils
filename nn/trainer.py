import numpy as np
import torch
import torch.nn as nn
import pickle
import logging
import datetime
import gc
import matplotlib.pyplot as plt
import os
import pandas as pd
import numpy as np
import seaborn as sns
import time
import warnings
import pickle
import sys
import shutil
from tensorboardX import SummaryWriter
from model import SharedDAE, SimpleDAE
from logging import Logger, StreamHandler, FileHandler
from torch import optim
from torch.utils.data import DataLoader
from sklearn.metrics import mean_squared_error, log_loss, roc_auc_score
from sklearn.model_selection import KFold, StratifiedKFold, train_test_split
from contextlib import contextmanager
from pandas.core.common import SettingWithCopyWarning
from numba.decorators import jit
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import Imputer
from collections import namedtuple
from scipy.special import erfinv
warnings.simplefilter(action='ignore', category=SettingWithCopyWarning)
warnings.simplefilter(action='ignore', category=FutureWarning)
# sys.path.append("/home/minteiko/developer/project/fm/")

# import xdeepfm

warnings.filterwarnings('ignore')

FEATS_EXCLUDED = ["ID_code", "target", "oof_preds"]


class DaeTrainer(object):

    def __init__(self, train_df, test_df, config):
        self.params = config.params
        self.opt_name = config.optimizer
        self.max_epoch = config.max_epoch
        self.log_interval = 100
        self.max_lr = config.lr
        self.base_lr = config.lr

        self.train_df = train_df
        self.test_df = test_df

        self.folder_name = config.folder_name
        self.model_type = config.model_type
        self.batchsize = config.batchsize

        self.debug = config.debug
        self.name = config.name

        self.loss_name = config.loss_name
        self.has_null = False

        self.control_init_weight = True

        self.start_epoch = 10
        self.end_epoch = 50
        self.max_alpha = 1

        self.early_stopping = config.early_stopping
        self.lr_patience = config.lr_patience

        self.use_rank_gauss = config.use_rank_gauss
        self.use_test = config.use_test

        self.log_name = "train"
        self.log_filename = "log.txt"

        self.log_path = config.log_path

        self.no_improve_lr = 0
        self.swap_p = 0.15

        self.set_logger()

        self.logger.info(config)


        self.use_tensorboard = config.use_tensorboard
        if self.use_tensorboard:
            self.build_tensorboard()

    def train(self):
        self.prepare_train()
        self.kfold_nn()

    def train_nn(self):
        self.min_val_loss = np.inf
        start = time.time()
        for epoch in range(self.max_epoch):
            self.net.train()
            for batch_idx, x_batch in enumerate(self.train_loader):
                x_batch, y_batch = x_batch[:, :-
                                           1].cuda(), x_batch[:, -1].cuda()
    #             x_batch = x_batch.type(torch.LongTensor).cuda()
                self.optimizer.zero_grad()
                output = self.net(x_batch)
                loss = self.criterion(output.squeeze(), y_batch)
                loss.backward()
                self.optimizer.step()
            self.net.eval()
            with torch.no_grad():
                loss = 0
                t_rocauc = 0
                for batch_idx, x_batch in enumerate(self.train_loader):
                    x_batch, y_batch = x_batch[:, :-
                                               1].cuda(), x_batch[:, -1].cuda()
        #             x_batch = x_batch.type(torch.LongTensor).cuda()
                    output = self.net(x_batch)
                    output = output.squeeze()
                    # print(batch_idx)
                    # print(output.size(), y_batch.size())
        #             cur_loss = criterion(output, y_batch)
                    cur_loss = self.criterion(output.squeeze(), y_batch)
                    t_rocauc += roc_auc_score(cuda2numpy(y_batch, False),
                                              cuda2numpy(output, False))
                    loss += cur_loss
                t_rocauc /= batch_idx+1
                t_loss = loss.item()/(batch_idx+1)
                epoch_message = "epoch [{}/{}] elapsed:{:.0f}s  Train {}:{:.6f} rocauc:{:.6f} ".format(
                    epoch+1, self.max_epoch, time.time()-start, self.loss_name, t_loss, t_rocauc)
                self.logger.info(epoch_message)
                loss = 0
                outputs = []
                val_rocauc = 0
                for batch_idx, x_batch in enumerate(self.valid_loader):
                    x_batch, y_batch = x_batch[:, :-
                                               1].cuda(), x_batch[:, -1].cuda()
        #             x_batch = x_batch.type(torch.LongTensor).cuda()
                    output = self.net(x_batch)
                    output = output.squeeze()
                    # print(batch_idx)
                    # print(output.size(), y_batch.size())
        #             cur_loss = criterion(output, y_batch)
                    cur_loss = self.criterion(output.squeeze(), y_batch)

                    val_rocauc += roc_auc_score(cuda2numpy(y_batch, False),
                                                cuda2numpy(output, False))
                    loss += cur_loss
                    outputs.append(output.squeeze().cpu().numpy())
                val_rocauc /= batch_idx+1
                loss = loss.item()/(batch_idx+1)
                epoch_message += 'Valid {}: {:.6f} rocauc: {:.6f}'.format(
                    self.loss_name, loss, val_rocauc)
                self.logger.info(epoch_message)

            if self.use_tensorboard:
                self.writer.add_scalars("data/loss", {
                    "train": t_loss,
                    "valid": loss
                }, epoch+1)
                for name, param in self.net.named_parameters():
                    self.writer.add_histogram(
                        "model/"+name, param.clone().cpu().detach().numpy(), epoch+1)
            check_stop = self.check_learning(epoch, val_rocauc, outputs)
            if check_stop:
                break
        self.logger.info("best result epoch:{}, {}:{}".format(
            self.best_iter+1, self.loss_name, self.min_val_loss))
        return self.min_val_loss

    def check_learning(self, epoch, val_loss):
        if self.min_val_loss > val_loss:
            self.min_val_loss = val_loss
            self.best_model = type(self.net)(
                **self.params)  # get a new instance
            # copy weights and stuff
            self.best_model.load_state_dict(self.net.state_dict())
            self.best_iter = epoch
            self.no_improve_lr = 0
            return False
        else:
            if epoch - self.best_iter >= self.early_stopping:
                self.logger.info(
                    f"not improve in {self.early_stopping} epoch, end train")
                self.net.eval()
                return True
            elif self.no_improve_lr >= self.lr_patience:
                self.no_improve_lr = 0
                self.logger.info("Setting lr to {}".format(
                    self.optimizer.param_groups[0]["lr"]))
            else:
                self.no_improve_lr += 1
            return False
        self.optimizer.param_groups[0]["lr"] *= 0.995

    def weighted_binary_cross_entropy(self, output, target, eps=1e-12):
        output = self.sigmoid(output)
        if self.weights is not None:
            assert len(self.weights) == 2

            loss = self.weights[1] * (target * torch.log(output+eps)) + \
                self.weights[0] * ((1 - target) * torch.log(1 - output + eps))
        else:
            loss = target * torch.log(output) + \
                (1 - target) * torch.log(1 - output)

        return torch.neg(torch.mean(loss))

    def make_criterion(self):
        if self.loss_name == "mse":
            self.criterion = nn.MSELoss()
        elif self.loss_name == "mae":
            self.criterion = self.mean_absolute_loss
        elif self.loss_name == 'hindge':
            self.criterion = nn.HingeEmbeddingLoss()
        elif self.loss_name == 'bce':
            self.criterion = nn.BCEWithLogitsLoss()
        else:
            raise NotImplementedError

    def mean_absolute_loss(self, y_pred, y_true):
        return torch.abs(y_pred - y_true)

    def prepare_train(self):
        if os.path.exists(self.folder_name) is False:
            os.mkdir(self.folder_name)
        self.feats = [
            f for f in self.train_df.columns if f not in FEATS_EXCLUDED]
        self.feats = [f for f in self.feats if len(
            self.train_df[f].unique()) > 1 and len(self.test_df[f].unique()) > 1]
        for col in sorted(self.feats):
            self.logger.info(col)
        self.logger.info("### use feature number: {}".format(len(self.feats)))
        self.params["input_dim"] = len(self.feats)

        self.make_criterion()

        if self.use_rank_gauss:
            data_feat_name = f"{self.folder_name}/transformed_rank_gauss_feats.csv"

        else:
            data_feat_name = f"{self.folder_name}/transformed_feats.csv"
        if os.path.exists(data_feat_name):
            self.train_df = pd.read_csv(data_feat_name)
            del self.test_df
            gc.collect()
        else:
            self.train_df = self.train_df[self.feats +
                                          ["target"]].append(self.test_df[self.feats])
            self.train_df[self.feats] = self.preprocess_for_nn(
                self.train_df[self.feats])
            if os.path.exists(self.folder_name) is False:
                os.mkdir(self.folder_name)
            self.train_df.to_csv(data_feat_name, index=False)
        self.logger.debug(f"Datafram Shape:{self.train_df.shape}")

    def setup_optimizer(self):
        if self.opt_name == "adam":
            self.optimizer = optim.Adam(self.net.parameters(), lr=self.max_lr,
                                        weight_decay=1e-4)
        elif self.opt_name == "sgd":
            self.optimizer = optim.SGD(
                self.net.parameters(), lr=self.max_lr, weight_decay=1e-4)
        else:
            raise NotImplementedError

    def make_model(self):
        # if self.model_type == "normal":
        #     self.net = SimpleDAE(**self.params)
        # elif self.model_type == "flatten":
        #     self.net = SharedDAE(**self.params)
        if self.model_type == "cross":
            self.net = CrossNet(**self.params)
        elif self.model_type == "flatten":
            self.net = Simple_NN(**self.params)
        elif self.model_type == "flatten_cnn":
            self.net = SimpleCNN(**self.params)
        elif self.model_type == "self_attn":
            self.net = SelfAttntionNet(**self.params)
        else:
            raise NotImplementedError

    def kfold_nn(self):
        self.logger.info("Starting Neural Net. Train shape: {}".format(
            self.train_df.shape))
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

        np.random.seed(0)
        torch.manual_seed(0)

        for fold_, (trn_idx, val_idx) in enumerate(self.folds.split(self.train_df[self.feats], self.train_df["target"])):
            train_path = os.path.join(
                self.transformed_folder_path, "trn_total{}_fold:{}.csv".format(self.num_folds, fold_+1))
            valid_path = os.path.join(
                self.transformed_folder_path, "val_total{}_fold:{}.csv".format(self.num_folds, fold_+1))

            flag_data = os.path.exists(train_path)
            if flag_data:
                self.logger.debug("load file from pickle ...")
                with open(train_path, "rb") as pkl:
                    trn = pickle.load(pkl)
                with open(valid_path, "rb") as pkl:
                    val = pickle.load(pkl)
                self.logger.debug("finish!")
            else:
                self.logger.info("make transformed data and save it ...")
                trn = self.train_df[self.feats+["target"]].iloc[trn_idx]
                val = self.train_df[self.feats+["target"]].iloc[val_idx]
                trn[self.feats], val[self.feats] = self.preprocess_for_nn(
                    trn[self.feats], val[self.feats])
                if os.path.exists(self.transformed_folder_path) is False:
                    os.mkdir(self.transformed_folder_path)
                with open(train_path, "wb") as pkl:
                    pickle.dump(trn, pkl)
                with open(valid_path, "wb") as pkl:
                    pickle.dump(val, pkl)
                self.logger.debug("finish!")

                self.logger.info("start training ")
                self.make_model()
                self.logger.info("########### Net Architecture ###########")
                self.logger.info(str(self.net))
                self.logger.info("########################################")
                if self.control_init_weight:
                    self.net.apply(self.init_weights)
                self.net.cuda()

        # if self.use_test:
        #     trn, val = train_test_split(
        #         self.train_df.loc[:, self.feats], test_size=0.2)
        #     self.train_loader = DataLoader(torch.FloatTensor(
        #         trn.values), batch_size=self.batchsize, shuffle=True)
        #     self.valid_loader = DataLoader(torch.FloatTensor(
        #         val.values), batch_size=self.batchsize, shuffle=False)
        # else:
                self.test_df = self.train_df.loc[self.train_df.target.isnull(
                ), self.feats]
                self.train_df = self.train_df.loc[self.train_df.target.notnull(
                ), self.feats]

                self.train_loader = DataLoader(torch.FloatTensor(
                    self.train_df.loc[:, self.feats].values), batch_size=self.batchsize, shuffle=True)
                self.valid_loader = DataLoader(torch.FloatTensor(
                    self.test_df.loc[:, self.feats].values), batch_size=self.batchsize, shuffle=False)
                self.setup_optimizer()
                min_val_loss = self.train_nn()
                self.best_model.cuda()
                del self.train_loader, self.valid_loader
                gc.collect()

        self.save_name = "cv={:.6f}".format(
            min_val_loss) + f"{self.loss_name}_{self.model_type}_{self.opt_name}"

        if self.use_rank_gauss:
            self.save_name += '_rg'
        else:
            self.save_name += '_ss'

        self.save_name = os.path.join(self.folder_name, self.save_name)
        if os.path.exists(self.save_name) is False:
            os.mkdir(self.save_name)
        if os.path.exists(self.save_name+"/"+self.log_filename) is False:
            shutil.move(self.log_filename, self.save_name)
        if os.path.exists(self.save_name+"/tensorboard") is False:
            shutil.move(self.log_path, self.save_name+"/tensorboard")
        np.save(self.save_name+"/feats", learned_feats)

    def save_train_output(self, cv_score):
        save_folder_name = "./{}/cv={:.6f}_NN_{}".format(
            self.folder_name, cv_score, self.model_type)
        if self.stratified:
            save_folder_name += "_stratified"
        if self.use_rank_gauss:
            save_folder_name += "_rankgauss"
        if os.path.exists(save_folder_name) is False:
            os.mkdir(save_folder_name)
            self.train_df[["ID_code", "oof_preds"]].to_csv(
                save_folder_name+"/oof_preds.csv", index=False)
        subm_path = "{}/subm_cv={:.6f}.csv".format(save_folder_name, cv_score)
        self.test_df[["ID_code", "target"]].to_csv(subm_path, index=False)

    def predict(self):
        self.best_model.eval()
        with torch.no_grad():
            predicted = []
            for batch_idx, x_batch in enumerate(self.test_loader):
                x_batch = x_batch.cuda()
    #             x_batch = x_batch.type(torch.LongTensor).cuda()
                output = self.best_model(x_batch)
                predicted.append(output.squeeze().cpu().numpy())
        return np.concatenate(predicted)

    def preprocess_for_nn(self, trn, val=None):
        self.logger.debug("transforming data ...")
        if self.has_null:
            inf_columns = trn.columns[(
                trn.max() == np.inf) | (trn.min() == -np.inf)]
            for col in inf_columns:
                trn[col] = trn[col].replace({np.inf: np.nan})
                trn[col] = trn[col].replace({-np.inf: np.nan})
                self.logger.debug(col, trn[col].nunique())
            imp = Imputer(strategy="median")
            trn = imp.fit_transform(trn)
    #     print("train shape:", trn.shape)
    #     print(trn.isnull().mean())

        if self.use_rank_gauss:
            self.logger.info("use rank gauss transformation")
            trn = DaeTrainer.rank_gauss(trn)
        else:
            self.logger.info("use standard scalar transformation")
            ss = StandardScaler()
            trn = ss.fit_transform(trn)

        if val is None:
            #         print("after shape:", trn.shape)
            return trn
        else:
            self.logger.debug("transforming valid data ...")
            if self.has_null:
                inf_columns = val.columns[(val.max() == np.inf)
                                          | (val.min() == -np.inf)]
                for col in inf_columns:
                    val[col] = val[col].replace({np.inf: np.nan})
                    val[col] = val[col].replace({-np.inf: np.nan})
                    # print(col, val[col].nunique())
                # print(val.isnull().mean())

                imp = Imputer(strategy="median")
                for col in val.columns:
                    self.logger.debug(col, val[col].nunique())
                val = imp.fit_transform(val)
            if self.use_rank_gauss:
                val = DaeTrainer.rank_gauss(val)
            else:
                val = ss.fit_transform(val)
    #         print(trn.shape)
    #         print(val.shape)
            return trn, val

    def init_weights(self, m):
        try:
            if m.weight.dim() > 1:
                nn.init.xavier_normal_(m.weight)
                m.bias.data.fill_(0.01)
        except AttributeError:
            pass

    def set_logger(self):
        self.logger = Logger(self.log_name+"_dae")
        self.logger.setLevel(logging.DEBUG)
        stream_handler = StreamHandler()
        stream_handler.setLevel(logging.DEBUG)
        file_handler = FileHandler(self.log_filename, mode="w")
        file_handler.setLevel(logging.INFO)

        self.logger.addHandler(stream_handler)
        self.logger.addHandler(file_handler)

    def build_tensorboard(self):
        self.writer = SummaryWriter(self.log_path)

    @staticmethod
    def rank_gauss(df):
        df = df.rank()
        print('calc min ...')
        m = df.min()
        print('calc max ...')

        M = df.max()
        df = (df-m)/((M-m))
        assert all(df.max()) == 1
        assert all(df.min()) == 0

        df = (df - 0.5)*(2 - 1e-9)
        df = erfinv(df)
        print('calc mean ...')
    #     df = df - df.mean()
        return df


def cuda2numpy(tensor, grad=True):
    if grad:
        tensor = tensor.detach()

    return tensor.cpu().numpy()
