import numpy as np
import argparse
import pandas as pd
import os
from dae import DaeTrainer
from collections import namedtuple

DATA_FOLDER = "/home/minteiko/developer/project/kaggle/santander/"


def make_config(folder_name,
                model_type,
                optimizer="adam",
                max_epoch=1000,
                early_stopping=40,
                batchsize=1024,
                debug=False,
                lr_patience=np.inf,
                name="",
                use_test=False,
                lr=0.001,
                loss_name="mse",
                use_rank_gauss=True,
                use_tensorboard=True,
                log_path="log/"
                ):
    if model_type == "normal":
        params = {
            # "input_dim": len(feats),
            "layer_units": [600, 600],
            "activations": ["relu","relu"]
        }

    elif model_type == "flatten":
        params = {
            # "input_dim":len(feats),
            "layer_units": [16, 8],
            "activations": ["relu", "relu"]
        }
    elif model_type == "flatten_cnn":
        params = {
            # "input_dim":len(feats),
            "hidden_dim": [128, 64, 32]
        }
    else:
        raise NotImplementedError
    Config = namedtuple('Config', ('params', 'use_tensorboard', 'use_rank_gauss', 'optimizer', 'early_stopping', 'lr_patience', 'max_epoch', 'log_path',
                                   'folder_name', 'lr', "model_type", "batchsize", "debug", "name", "loss_name", "use_test"))
    config = Config(params=params, optimizer=optimizer, use_tensorboard=use_tensorboard, use_rank_gauss=use_rank_gauss, max_epoch=max_epoch, folder_name=folder_name, early_stopping=early_stopping, lr_patience=lr_patience,
                    log_path=log_path, model_type=model_type, batchsize=batchsize, debug=debug, name=name, lr=lr, loss_name=loss_name, use_test=use_test)
    return config


def string2bool(s):

    return s.lower() == "true"


def make_parametors():
    parser = argparse.ArgumentParser()

    # Model hyper-parameters
    parser.add_argument('--model', type=str, default='normal',
                        choices=['flatten', 'normal'])
    parser.add_argument('--folder', default=DATA_FOLDER+"/feature/dae")
    parser.add_argument('--loss_name', default='mse')
    parser.add_argument('--optimizer', default='adam', choices=["adam", "sgd"])
    parser.add_argument('--early_stopping', type=int, default=30)
    parser.add_argument('--lr_patience', type=int, default=np.inf)
    parser.add_argument('--lr', type=int, default=0.003)

    parser.add_argument('--max_epoch', type=int, default=1000)
    parser.add_argument('--batchsize', type=int, default=1024)
    parser.add_argument('--use_rank_gauss', action="store_true")
    parser.add_argument('--use_test', action="store_true")

    parser.add_argument('--debug', action="store_true")
    parser.add_argument('--use_tensorboard', type=string2bool, default="true")

    parser.add_argument('--name', default="")
    parser.add_argument('--log_path', default="tensorboard_log/")

    return parser.parse_args()


def make_directory(path):
    if os.path.exists(path) is False:
        os.mkdir(path)


def main():
    args = make_parametors()
    log_path = os.path.join(args.folder, args.log_path)
    make_directory(log_path)

    config = make_config(args.folder,
                         args.model,
                         args.optimizer,
                         args.max_epoch,
                         args.early_stopping,
                         args.batchsize,
                         args.debug,
                         args.lr_patience,
                         args.name,
                         args.use_test,
                         args.lr,
                         args.loss_name,
                         args.use_rank_gauss,
                         args.use_tensorboard,
                         log_path
                         )
    train = pd.read_csv(DATA_FOLDER + "/train.csv")
    test = pd.read_csv(DATA_FOLDER + "/test.csv")
    trainer = DaeTrainer(train, test, config)
    trainer.train()


if __name__ == "__main__":
    main()
