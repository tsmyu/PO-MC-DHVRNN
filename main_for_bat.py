from matplotlib import image
from sequencing import get_sequences, get_bat_sequence_data
from utilities import *
from helpers import *
from preprocessing import *
from vrnn.datasets import GeneralDataset
from vrnn.models.utils import num_trainable_params, roll_out
from vrnn.models import load_model
from datetime import datetime
from math import sqrt
import glob
import os
import sys
import math
import warnings
import copy
import time
from copy import deepcopy
import argparse
import random

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import hmmlearn

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

# from model_wrapper import ModelWrapper

import tensorboardX as tbx

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["HDF5_USE_FILE_LOCKING"] = "FALSE"

# customized ftns

# from scipy import signal

# Keisuke Fujii, 2020
# modifying the codes
# https://github.com/samshipengs/Coordinated-Multi-Agent-Imitation-Learning
# https://github.com/ezhan94/multiagent-programmatic-supervision

parser = argparse.ArgumentParser()
parser.add_argument("--data", type=str, required=True)
parser.add_argument("--n_GorS", type=int, required=True)
parser.add_argument("--n_roles", type=int, required=True)
parser.add_argument("--val_devide", type=int, default=10)
parser.add_argument("--hmm_iter", type=int, default=500)
parser.add_argument("-t_step", "--totalTimeSteps", type=int, default=796)
parser.add_argument("--overlap", type=int, default=40)
parser.add_argument("-k", "--k_nearest", type=int, default=0)
parser.add_argument("--batchsize", type=int, required=True)
parser.add_argument("--n_epoch", type=int, required=True)
parser.add_argument("--attention", type=int, default=-1)
parser.add_argument("--normalize", action="store_true")
parser.add_argument("--model", type=str, required=True)
parser.add_argument(
    "-ev_th",
    "--event_threshold",
    type=int,
    required=True,
    help="event with frames less than the threshold will be removed",
)
parser.add_argument("--fs", type=int, default=10)
# parser.add_argument('-subs_fac','--subsample_factor', type=int, required=True, help='too much data should be downsampled by subs_fac')
# parser.add_argument('--filter', action='store_true')
parser.add_argument("--body", action="store_true")
parser.add_argument("--acc", type=int, default=0)
parser.add_argument("--vel_in", action="store_true")
parser.add_argument("--in_out", action="store_true")
parser.add_argument("--in_sma", action="store_true")
# parser.add_argument('--meanHMM', action='store_true')
parser.add_argument("--cont", action="store_true")
parser.add_argument("--numProcess", type=int, default=16)
parser.add_argument("--TEST", action="store_true")
parser.add_argument("--Sanity", action="store_true")
parser.add_argument("--hard_only", action="store_true")
parser.add_argument("--wo_macro", action="store_true")
parser.add_argument("--res", action="store_true")
parser.add_argument("--jrk", type=float, default=0)
parser.add_argument("--lam_acc", type=float, default=0)
parser.add_argument("--pretrain", type=int, default=0)
parser.add_argument("--pretrain2", type=int, default=0)
parser.add_argument("--finetune", action="store_true")
parser.add_argument("--drop_ind", action="store_true")
args, _ = parser.parse_known_args()

# directories
main_dir = "../"  # './'
game_dir = main_dir + "data_" + args.data + "/"
Data = LoadData(main_dir, game_dir, args.data)
path_init = "./weights/"  # './weights_vrnn/init/'
args.seed = 200


def run_epoch(model, train, rollout, hp):
    loader = train_loader if train == 1 else val_loader if train == 0 else test_loader

    losses = {}
    losses2 = {}
    for batch_idx, data in enumerate(loader):
        # print(str(batch_idx))
        d1 = {"batch_idx": batch_idx}
        hp.update(d1)

        if args.cuda:
            data = data.cuda()  # , data_y.cuda()
        # (batch, agents, time, feat) => (time, agents, batch, feat)
        data = data.permute(2, 1, 0, 3)  # , data.transpose(0, 1)

        if train == 1:
            batch_losses, batch_losses2 = model(data, rollout, train, hp=hp)
            optimizer.zero_grad()
            total_loss = sum(batch_losses.values())
            total_loss.backward()
            optimizer.step()
        else:
            _, _, _, batch_losses, batch_losses2, prediction = model.sample(
                data, rollout=True, burn_in=hp["burn_in"], L_att=hp["L_att"]
            )
            # x_pre = float(prediction[1][4].item())
            # y_pre = float(prediction[1][5].item())
            # prediction_list.append([x_pre, y_pre])
            # print(prediction_list)
            # writer.add_scalar('test/prediction', x_pre, y_pre)

        for key in batch_losses:
            if batch_idx == 0:
                losses[key] = batch_losses[key].item()
            else:
                losses[key] += batch_losses[key].item()

        for key in batch_losses2:
            if batch_idx == 0:
                losses2[key] = batch_losses2[key].item()
            else:
                losses2[key] += batch_losses2[key].item()

    for key in losses:
        losses[key] /= len(loader.dataset)
    for key in losses2:
        losses2[key] /= len(loader.dataset)
    return losses, losses2


def loss_str(losses):
    ret = ""
    for key in losses:
        if (
            "L" in key
            and not "mac" in key
            and not "vel" in key
            and not "acc" in key
            and not "jrk" in key
        ):
            ret += " {}: {:.0f} |".format(key, losses[key])
        elif "jrk" in key or "vel" in key or "acc" in key:
            ret += " {}: {:.3f} |".format(key, losses[key])
        else:
            ret += " {}: {:.3f} |".format(key, losses[key])
    return ret[:-2]


def run_sanity(args, game_files):
    for j in range(4):
        with open(game_files + str(j) + ".pkl", "rb") as f:
            if j == 0:
                data = np.load(f, allow_pickle=True)[0]
            else:
                tmp = np.load(f, allow_pickle=True)[0]
                data = np.concatenate([data, tmp], axis=1)

    n_agents, batchSize, _, x_dim = data.shape
    n_feat = args.n_feat
    burn_in = args.burn_in
    fs = args.fs
    # nrm = args.normalize
    GT = data.copy()
    losses = {}
    losses["e_pos"] = np.zeros(batchSize)
    losses["e_pmax"] = np.zeros((batchSize, args.horizon - burn_in))
    losses["e_vmax"] = np.zeros((batchSize, args.horizon - burn_in))
    losses["e_amax"] = np.zeros((batchSize, args.horizon - burn_in))
    losses["e_vel"] = np.zeros(batchSize)
    losses["e_acc"] = np.zeros(batchSize)
    losses["L_jrk"] = np.zeros(batchSize)
    next_acc = np.zeros([batchSize, 2])

    assert args.in_sma and args.acc >= 2

    for t in range(args.horizon):
        for i in range(n_agents):
            if args.in_out:
                current_pos = unnormalize(data[i, :, t, 0:2], args)
                current_vel = unnormalize(data[i, :, t, 2:4], args)
                next_pos0 = unnormalize(GT[i, :, t + 1, 0:2], args)
                next_vel0 = unnormalize(GT[i, :, t + 1, 2:4], args)
            elif args.in_sma:
                current_pos = unnormalize(
                    data[i, :, t, n_feat * i + 0 : n_feat * i + 2], args
                )
                current_vel = unnormalize(
                    data[i, :, t, n_feat * i + 2 : n_feat * i + 4], args
                )
                current_acc = unnormalize(
                    data[i, :, t, n_feat * i + 4 : n_feat * i + 6], args
                )
                next_pos0 = unnormalize(
                    GT[i, :, t + 1, n_feat * i + 0 : n_feat * i + 2], args
                )
                next_vel0 = unnormalize(
                    GT[i, :, t + 1, n_feat * i + 2 : n_feat * i + 4], args
                )
                next_acc0 = unnormalize(
                    GT[i, :, t + 1, n_feat * i + 4 : n_feat * i + 6], args
                )
                if t > 0:
                    past_vel = unnormalize(
                        data[i, :, t - 1, n_feat * i + 2 : n_feat * i + 4], args
                    )
            else:
                current_pos = unnormalize(
                    data[i, :, t, n_feat * i + 3 : n_feat * i + 5], args
                )
                current_vel = unnormalize(
                    data[i, :, t, n_feat * i + 5 : n_feat * i + 7], args
                )
                next_pos0 = unnormalize(
                    GT[i, :, t + 1, n_feat * i + 3 : n_feat * i + 5], args
                )
                next_vel0 = unnormalize(
                    GT[i, :, t + 1, n_feat * i + 5 : n_feat * i + 7], args
                )

            losses["L_jrk"] += batch_error(current_acc, next_acc0)
            if t >= burn_in:
                next_pos = current_pos + past_vel * fs
                next_vel = current_vel

                losses["e_pos"] += batch_error(next_pos, next_pos0)
                losses["e_pmax"][:, t - burn_in] += batch_error(next_pos, next_pos0)
                losses["e_vel"] += batch_error(next_vel, next_vel0)
                losses["e_vmax"][:, t - burn_in] += batch_error(next_vel, next_vel0)

                losses["e_acc"] += batch_error(next_acc, next_acc0)
                losses["e_amax"][:, t - burn_in] += batch_error(next_acc, next_acc0)
                if args.in_out:
                    data[i, :, t + 1, :2] = np.concatenate([next_pos], 1)
                elif args.in_sma:
                    data[i, :, t + 1, n_feat * i + 0 : n_feat * i + 2] = np.concatenate(
                        [next_pos], 1
                    )
                else:
                    data[i, :, t + 1, n_feat * i + 3 : n_feat * i + 5] = np.concatenate(
                        [next_pos], 1
                    )
    # del data
    losses["e_pos"] /= args.horizon - burn_in
    losses["e_vel"] /= args.horizon - burn_in
    losses["e_acc"] /= args.horizon - burn_in
    losses["L_jrk"] /= args.horizon
    losses["e_pmax"] = np.max(losses["e_pmax"], 1)
    losses["e_vmax"] = np.max(losses["e_vmax"], 1)
    losses["e_amax"] = np.max(losses["e_amax"], 1)
    avgL2_m = {}
    avgL2_sd = {}

    for key in losses:
        avgL2_m[key] = np.mean(losses[key])
        avgL2_sd[key] = np.std(losses[key])

    print(
        "Velocity (mean & best):"
        + " $"
        + "{:.2f}".format(avgL2_m["e_pos"])
        + " \pm "
        + "{:.2f}".format(avgL2_sd["e_pos"])
        + "$ &"
        + " $"
        + "{:.2f}".format(avgL2_m["e_vel"])
        + " \pm "
        + "{:.2f}".format(avgL2_sd["e_vel"])
        + "$ &"
        + " $"
        + "{:.2f}".format(avgL2_m["e_acc"])
        + " \pm "
        + "{:.2f}".format(avgL2_sd["e_acc"])
        + "$ &"
        + " $"
        + "{:.2f}".format(avgL2_m["L_jrk"])
        + " \pm "
        + "{:.2f}".format(avgL2_sd["L_jrk"])
        + "$ &"
    )
    """print('Velocity (max):'
            +' $' + '{:.2f}'.format(avgL2_m['e_pmax'])+' \pm '+'{:.2f}'.format(avgL2_sd['e_pmax'])+'$ &'
            +' $' + '{:.2f}'.format(avgL2_m['e_vmax'])+' \pm '+'{:.2f}'.format(avgL2_sd['e_vmax'])+'$ &'
            +' $' + '{:.2f}'.format(avgL2_m['e_amax'])+' \pm '+'{:.2f}'.format(avgL2_sd['e_amax'])+'$ &'
            )"""

    losses["e_pos"] = np.mean(losses["e_pos"])
    losses["e_pmax"] = np.mean(losses["e_pmax"])
    losses["e_vel"] = np.mean(losses["e_vel"])
    return losses


def batch_error(predict, true):
    error = np.sqrt(np.sum((predict[:, :2] - true[:, :2]) ** 2, 1))
    return error


def unnormalize(data, args):
    # not used (maybe wrong)!!!
    if args.normalize:
        if args.dataset == "nba":
            feet_m = 0.3048
            LENGTH = 94 * feet_m
            WIDTH = 50 * feet_m
            SHIFT0 = [0, 0]  # [47*feet_m,25*feet_m]
        elif args.dataset == "soccer":
            LENGTH = 52.5
            WIDTH = 34
            SHIFT0 = [0, 0]

        dim = data.ndim
        SEQUENCE_DIMENSION = data.shape[-1]
        NORMALIZE = np.array([LENGTH, WIDTH]) * int(SEQUENCE_DIMENSION / 2)
        SHIFT = SHIFT0 * int(SEQUENCE_DIMENSION / 2)

        if dim == 2:
            NORMALIZE = np.tile(NORMALIZE, (data.shape[0], 1))
        data = np.multiply(data, NORMALIZE)  # + SHIFT

    return data


def read_pickle_data():
    try:
        with open("data/bats/BAT_FLIGHT_TRAIN.pkl", "rb") as f:
            X_train_all = np.load(f, allow_pickle=True)
    except:
        raise FileExistsError("train pickle is not exist.")

    # test pickle load
    try:
        with open("data/bats/BAT_FLIGHT_TEST.pkl", "rb") as f:
            X_test_all = np.load(f, allow_pickle=True)
    except:
        raise FileExistsError("test pickle is not exist.")

    return X_train_all, X_test_all


def modify_train_data(
    train_list, n_roles, ind_train, totalTimeSteps, featurelen, len_seqs
):
    X_all = np.zeros([n_roles, len(ind_train), totalTimeSteps + 4, featurelen])
    X_val_all = np.zeros([n_roles, len(ind_val), totalTimeSteps + 4, featurelen])
    for i, X_train in enumerate(train_list):
        i_tr = 0
        i_val = 0
        for j in range(len_seqs):
            if set([j]).issubset(set(ind_train)):
                for r in range(totalTimeSteps + 4):
                    X_all[i][i_tr][r][:] = np.squeeze(X_train[j][r, :])
                i_tr += 1
            else:
                for r in range(totalTimeSteps + 4):
                    X_val_all[i][i_val][r][:] = np.squeeze(X_train[j][r, :])
                i_val += 1

    return X_all, X_val_all


def modify_test_data(
    test_list, n_roles, len_seqs_test, totalTimeSteps_test, featurelen
):
    X_test_test_all = np.zeros(
        [n_roles, len_seqs_test, totalTimeSteps_test + 4, featurelen]
    )

    for i, X_test in enumerate(test_list):
        i_te = 0
        for j in range(len_seqs_test):
            for r in range(totalTimeSteps_test + 4):
                X_test_test_all[i][j][r][:] = np.squeeze(X_test[j][r, :])

    return X_test_test_all


def make_params(args, n_feat, outputlen0, featurelen, totalTimeSteps):
    temperature = 1
    if not torch.cuda.is_available():
        args.cuda = False
        print("cuda is not used")
    else:
        args.cuda = True
        print("cuda is used")

    ball_dim = 0 if args.acc >= 0 else 2
    args.burn_in = int(totalTimeSteps / 3 * 2)
    args.horizon = totalTimeSteps
    params = {
        "model": args.model,
        "res": args.res,
        "dataset": args.data,
        "x_dim": outputlen0,
        "y_dim": featurelen,
        "z_dim": 64,
        "h_dim": 64,
        "rnn_dim": 100,
        "rnn_att_dim": 32,
        "n_layers": 2,
        "len_seq": totalTimeSteps,
        "n_agents": args.n_roles,
        "normalize": args.normalize,
        "in_out": args.in_out,
        "in_sma": args.in_sma,
        "seed": args.seed,
        "cuda": args.cuda,
        "n_feat": n_feat,
        "fs": fs,
        "embed_size": 32,
        "embed_ball_size": 32,
        "burn_in": args.burn_in,
        "horizon": args.horizon,
        "acc": args.acc,
        "body": args.body,
        "hard_only": args.hard_only,
        "jrk": args.jrk,
        "lam_acc": args.lam_acc,
        "ball_dim": ball_dim,
        "n_all_agents": 1,
        "temperature": temperature,
    }

    return params


if __name__ == "__main__":
    numProcess = args.numProcess
    os.environ["OMP_NUM_THREADS"] = str(numProcess)
    TEST = args.TEST
    # bat用のコードにする
    # シンプルにするために使うパラメータをマジックナンバー的に記入する

    args.meanHMM = True  # sorting sequences using meanHMM
    args.in_sma = True  # True: 2dim, False: 3dim
    acc = 0  # output is vel
    vel_in = 1  # input vel
    args.filter = True
    assert not (args.in_out and args.in_sma)

    global fs
    fs = args.fs
    dt = 1 / fs

    if args.data == "bat":
        n_pl = 1
        subsample_factor = 50 * dt
    else:
        raise FileExistsError("This branch codes only for bat data.")

    args.subsample_factor = subsample_factor
    event_threshold = args.event_threshold
    n_roles = args.n_roles
    n_GorS = args.n_GorS
    val_devide = args.val_devide
    batchSize = args.batchsize
    totalTimeSteps = args.totalTimeSteps
    totalTimeSteps_test = totalTimeSteps
    lidar_dim = 201
    args.n_all_agents = 1

    if args.in_sma:
        outputlen0 = 2
        n_feat = 5 + lidar_dim
    else:
        outputlen0 = 3
        n_feat = 8 + lidar_dim

    X_train_all, X_test_all = read_pickle_data()

    # for train data---------------
    # 入力次元に整形、次の速度を正解データとしてYに整形
    X_train_all, Y_train_all = get_bat_sequence_data(X_train_all, args.in_sma)

    len_seqs = len(X_train_all[0])
    X_ind = np.arange(len_seqs)
    # train dataをtrainとvalidationに分割
    ind_train, ind_val, _, _ = train_test_split(
        X_ind, X_ind, test_size=1 / val_devide, random_state=42
    )

    featurelen = X_train_all[0][0].shape[1]
    len_seqs_tr = len(ind_train)
    offSet_tr = math.floor(len_seqs_tr / batchSize)
    batchSize_val = len(ind_val)
    X_all, X_val_all = modify_train_data(
        X_train_all, n_roles, ind_train, totalTimeSteps, featurelen, len_seqs
    )
    del X_train_all
    print("created train sequences--------------------------------")

    # for test data-----------------
    # 入力次元に整形、次の速度を正解データとしてYに整形
    X_test_all, Y_test_all = get_bat_sequence_data(X_test_all, args.in_sma)

    len_seqs_val = len(X_val_all[0])
    len_seqs_test = len(X_test_all[0])
    batchSize_test = len_seqs_test
    len_seqs_test0 = len_seqs_test
    ind_test = np.arange(len_seqs_test)

    X_test_test_all = modify_test_data(
        X_test_all, n_roles, len_seqs_test, totalTimeSteps_test, featurelen
    )
    print("created test sequences---------------------------------")

    # save to pickle file
    for j in range(offSet_tr):
        tmp_data = X_all[:, j * batchSize : (j + 1) * batchSize, :, :]
        with open(f"data/bats/bat_tr_{j}.pkl", "wb") as f:
            pickle.dump([tmp_data, len_seqs_val, len_seqs_test], f, protocol=5)
    print("saved train sequences---------------------------------")

    J = 2
    batchval = int(len_seqs_val / J)
    for j in range(J):
        if j < J - 1:
            tmp_data = X_val_all[:, j * batchval : (j + 1) * batchval, :, :]
        else:
            tmp_data = X_val_all[:, j * batchval :, :, :]
        with open(f"data/bats/bat_val_{j}.pkl", "wb") as f:
            pickle.dump(tmp_data, f, protocol=5)
    print("saved validation sequences---------------------------------")

    batchte = int(len_seqs_test / J)
    for j in range(J):
        if j < J - 1:
            tmp_data = X_test_test_all[:, j * batchte : (j + 1) * batchte, :, :]
        else:
            tmp_data = X_test_test_all[:, j * batchte :, :, :]
        with open(f"data/bats/bat_te_{j}.pkl", "wb") as f:
            pickle.dump(tmp_data, f, protocol=5)
    print("saved test sequences---------------------------------")

    del X_val_all, X_test_test_all, tmp_data

    # count batches
    offSet_tr = len(glob.glob("data/bats/bat_tr_*.pkl"))
    print(f"train batch num:{offSet_tr}")

    # load train init data
    with open("data/bats/bat_tr_0.pkl", "rb") as f:
        X_all, len_seqs_val, len_seqs_test = np.load(f, allow_pickle=True)
    featurelen = X_all.shape[3]
    len_seqs_tr = batchSize * offSet_tr
    print(
        "featurelen: "
        + str(featurelen)
        + " train_seqs: "
        + str(len_seqs_tr)
        + " val_seqs: "
        + str(len_seqs_val)
        + " test_seqs: "
        + str(len_seqs_test)
    )

    # parameters for VRNN ------------------------------------
    init_file_name0 = f"{path_init}/sub{fs}_bat"
    if not os.path.isdir(init_file_name0):
        os.makedirs(init_file_name0)
    init_pthname = f"{init_file_name0}_state_dict"
    # init_pthname0 = f""
    print(f"model:{init_file_name0}")

    if not os.path.isdir(init_pthname):
        os.makedirs(init_pthname)

    # make params
    params = make_params(args, n_feat, outputlen0, featurelen, totalTimeSteps)
    # Set manual seed
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.cuda:
        torch.cuda.manual_seed(args.seed)

    # Load model
    model = load_model(args.model, params, parser)
    if args.cuda:
        model.cuda()
    params["total_params"] = num_trainable_params(model)
    # Create save path and saving parameters
    pickle.dump(params, open(init_file_name0 + "/params.p", "wb"), protocol=2)

    # Continue a previous experiment, or start a new one
    if args.cont:
        print("Continue")
        if args.pretrain > 0:
            if os.path.exists("{}_best_pretrain.pth".format(init_file_name0)):
                state_dict = torch.load("{}_best_pretrain.pth".format(init_file_name0))
                model.load_state_dict(state_dict)
                print("best pretrain body model was loaded")
            else:
                print("args.cont = True but file did not exist")
        else:
            if os.path.exists("{}_best.pth".format(init_pthname)):
                # state_dict = torch.load('{}_12.pth'.format(init_pthname))
                state_dict = torch.load("{}_best.pth".format(init_pthname))
                model.load_state_dict(state_dict)
                print("best model was loaded")
            else:
                print("args.cont = True but file did not exist")
    else:
        print("New one")
        pass

    # Dataset loaders
    num_workers = 1
    kwargs = {"num_workers": num_workers, "pin_memory": True} if args.cuda else {}
    kwargs2 = {"num_workers": 4, "pin_memory": True} if args.cuda else {}
    print(f"num workers:{num_workers}")
    batchSize_val = len_seqs_val if len_seqs_val <= batchSize else batchSize
    batchSize_test = len_seqs_test if len_seqs_test <= int(batchSize / 2) else batchSize

    if not TEST:
        train_loader = DataLoader(
            GeneralDataset(args, len_seqs_tr, train=1, normalize_data=args.normalize),
            batch_size=batchSize,
            shuffle=False,
            **kwargs,
        )
        val_loader = DataLoader(
            GeneralDataset(args, len_seqs_val, train=0, normalize_data=args.normalize),
            batch_size=batchSize_val,
            shuffle=False,
            **kwargs2,
        )
    test_loader = DataLoader(
        GeneralDataset(args, len_seqs_test, train=-1, normalize_data=args.normalize),
        batch_size=batchSize_test,
        shuffle=False,
        **kwargs2,
    )
    print(f"batch train:{batchSize} val:{batchSize_val} test:{batchSize_test}")

    # train start
    writer = tbx.SummaryWriter()
    best_val_loss = 0
    epochs_since_best = 0
    lr = 1e-3
    epoch_first_best = -1

    pretrain_time = 0
    pretrain2_time = 0
    L_att = False

    save_every = 100

    hyperparams = {
        "model": args.model,
        "acc": acc,
        "burn_in": params["horizon"],
        "L_att": L_att,
        "pretrain": (0 < pretrain_time),
        "pretrain2": (0 < pretrain2_time),
    }
    optimizer = torch.optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()), lr=lr
    )
    if not TEST:
        for e in range(args.n_epoch):
            epoch = e + 1
            print(f"epoch:{epoch}")
            pretrain = epoch <= pretrain_time
            pretrain2 = epoch <= pretrain2_time
            hyperparams["pretrain"] = pretrain
            hyperparams["pretrain2"] = pretrain2

            if epochs_since_best == 5:
                # Load previous best model
                filename = "{}_best.pth".format(init_pthname)
                state_dict = torch.load(filename)
                epochs_since_best = 0
            else:
                if not hyperparams["pretrain"] and not args.finetune:
                    epochs_since_best += 1
            start_time = time.time()
            # train
            # train run
            train_loss, train_loss2 = run_epoch(
                model, train=1, rollout=False, hp=hyperparams
            )
            # print(f"Train:{loss_str(train_loss)} | {loss_str(train_loss2)}")
            print(f"Train:{train_loss} | {train_loss2}")
            # validation run
            val_loss, val_loss2 = run_epoch(
                model, train=0, rollout=True, hp=hyperparams
            )
            print("RO Val:\t" + loss_str(val_loss) + "|" + loss_str(val_loss2))

            total_val_loss = sum(val_loss.values())
            epoch_time = time.time() - start_time
            print("Time:\t {:.3f}".format(epoch_time))
            # for tensorboardX of train
            writer.add_scalars("train/loss for backpropagation", train_loss, epoch)
            writer.add_scalars("train/loss", train_loss2, epoch)
            # for tensorboardX of validation
            writer.add_scalars("val/loss for backpropagation", val_loss, epoch)
            writer.add_scalars("val/loss", val_loss2, epoch)

            #
            if e > epoch_first_best and (
                best_val_loss == 0 or total_val_loss < best_val_loss
            ):
                best_val_loss_prev = best_val_loss
                best_val_loss = total_val_loss
                epochs_since_best = 0

                filename = "{}_best.pth".format(init_pthname)
                torch.save(model.state_dict(), filename)
                print("##### Best model #####")
                if (
                    epoch > pretrain_time
                    and (best_val_loss_prev - best_val_loss) / best_val_loss < 0.0001
                    and best_val_loss_prev != 0
                ):
                    print(
                        "best loss - current loss: "
                        + str(best_val_loss_prev)
                        + " - "
                        + str(best_val_loss)
                    )
                    break
            if epoch % save_every == 0:
                filename = "{}_{}.pth".format(init_pthname, epoch)
                torch.save(model.state_dict(), filename)
                print("########## Saved model ##########")
        print("Best Val Loss: {:.4f}".format(best_val_loss))

    # TEST start
    # Load params
    params = pickle.load(open(init_file_name0 + "/params.p", "rb"))

    # Load model
    state_dict = torch.load(
        "{}_best.pth".format(init_pthname, params["model"]),
        map_location=lambda storage, loc: storage,
    )
    model.load_state_dict(state_dict)
    loader = test_loader
    n_sample = 10
    n_smp_b = 10
    rep_smp = int(n_sample / n_smp_b)
    i = 0
    if True:
        print("test sample")
        samples = [
            np.zeros((args.horizon + 1, args.n_roles, len_seqs_test, featurelen))
            for t in range(n_sample)
        ]
        samples_true = [
            np.zeros((args.horizon + 1, args.n_roles, len_seqs_test, featurelen))
            for t in range(n_sample)
        ]
        hard_att = np.zeros(
            (
                args.horizon,
                args.n_roles,
                len_seqs_test,
                args.n_all_agents + 1,
                n_sample,
            )
        )
        loss_i = [{} for t in range(n_sample)]
        losses = {}
        losses2 = {}

        for r in range(rep_smp):
            start_time = time.time()
            for batch_idx, data in enumerate(loader):
                if args.cuda:
                    data = data.cuda()  # , data_y.cuda()
                    # (batch, agents, time, feat) => (time, agents, batch, feat)
                data = data.permute(2, 1, 0, 3)
                sample, _, _, output, output2, prediction = model.sample(
                    data,
                    rollout=True,
                    burn_in=args.burn_in,
                    L_att=L_att,
                    CF_pred=False,
                    n_sample=n_smp_b,
                    TEST=True,
                )
                for i in range(n_smp_b):
                    sample0 = (
                        sample.detach().cpu().numpy()
                        if n_smp_b == 1
                        else sample[i].detach().cpu().numpy()
                    )
                    data0 = (
                        data.detach().cpu().numpy()
                        if n_smp_b == 1
                        else data.detach().cpu().numpy()
                    )
                    samples[r * n_smp_b + i][
                        :,
                        :,
                        batch_idx * batchSize_test : (batch_idx + 1) * batchSize_test,
                    ] = sample0[:-3]
                    samples_true[r * n_smp_b + i][
                        :,
                        :,
                        batch_idx * batchSize_test : (batch_idx + 1) * batchSize_test,
                    ] = data0[:-3]
                del sample, sample0
                for key in output:
                    if batch_idx == 0 and r == 0:
                        losses[key] = np.zeros(n_sample)
                        losses2[key] = np.zeros((n_sample, len_seqs_test))
                    losses[key][r * n_smp_b : (r + 1) * n_smp_b] += np.sum(
                        output[key].detach().cpu().numpy(), axis=1
                    )
                    losses2[key][
                        r * n_smp_b : (r + 1) * n_smp_b,
                        batch_idx * batchSize_test : (batch_idx + 1) * batchSize_test,
                    ] = (
                        output[key].detach().cpu().numpy()
                    )

                for key in output2:
                    if batch_idx == 0 and r == 0:
                        losses[key] = np.zeros(n_sample)
                        losses2[key] = np.zeros((n_sample, len_seqs_test))
                    losses[key][r * n_smp_b : (r + 1) * n_smp_b] += np.sum(
                        output2[key].detach().cpu().numpy(), axis=1
                    )
                    losses2[key][
                        r * n_smp_b : (r + 1) * n_smp_b,
                        batch_idx * batchSize_test : (batch_idx + 1) * batchSize_test,
                    ] = (
                        output2[key].detach().cpu().numpy()
                    )
            for i in range(n_smp_b):
                for key in losses:
                    loss_i[r * n_smp_b + i][key] = losses[key][r * n_smp_b + i] / len(
                        test_loader.dataset
                    )
                print(
                    "Test sample"
                    + str(r * n_smp_b + i)
                    + ":\t"
                    + loss_str(loss_i[r * n_smp_b + i])
                )
                writer.add_scalars(
                    "test/loss", loss_i[r * n_smp_b + i], r * n_smp_b + i
                )

            epoch_time = time.time() - start_time
            print("Time:\t {:.3f}".format(epoch_time))
        # Save samples
        experiment_path = "{}/experiments/sample".format(init_file_name0)
        if not os.path.exists(experiment_path):
            os.makedirs(experiment_path)
        else:
            pickle.dump(
                [samples, samples_true, hard_att, losses2],
                open(experiment_path + "/samples.p", "wb"),
                protocol=5,
            )
    writer.close()
