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
parser.add_argument("--fs", type=int, default=100)
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


def run_epoch(train, rollout, hp):
    loader = (
        train_loader
        if train == 1
        else val_loader
        if train == 0
        else test_loader
    )

    losses = {}
    losses2 = {}
    for batch_idx, (data, macro_intents) in enumerate(loader):
        # print(str(batch_idx))
        d1 = {"batch_idx": batch_idx}
        hp.update(d1)

        if args.cuda:
            data = data.cuda()  # , data_y.cuda()
            if "MACRO" in args.model:
                macro_intents = macro_intents.cuda()
        # (batch, agents, time, feat) => (time, agents, batch, feat)
        data = data.permute(2, 1, 0, 3)  # , data.transpose(0, 1)
        if "MACRO" in args.model:
            macro_intents = macro_intents.transpose(0, 1)

        if train == 1:
            if "MACRO" in args.model:
                batch_losses, batch_losses2 = model(
                    data, rollout, train, macro_intents, hp=hp
                )
            else:
                batch_losses, batch_losses2 = model(data, rollout, train, hp=hp)
            optimizer.zero_grad()
            total_loss = sum(batch_losses.values())
            total_loss.backward()
            if hp["model"] != "RNN_ATTENTION":
                nn.utils.clip_grad_norm_(model.parameters(), clip)
            optimizer.step()
        else:
            if "MACRO" in args.model:
                if hp["pretrain"]:
                    batch_losses, batch_losses2 = model(
                        data, rollout, train, macro_intents, hp=hp
                    )
                else:
                    (
                        _,
                        _,
                        _,
                        batch_losses,
                        batch_losses2,
                        prediction,
                    ) = model.sample(
                        data,
                        macro_intents,
                        rollout=True,
                        burn_in=hp["burn_in"],
                        L_att=hp["L_att"],
                    )
                    # x_pre = float(prediction[1][4].item())
                    # y_pre = float(prediction[1][5].item())
                    # prediction_list.append([x_pre, y_pre])
                    # print(prediction_list)
                    # writer.add_scalar('test/prediction', x_pre, y_pre)
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
                losses["e_pmax"][:, t - burn_in] += batch_error(
                    next_pos, next_pos0
                )
                losses["e_vel"] += batch_error(next_vel, next_vel0)
                losses["e_vmax"][:, t - burn_in] += batch_error(
                    next_vel, next_vel0
                )

                losses["e_acc"] += batch_error(next_acc, next_acc0)
                losses["e_amax"][:, t - burn_in] += batch_error(
                    next_acc, next_acc0
                )
                if args.in_out:
                    data[i, :, t + 1, :2] = np.concatenate([next_pos], 1)
                elif args.in_sma:
                    data[
                        i, :, t + 1, n_feat * i + 0 : n_feat * i + 2
                    ] = np.concatenate([next_pos], 1)
                else:
                    data[
                        i, :, t + 1, n_feat * i + 3 : n_feat * i + 5
                    ] = np.concatenate([next_pos], 1)
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


def label_macro_intents(data, window_size=0):
    """Computes and saves labeling functions for basketball.
    Args:
        window_size (int): If positive, will label macro-intents every window_size timesteps.
                            Otherwise, will label stationary positions as macro-intents.
    """

    N_AGENTS, N, SEQUENCE_LENGTH, SEQUENCE_DIMENSION = data.shape
    n_all_agents = 10 if N_AGENTS == 5 else 22
    n_feat = int((SEQUENCE_DIMENSION - 4) / n_all_agents)

    # Compute macro-intents
    macro_intents_all = np.zeros(
        (N, SEQUENCE_LENGTH, N_AGENTS)
    )  # data.shape[1]

    for i in range(N):
        for k in range(N_AGENTS):
            if n_feat < 10:
                data_in = data[0, i, :, 2 * k : 2 * k + 2]
            else:
                data_in = data[0, i, :, 2 * k + 3 : 2 * k + 5]
            if window_size > 0:
                macro_intents_all[i, :, k] = compute_macro_intents_fixed(
                    data_in, N_AGENTS, window=window_size
                )
            else:
                macro_intents_all[i, :, k] = compute_macro_intents_stationary(
                    data_in, N_AGENTS
                )

    return macro_intents_all


def compute_macro_intents_stationary(track, N_AGENTS):
    """Computes macro-intents as next stationary points in the trajectory."""

    SPEED_THRESHOLD = 0.5 * 0.3048

    velocity = track[1:, :] - track[:-1, :]
    speed = np.linalg.norm(velocity, axis=-1)
    stationary = speed < SPEED_THRESHOLD
    # assume last frame always stationary
    stationary = np.append(stationary, True)

    T = len(track)
    macro_intents = np.zeros(T)
    for t in reversed(range(T)):
        if t + 1 == T:  # assume position in last frame is always a macro intent
            macro_intents[t] = get_macro_intent(track[t], N_AGENTS, t)
        # from stationary to moving indicated a change in macro intent
        elif stationary[t] and not stationary[t + 1]:
            macro_intents[t] = get_macro_intent(track[t], N_AGENTS, t)
        else:  # otherwise, macro intent is the same
            macro_intents[t] = macro_intents[t + 1]
    return macro_intents


def get_macro_intent(position, N_AGENTS, t):
    """Computes the macro-intent index."""
    N_MACRO_X = 9 if N_AGENTS == 5 else 17  # 26#34# # 105m/2/3
    N_MACRO_Y = 10 if N_AGENTS == 5 else 11  # 17#22# # 68m/2/3
    MACRO_SIZE = 50 * 0.3048 / N_MACRO_Y if N_AGENTS == 5 else 34 / N_MACRO_Y

    eps = 1e-4  # hack to make calculating macro_x and macro_y cleaner

    if N_AGENTS == 5:
        x = bound(position[0], 0, N_MACRO_X * MACRO_SIZE - eps)
        y = bound(position[1], 0, N_MACRO_Y * MACRO_SIZE - eps)
        macro_x = int(x / MACRO_SIZE)
        macro_y = int(y / MACRO_SIZE)
        macro = macro_x * N_MACRO_Y + macro_y
    else:
        x = bound(
            position[0],
            -N_MACRO_X * MACRO_SIZE + eps,
            N_MACRO_X * MACRO_SIZE - eps,
        )
        y = bound(
            position[1],
            -N_MACRO_Y * MACRO_SIZE + eps,
            N_MACRO_Y * MACRO_SIZE - eps,
        )
        macro_x = int(x / MACRO_SIZE) + N_MACRO_X
        macro_y = int(y / MACRO_SIZE) + N_MACRO_Y
        macro = macro_x * N_MACRO_Y * 2 + macro_y
        # if np.isnan(macro) or macro < 0:
    return macro


def bound(val, lower, upper):
    """Clamps val between lower and upper."""
    if val < lower:
        return lower
    elif val > upper:
        return upper
    else:
        return val


if __name__ == "__main__":
    numProcess = args.numProcess
    os.environ["OMP_NUM_THREADS"] = str(numProcess)
    TEST = args.TEST
    args.meanHMM = True
    args.in_sma = True
    acc = args.acc
    args.vel_in = 1 if args.vel_in else 2
    if acc == -1:
        args.vel_in = -1
    elif acc == 0 or acc == 1:
        args.vel_in = 1
    vel_in = args.vel_in
    args.velocity = args.vel_in
    args.filter = True
    assert not (args.in_out and args.in_sma)
    assert not (args.vel_in == 1 and acc >= 2)
    
    all_games_id = [
        i.split(os.sep)[-1].split(".")[0]
        for i in glob.glob(game_dir + "/*.pkl")
    ]
    global fs
    fs = 1 / args.fs
    if args.data == "bat":
        n_pl = 1
        subsample_factor = 50 * fs
    
    args.subsample_factor = subsample_factor
    event_thresholod = args.event_threshold
    n_roles = args.n_roles
    n_GorS = args.n_Gors
    val_devide = args.val_devide
    batchSize = args.batchsize
    overlapWindow = args.overlap
    totalTimeSteps = args.totalTimeSteps
    
    game_file0 = "./data/all_" + args.data + "_games_" + str(n_GorS) + "_"
    game_file0 = (game_file0 + "unnorm" if not args.normalize else game_file0 + "norm")
    
    game_file0 = game_file0 + "_filt"
    game_file0 = game_file0 + "_acc"
    k_nearest = args.k_nearest
    if k_nearest == 0:
        game_file0 = game_file0 + "_k0"
    
    if args.meanHMM:
        game_file0 = game_file0 + "_meanHMM"
    
    game_file0 = game_file0 + "/"
    if not os.path.isdir(game_file0):
        os.makedirs(game_file0)
    
    game_files_pre = game_file0 + "_pre"
    
    game_file0 = game_file0 + "Fs" + str(args.fs)
    
    if acc == -1:
        game_file0 = game_file0 + "_pos"
    if args.vel_in == 1:
        game_file0 = game_file0 + "_vel"
    if args.in_sma:
        game_file0 = game_file0 + "_inSimple"
    elif args.in_out:
        game_file0 = game_file0 + "_inout"
    
    game_file0 = game_file0 + "_" + str(batchSize) + "_" + str(totalTimeSteps)
    print(game_file0)
    game_files = game_file0
    game_files_val = game_file0 + "_val" + ".pkl"
    game_files_te = game_file0 + "_te" + ".pkl"
    
    activeRoleInd = range(n_roles) #### 1
    activeRole = []
    activeRole.extend([str(n) for n in range(n_roles)])
    
    if acc == 0 or acc == -1 or acc == 4:
        if args.in_sma: ### in_smaで2次元か３次元かを設定
            outputlen0 = 2 ###出力：２次元速度
        else:
            outputlen0 = 3
    elif acc == 3:
        outputlen0 = 6
    else:
        outputlen0 = 4
    
    numOfPrevSteps = 1
    totalTimeSteps_test = totalTimeSteps
    states_num = 251 ###入力：状態の次元
    delete_num = 2 ###EnvとBatだけ削除
    if args.in_sma:
        # [X, Y, Vx, Vy, theta, pulse_flag, Env, Bat, States]
        n_feat = 8 + states_num
    else:
        # [X, Y, Z, Vx, Vy, Vz, theta, pulse_flag, Env, Bat, States]
        n_feat = 10 + states_num
    
    try:
        with open(
            os.path.dirname(game_files) + "/bats/train.pkl",
            "rb"
        ) as f:
            X_train_all = np.load(f, allow_pickle=True) ###pklファイルの全データ
    except:
        raise FileExistsError("train pkl is not exist")
    
    try:
        with open(
            os.path.dirname(game_files) + "/bats/test.pkl",
            "rb"
        ) as f:
            X_test_all = np.load(f, allow_pickle=True)
    except:
        raise FileExistsError("test pkl is not exist")
    
    X_train_all, Y_train_all = get_bat_sequence_data(
        X_train_all, args.in_sma
    ) ### X_train_all：全データ，Y_train_all：２次元速度
    
    len_seqs = len(X_test_all[0]) ###エピソード数？
    X_ind = np.arange(len_seqs)
    ind_train, ind_val, _, _ = train_test_split(
        X_ind, X_ind, test_size=1/val_devide, random_state=42
    ) ### trainとvalidationに与えたindex
    
    featurelen = X_train_all[0][0].shape[1] ###特徴量の数（action＋state）
    len_seqs_tr = len(ind_train) ###trainのエピソード数
    offSet_tr = math.floor(len_seqs_tr / batchSize) ###小数点以下切り捨て，１バッチに対するエピソード数
    batchSize_val = len(ind_val) ###validationのエピソード数：validationのバッチサイズ？
    
    X_all = np.zeros([n_roles, len(ind_train), totalTimeSteps + 4, featurelen])
    X_val_all = np.zeros([
        n_roles, len(ind_val), totalTimeSteps + 4, featurelen
    ])
    for i, X_train in enumerate(X_train_all):
        i_tr = 0
        i_val = 0
        for b in range(len_seqs):
            if set([b]).issubset(set(ind_train)): ###ここでtrainかvalidationをする、エピソードの集合にtrainのインデックスが含まれていたら
                for r in range(totalTimeSteps + 4):
                    X_all[i][i_tr][r][:] = np.squeeze(X_train[b][r, :]) ###train用のデータに格納
                    i_tr += 1
            else: ###エピソードの集合にvalidationのインデックスが含まれていたら
                for r in range(totalTimeSteps + 4):
                    X_val_all[i][i_val][r][:] = np.squeeze(X_train[b][r, :]) ###validation用のデータに格納
                i_val += 1
    
    print("create train sequences")
    
    del X_train_all
    
    macro_intents = label_macro_intents(X_all)
    macro_intents_val = label_macro_intents(X_val_all)
    
    X_test_all, Y_test_all = get_bat_sequence_data(X_test_all, args.in_sma)
    
    if args.in_out:
        X_test_test_all = Y_test_all ### test用のデータと正解データに分ける
    
    len_seqs_val = len(X_val_all[0])
    len_seqs_test = len(X_test_all[0])
    batchSize_test = len_seqs_test
    len_seqs_test0 = len_seqs_test
    ind_test = np.arange(len_seqs_test)
    
    X_test_test_all = np.zeros(
        [n_roles, len_seqs_test, totalTimeSteps + 4, featurelen]
    )
    for i, X_test in enumerate(X_test_all):
        i_te = 0
        for b in range(len_seqs_test0):
            if args.data == "nba":
                if set([b]).issubset(set(ind_test)):
                    for r in range(totalTimeSteps + 4):
                        X_test_test_all[i][i_te][r][:] = np.squeeze(X_test[b][r, ;])
                    i_te += 1
            elif args.data == "soccer":
                for r in range(totalTimeSteps_test + 4):
                    X_test_test_all[i][b][r][:] = np.squeeze(X_test[b][r, :])
            elif args.data == "bat":
                for r in range(TotalTimeSteps_test + 4):
                    X_test_test_all[i][b][r][:] = np.squeeze(X_test[b][r, :]) ###np.squeezeで１の次元を削除する
    
    print("create test sequences")
    
    for j in range(offSet_tr): ###offSet_tr:１バッチに対するエピソード数
        tmp_data = X_all[:, j * batchSize : (j + 1) * batchSize, :, :]
        tmp_label = macro_intents[j * batchSize : (j + 1) * batchSize, :, :]
        