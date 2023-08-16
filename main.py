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
parser.add_argument("-t_step", "--totalTimeSteps", type=int, default=356)
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
    loader = train_loader if train == 1 else val_loader if train == 0 else test_loader

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
                    _, _, _, batch_losses, batch_losses2, prediction = model.sample(
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
    macro_intents_all = np.zeros((N, SEQUENCE_LENGTH, N_AGENTS))  # data.shape[1]

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
            position[0], -N_MACRO_X * MACRO_SIZE + eps, N_MACRO_X * MACRO_SIZE - eps
        )
        y = bound(
            position[1], -N_MACRO_Y * MACRO_SIZE + eps, N_MACRO_Y * MACRO_SIZE - eps
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
    # pre-process----------------------------------------------
    args.meanHMM = True  # sorting sequences using meanHMM
    args.in_sma = True  # small multi-agent data
    # normalize = False
    acc = args.acc  # output: 0: vel, 1: pos+vel, 2:vel+acc, 3: pos+vel+acc
    args.vel_in = 1 if args.vel_in else 2  # input 1: vel 2: vel+acc
    if acc == -1:
        args.vel_in = -1  # position only
    elif acc == 0 or acc == 1:
        args.vel_in = 1
    vel_in = args.vel_in
    args.velocity = args.vel_in
    # args.hmm_iter = 500
    args.filter = True
    assert not (args.in_out and args.in_sma)
    assert not (args.vel_in == 1 and acc >= 2)

    # all game ids file name, note that '/' or '\\' depends on the environment
    all_games_id = [
        i.split(os.sep)[-1].split(".")[0] for i in glob.glob(game_dir + "/*.pkl")
    ]
    global fs
    fs = 1 / args.fs
    if args.data == "nba":
        n_pl = 5
        subsample_factor = 25 * fs
    elif args.data == "soccer":
        n_pl = 11
        subsample_factor = 10 * fs
    elif args.data == "bat":
        n_pl = 1
        # note default fs is 60 Hz
        subsample_factor = 50 * fs

    args.subsample_factor = subsample_factor
    event_threshold = args.event_threshold
    n_roles = args.n_roles
    n_GorS = args.n_GorS  # games if NBA and seqs if soccer
    val_devide = args.val_devide
    batchSize = args.batchsize
    overlapWindow = args.overlap
    totalTimeSteps = args.totalTimeSteps

    # save the processed file to disk to avoid repeated work
    game_file0 = "./data/all_" + args.data + "_games_" + str(n_GorS) + "_"
    game_file0 = game_file0 + "unnorm" if not args.normalize else game_file0 + "norm"

    game_file0 = game_file0 + "_filt"
    game_file0 = game_file0 + "_acc"
    k_nearest = args.k_nearest  # 3
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
    # if args.normalize:
    #    game_file0 = game_file0 + '_norm'
    game_file0 = game_file0 + "_" + str(batchSize) + "_" + str(totalTimeSteps)
    print(game_file0)
    game_files = game_file0
    game_files_val = game_file0 + "_val" + ".pkl"
    game_files_te = game_file0 + "_te" + ".pkl"

    activeRoleInd = range(n_roles)
    activeRole = []
    activeRole.extend([str(n) for n in range(n_roles)])  # need to be reconsidered

    if acc == 0 or acc == -1 or acc == 4:  # vel/pos/acc only
        if args.in_sma:
            outputlen0 = 2
        else:
            outputlen0 = 3
    elif acc == 3:  # all
        outputlen0 = 6
    else:
        outputlen0 = 4

    # We are only looking at the most recent character each time.
    numOfPrevSteps = 1
    totalTimeSteps_test = totalTimeSteps
    if args.in_sma:
        n_feat = 7
        # n_feat = 6 if vel_in == 2 else 4
        # if acc == -1:
        #     n_feat = 2
    else:
        n_feat = 10
    # elif args.in_out:
    #     n_feat = 6 if vel_in == 2 else 4
    # else:
    #     n_feat = 15 if vel_in == 2 else 13

    # train pickle load
    try:
        with open(
            os.path.dirname(game_files) + "/bats/BAT_FLIGHT_TRAIN.pkl", "rb"
        ) as f:
            X_train_all = np.load(f, allow_pickle=True)
    except:
        raise FileExistsError("train pickle is not exist.")

    # test pickle load
    try:
        with open(os.path.dirname(game_files) + "/bats/BAT_FLIGHT_TEST.pkl", "rb") as f:
            X_test_all = np.load(f, allow_pickle=True)
    except:
        raise FileExistsError("test pickle is not exist.")

    X_train_all, Y_train_all = get_bat_sequence_data(
        X_train_all, args.in_sma
    )  # [role][seqs][steps,feats]

    len_seqs = len(X_train_all[0])
    X_ind = np.arange(len_seqs)
    ind_train, ind_val, _, _ = train_test_split(
        X_ind, X_ind, test_size=1 / val_devide, random_state=42
    )

    featurelen = X_train_all[0][0].shape[1]
    len_seqs_tr = len(ind_train)
    # print(len_seqs_tr)
    offSet_tr = math.floor(len_seqs_tr / batchSize)
    batchSize_val = len(ind_val)

    X_all = np.zeros([n_roles, len(ind_train), totalTimeSteps + 4, featurelen])
    X_val_all = np.zeros([n_roles, len(ind_val), totalTimeSteps + 4, featurelen])
    for i, X_train in enumerate(X_train_all):
        i_tr = 0
        i_val = 0
        for b in range(len_seqs):
            if set([b]).issubset(set(ind_train)):
                for r in range(totalTimeSteps + 4):
                    X_all[i][i_tr][r][:] = np.squeeze(X_train[b][r, :])
                i_tr += 1
            else:
                for r in range(totalTimeSteps + 4):
                    X_val_all[i][i_val][r][:] = np.squeeze(X_train[b][r, :])
                i_val += 1

    print("create train sequences")

    del X_train_all

    # macro intents
    macro_intents = label_macro_intents(X_all)
    macro_intents_val = label_macro_intents(X_val_all)

    # for test data-------------
    X_test_all, Y_test_all = get_bat_sequence_data(X_test_all, args.in_sma)

    if args.in_out:
        X_test_test_all = Y_test_all

    len_seqs_val = len(X_val_all[0])
    len_seqs_test = len(X_test_all[0])
    batchSize_test = len_seqs_test  # args.batchsize # 32
    len_seqs_test0 = len_seqs_test
    ind_test = np.arange(len_seqs_test)

    # if args.data == 'nba':
    #     X_ind = np.arange(len_seqs_test)
    #     _, ind_test, _, _ = train_test_split(
    #         X_ind, X_ind, test_size=1/3, random_state=42)
    #     len_seqs_test = len(ind_test)

    X_test_test_all = np.zeros(
        [n_roles, len_seqs_test, totalTimeSteps_test + 4, featurelen]
    )
    for i, X_test in enumerate(X_test_all):
        i_te = 0
        for b in range(len_seqs_test0):
            if args.data == "nba":
                if set([b]).issubset(set(ind_test)):
                    for r in range(totalTimeSteps + 4):
                        X_test_test_all[i][i_te][r][:] = np.squeeze(X_test[b][r, :])
                    i_te += 1
            elif args.data == "soccer":
                for r in range(totalTimeSteps_test + 4):
                    X_test_test_all[i][b][r][:] = np.squeeze(X_test[b][r, :])
            elif args.data == "bat":
                for r in range(totalTimeSteps_test + 4):
                    X_test_test_all[i][b][r][:] = np.squeeze(X_test[b][r, :])

    print("create test sequences")
    # if offSet_tr > 0:
    # print(offSet_tr)
    for j in range(offSet_tr):
        tmp_data = X_all[:, j * batchSize : (j + 1) * batchSize, :, :]
        tmp_label = macro_intents[j * batchSize : (j + 1) * batchSize, :, :]
        with open(game_files + "_tr" + str(j) + ".pkl", "wb") as f:
            pickle.dump(
                [tmp_data, len_seqs_val, len_seqs_test, tmp_label], f, protocol=4
            )

    J = 2
    batchval = int(len_seqs_val / J)
    for j in range(J):
        if j < J - 1:
            tmp_data = X_val_all[:, j * batchval : (j + 1) * batchval, :, :]
            tmp_label = macro_intents_val[j * batchval : (j + 1) * batchval, :, :]
        else:
            tmp_data = X_val_all[:, j * batchval :, :, :]
            tmp_label = macro_intents_val[j * batchval :, :, :]
        with open(game_files + "_val_" + str(j) + ".pkl", "wb") as f:
            pickle.dump([tmp_data, tmp_label], f, protocol=4)
    # with open(game_files_val, 'wb') as f:
    #    pickle.dump([X_val_all,macro_intents_val], f, protocol=4)

    macro_intents_te = label_macro_intents(X_test_test_all)
    batchte = int(len_seqs_test / J)
    for j in range(J):
        if j < J - 1:
            tmp_data = X_test_test_all[:, j * batchte : (j + 1) * batchte, :, :]
            tmp_label = macro_intents_te[j * batchte : (j + 1) * batchte, :, :]
        else:
            tmp_data = X_test_test_all[:, j * batchte :, :, :]
            tmp_label = macro_intents_te[j * batchte :, :, :]
        with open(game_files + "_te_" + str(j) + ".pkl", "wb") as f:
            pickle.dump([tmp_data, tmp_label], f, protocol=4)
    # with open(game_files_te, 'wb') as f:
    #    pickle.dump([X_test_test_all,macro_intents_te], f, protocol=4)

    del X_val_all, X_test_test_all, tmp_data

    print("save train and test sequences")
    with open(game_files + "_tr" + str(0) + ".pkl", "rb") as f:
        X_all, len_seqs_val, len_seqs_test, macro_intents = np.load(
            f, allow_pickle=True
        )

    # count batches
    offSet_tr = len(glob.glob(game_files + "_tr*.pkl"))
    # variables
    featurelen = X_all.shape[3]  # [0][0][0]#see get_sequences in sequencing.py
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

    # parameters for VRNN -----------------------------------
    init_filename0 = path_init + "sub" + str(args.fs) + "_"
    init_filename0 = init_filename0 + "filt_"
    if args.vel_in == 1:
        init_filename0 = init_filename0 + "vel_"
    if args.meanHMM:
        init_filename0 = init_filename0 + "meanHMM_"
    if args.in_sma:
        init_filename0 = init_filename0 + "inSimple_"
    elif args.in_out:
        init_filename0 = init_filename0 + "inout_"
    init_filename0 = init_filename0 + "acc_" + str(args.acc) + "_"
    init_filename0 = (
        init_filename0 + "norm/" if args.normalize else init_filename0 + "unnorm/"
    )
    if args.attention == 3:
        init_filename00 = init_filename0 + args.data + "_att3/"
    else:
        init_filename00 = init_filename0 + args.data + "/"

    init_filename0 = init_filename0 + args.model + "_" + args.data + "/"
    init_filename0 = (
        init_filename0
        + "att_"
        + str(args.attention)
        + "_"
        + str(batchSize)
        + "_"
        + str(totalTimeSteps)
    )
    if args.wo_macro and "MACRO" in args.model:
        init_filename0 = init_filename0 + "_wo_macro"
    if args.drop_ind:
        init_filename0 = init_filename0 + "_drop_ind"
    init_filename000 = init_filename0
    if args.body:
        init_filename0 = init_filename0 + "_body"
    if args.jrk > 0:
        init_filename0 = init_filename0 + "_jrk"
    if args.lam_acc > 0:
        init_filename0 = init_filename0 + "_lacc"
    if args.finetune:
        init_filename0 = init_filename0 + "_finetune"
    if args.res:
        init_filename0 = init_filename0 + "_res"

    # if args.hard_only and args.attention == 3:
    #    init_filename0 = init_filename0 + '_hard_only'

    if not os.path.isdir(init_filename0):
        os.makedirs(init_filename0)
    init_pthname = "{}_state_dict".format(init_filename0)
    init_pthname0 = "{}_state_dict".format(init_filename00)
    print("model: " + init_filename0)

    if not os.path.isdir(init_pthname):
        os.makedirs(init_pthname)
    if not os.path.isdir(init_pthname0):
        os.makedirs(init_pthname0)

    if args.n_GorS == 7500 and args.data == "soccer":
        batchSize = int(batchSize / 2)
    # args.hard_only = True
    args.dataset = args.data
    args.n_feat = n_feat
    args.fs = fs
    args.game_files = game_files
    args.game_files_val = game_files_val
    args.game_files_te = game_files_te
    args.start_lr = 1e-3
    args.min_lr = 1e-3
    clip = True  # gradient clipping
    args.seed = 200
    save_every = 100
    args.batch_size = batchSize
    # args.normalize = normalize # default: False
    # args.cont = False # continue training previous best model
    args.x_dim = outputlen0  # output
    args.y_dim = featurelen  # input
    args.m_dim = 90 if args.data == "nba" else 34 * 22  # 26*17*4#34*22*4
    args.z_dim = 64
    args.h_dim = 64  # 128
    args.rnn_dim = 100  # 100n
    args.n_layers = 2
    args.rnn_micro_dim = args.rnn_dim
    args.rnn_macro_dim = 100
    args.burn_in = int(totalTimeSteps / 3 * 2)  # 予測に使う長さ
    args.horizon = totalTimeSteps
    args.n_agents = len(activeRole)
    if args.data == "soccer":
        args.n_all_agents = 22
    elif args.data == "bat":
        args.n_all_agents = 1
    else:
        args.n_all_agents = 10
    # args.n_all_agents = 22 if args.data == 'soccer' else 10
    if not torch.cuda.is_available():
        args.cuda = False
        print("cuda is not used")
    else:
        args.cuda = True
        print("cuda is used")
    ball_dim = 0 if acc >= 0 else 2
    """if args.data == 'nba':
        ball_dim = 7 if acc else 5 
    elif args.data == 'soccer':    
        ball_dim = 6 if acc else 4"""
    # Parameters to save
    pretrain2_time = args.pretrain2 if args.body else 0
    args.pretrain2 = pretrain2_time
    temperature = 1 if args.data == "soccer" else 1

    params = {
        "model": args.model,
        "attention": args.attention,
        "wo_macro": args.wo_macro,
        "res": args.res,
        "dataset": args.dataset,
        "x_dim": args.x_dim,
        "y_dim": args.y_dim,
        "z_dim": args.z_dim,
        "h_dim": args.h_dim,
        "m_dim": args.m_dim,
        "rnn_dim": args.rnn_dim,
        "rnn_att_dim": 32,
        "n_layers": args.n_layers,
        "len_seq": totalTimeSteps,
        "generative": False,
        "n_agents": args.n_agents,
        "min_lr": args.min_lr,
        "start_lr": args.start_lr,
        "normalize": args.normalize,
        "in_out": args.in_out,
        "in_sma": args.in_sma,
        "seed": args.seed,
        "cuda": args.cuda,
        "n_feat": n_feat,
        "fs": fs,
        "embed_size": 32,  # 8
        "embed_ball_size": 32,  # 8
        "burn_in": args.burn_in,
        "horizon": args.horizon,
        "rnn_micro_dim": args.rnn_micro_dim,
        "rnn_macro_dim": args.rnn_macro_dim,
        "acc": acc,
        "body": args.body,
        "hard_only": args.hard_only,
        "jrk": args.jrk,
        "lam_acc": args.lam_acc,
        "ball_dim": ball_dim,
        "n_all_agents": args.n_all_agents,
        "temperature": temperature,
        "drop_ind": args.drop_ind,
        "pretrain2": args.pretrain2,
        "init_pthname0": init_pthname0,
    }

    # 'pretrain' : args.pretrain,

    prediction_list = []

    # Set manual seed
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.cuda:
        torch.cuda.manual_seed(args.seed)

    ####### Sanity check ##################
    if args.Sanity:
        losses = run_sanity(args, game_files + "_te_")

    # Load model

    model = load_model(args.model, params, parser)

    if args.cuda:
        model.cuda()
    # Update params with model parameters
    params = model.params
    params["total_params"] = num_trainable_params(model)

    # Create save path and saving parameters
    pickle.dump(params, open(init_filename0 + "/params.p", "wb"), protocol=2)

    # Continue a previous experiment, or start a new one
    if args.cont:
        print("args.cont = True")
        if "MACRO" in args.model and args.pretrain > 0:
            if os.path.exists("{}_best_pretrain.pth".format(init_pthname0)):
                state_dict = torch.load("{}_best_pretrain.pth".format(init_pthname0))
                model.load_state_dict(state_dict)
                print("best pretrain model was loaded")
            else:
                print("args.cont = True but file did not exist")

        elif args.pretrain2 > 0:
            if os.path.exists("{}_best_pretrain2.pth".format(init_pthname0)):
                state_dict = torch.load("{}_best_pretrain2.pth".format(init_pthname0))
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
        print("args.cont = False")
        if "MACRO" in args.model and not args.wo_macro and args.pretrain == 0:
            # https://discuss.pytorch.org/t/how-to-transfer-learned-weight-in-the-same-model-without-last-layer/32824
            pretrained_dict = torch.load("{}_best_pretrain.pth".format(init_pthname0))
            model_dict = model.state_dict()
            pretrained_list = list(pretrained_dict.items())
            # 1. filter out unnecessary keys
            pretrained_dict = {k: v for k, v in pretrained_list[:20] if k in model_dict}
            # 2. overwrite entries in the existing state dict
            model_dict.update(pretrained_dict)
            # 3. load the new state dict
            model.load_state_dict(model_dict)
            print("pretrained model was loaded")
        if args.finetune:  # args.pretrain2 == 0 and args.body:
            # this did not work well
            pretrained_dict = torch.load(
                "{}_state_dict_best.pth".format(init_filename000)
            )  # _pretrain2
            model_dict = model.state_dict()
            pretrained_list = list(pretrained_dict.items())
            model_dict.update(pretrained_dict)
            model.load_state_dict(model_dict)
            lt = 14 if args.wo_macro else 15  # 14-16: decoder
            lt2 = 17 if args.wo_macro else 18  # 17: microRNN
            cntr = 0
            for child in model.children():
                cntr += 1
                if cntr < lt or cntr > lt2:
                    # print(str(cntr))
                    # print(child)
                    for param in child.parameters():
                        param.requires_grad = False
            print("pretrained model2 was loaded")

    print("############################################################")

    # Dataset loaders
    num_workers = 1  # int(args.numProcess/2)
    kwargs = {"num_workers": num_workers, "pin_memory": True} if args.cuda else {}
    kwargs2 = {"num_workers": 4, "pin_memory": True} if args.cuda else {}
    print("num_workers:" + str(num_workers))
    batchSize_val = len_seqs_val if len_seqs_val <= batchSize else batchSize
    batchSize_test = (
        len_seqs_test if len_seqs_test <= int(batchSize / 2) else batchSize
    )  # int(/4)
    if args.n_GorS == 7500 and args.dataset == "soccer":
        batchSize_val = int(batchSize / 4 * 3)
        batchSize_test = 128
        if "MACRO" in args.model and (not args.wo_macro or args.attention == 3):
            batchSize_test = 80
            batchSize_val = 80
        elif "MACRO" in args.model and (not args.wo_macro and args.attention == 3):
            batchSize_test = 64
            batchSize_val = 64
        if "MACRO" in args.model and not args.wo_macro:
            batchSize = int(batchSize / 4 * 3)
        if args.attention == 3:
            batchSize_val = 80  # int(batchSize_val/4*3)
    elif args.n_GorS >= 50 and args.dataset == "nba":
        batchSize_test = int(batchSize / 8)
        # if args.attention == 3:
        #    batchSize_test = int(batchSize_test/4*3)
    if not TEST:
        train_loader = DataLoader(
            GeneralDataset(args, len_seqs_tr, train=1, normalize_data=args.normalize),
            batch_size=batchSize,
            shuffle=False,
            **kwargs
        )
        val_loader = DataLoader(
            GeneralDataset(args, len_seqs_val, train=0, normalize_data=args.normalize),
            batch_size=batchSize_val,
            shuffle=False,
            **kwargs2
        )
    test_loader = DataLoader(
        GeneralDataset(args, len_seqs_test, train=-1, normalize_data=args.normalize),
        batch_size=batchSize_test,
        shuffle=False,
        **kwargs2
    )
    print(
        "batch train: "
        + str(batchSize)
        + " val:"
        + str(batchSize_val)
        + " test: "
        + str(batchSize_test)
    )

    ###### TRAIN LOOP ##############
    writer = tbx.SummaryWriter()
    best_val_loss = 0
    epochs_since_best = 0
    lr = max(args.start_lr, args.min_lr)  # if not args.finetune else 1e-4
    epoch_first_best = -1
    # print('epoch_first_best: '+str(epoch_first_best))

    pretrain_time = args.pretrain if "MACRO" in args.model else 0

    L_att = False
    # hyperparams = {'model': args.model,'acc': acc,'burn_in': args.burn_in}
    hyperparams = {
        "model": args.model,
        "acc": acc,
        "burn_in": args.horizon,
        "L_att": L_att,
        "pretrain": (0 < pretrain_time),
        "pretrain2": (0 < pretrain2_time),
    }

    if not TEST:
        for e in range(args.n_epoch):
            epoch = e + 1
            print("epoch " + str(epoch))
            pretrain = epoch <= pretrain_time
            pretrain2 = epoch <= pretrain2_time
            hyperparams["pretrain"] = pretrain
            hyperparams["pretrain2"] = pretrain2

            # Set a custom learning rate schedule
            if epochs_since_best == 5:  # and lr > args.min_lr:
                # Load previous best model
                filename = "{}_best.pth".format(init_pthname)
                if epoch <= pretrain_time:
                    filename = "{}_best_pretrain.pth".format(init_pthname0)
                elif epoch <= pretrain_time + pretrain2_time:
                    filename = "{}_best_pretrain2.pth".format(init_pthname)

                state_dict = torch.load(filename)

                # Decrease learning rate
                # lr = max(lr/3, args.min_lr)
                # print('########## lr {} ##########'.format(lr))
                epochs_since_best = 0
            else:
                if not hyperparams["pretrain"] and not args.finetune:
                    # lr = lr*0.99 # 9
                    print("########## lr {:.4e} ##########".format(lr))
                    epochs_since_best += 1

            # Remove parameters with requires_grad=False (https://github.com/pytorch/pytorch/issues/679)
            if not args.finetune:
                optimizer = torch.optim.Adam(
                    filter(lambda p: p.requires_grad, model.parameters()), lr=lr
                )
            else:
                optimizer = torch.optim.SGD(
                    filter(lambda p: p.requires_grad, model.parameters()),
                    lr=5e-4,
                    momentum=0.9,
                )
            start_time = time.time()

            print(
                "pretrain:"
                + str(hyperparams["pretrain"])
                + " pretrain2:"
                + str(hyperparams["pretrain2"])
                + " L_att:"
                + str(L_att)
            )
            hyperparams["burn_in"] = args.horizon
            hyperparams["L_att"] = L_att
            # hyperparams = {'model': args.model,'acc': acc,'burn_in': args.horizon,'L_att':L_att}
            train_loss, train_loss2 = run_epoch(train=1, rollout=False, hp=hyperparams)
            print("Train:\t" + loss_str(train_loss) + "|" + loss_str(train_loss2))

            if not hyperparams["pretrain"]:  # epoch % 5 == 3:
                hyperparams["burn_in"] = args.burn_in
                # hyperparams = {'model': args.model,'acc': acc,'burn_in': args.burn_in,'L_att':L_att}
                val_loss, val_loss2 = run_epoch(train=0, rollout=True, hp=hyperparams)
                print("RO Val:\t" + loss_str(val_loss) + "|" + loss_str(val_loss2))

            else:
                hyperparams["burn_in"] = args.horizon
                val_loss, val_loss2 = run_epoch(train=0, rollout=False, hp=hyperparams)
                print("Val:\t" + loss_str(val_loss) + "|" + loss_str(val_loss2))

            total_val_loss = sum(val_loss.values())

            epoch_time = time.time() - start_time
            print("Time:\t {:.3f}".format(epoch_time))

            # for tensorboardX of train
            writer.add_scalars("train/loss for backpropagation", train_loss, epoch)
            writer.add_scalars("train/loss", train_loss2, epoch)
            # for tensorboardX of validation
            writer.add_scalars("val/loss for backpropagation", val_loss, epoch)
            writer.add_scalars("val/loss", val_loss2, epoch)

            # Best model on test set
            if e > epoch_first_best and (
                best_val_loss == 0 or total_val_loss < best_val_loss
            ):
                best_val_loss_prev = best_val_loss
                best_val_loss = total_val_loss
                epochs_since_best = 0

                filename = "{}_best.pth".format(init_pthname)
                if epoch <= pretrain_time:
                    filename = "{}_best_pretrain.pth".format(init_pthname0)
                elif epoch <= pretrain_time + pretrain2_time:
                    filename = "{}_best_pretrain2.pth".format(init_pthname)

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

            # Periodically save model
            if epoch % save_every == 0:
                filename = "{}_{}.pth".format(init_pthname, epoch)
                torch.save(model.state_dict(), filename)
                print("########## Saved model ##########")

            # End of pretrain stage
            if epoch == pretrain_time:
                print("########## END pretrain ##########")
                best_val_loss = 0
                epochs_since_best = 0
                lr = max(args.start_lr, args.min_lr)

                state_dict = torch.load("{}_best_pretrain.pth".format(init_pthname0))
                model.load_state_dict(state_dict)

            elif epoch == pretrain_time + pretrain2_time:
                print("########## END pretrain2 ##########")
                best_val_loss = 0
                epochs_since_best = 0
                lr = max(args.start_lr, args.min_lr)

                state_dict = torch.load("{}_best_pretrain2.pth".format(init_pthname))
                model.load_state_dict(state_dict)
                pretrain2_model = model
                pretrained2_list = list(state_dict.items())

                params["pretrain2"] = False
                model = load_model(args.model, params, parser)
                if args.cuda:
                    model.cuda()
                model_dict = model.state_dict()

                pretrained2_dict = {
                    k: v for k, v in pretrained2_list if k in model_dict
                }
                model_dict.update(pretrained_dict)
                model.load_state_dict(model_dict)
                print("pretrained2 model was loaded")

        print("Best Val Loss: {:.4f}".format(best_val_loss))

    # Load params
    params = pickle.load(open(init_filename0 + "/params.p", "rb"))

    # Load model
    state_dict = torch.load(
        "{}_best.pth".format(init_pthname, params["model"]),
        map_location=lambda storage, loc: storage,
    )
    model.load_state_dict(state_dict)
    # for tensorboardX of model graph
    # dataiter = iter(train_loader)
    # data_for_visual, labels = dataiter.next()
    # data_for_visual = data_for_visual.permute(2, 1, 0, 3)
    # dummy_input = torch.rand((4,1,100,10))
    # model_wrapper = ModelWrapper(model)
    # writer.add_graph(model_wrapper, data_for_visual)

    # Load ground-truth states from test set
    loader = test_loader
    n_sample = 10  # 10
    n_smp_b = 10 if args.dataset == "nba" or args.dataset == "bat" else 1  # 10
    if args.n_GorS >= 50 and args.dataset == "nba" and args.attention == 3:
        n_smp_b = 5
    rep_smp = int(n_sample / n_smp_b)
    i = 0
    if True:
        print("test sample")
        # Sample trajectory
        samples = [
            np.zeros((args.horizon + 1, args.n_agents, len_seqs_test, featurelen))
            for t in range(n_sample)
        ]
        samples_true = [
            np.zeros((args.horizon + 1, args.n_agents, len_seqs_test, featurelen))
            for t in range(n_sample)
        ]
        hard_att = np.zeros(
            (
                args.horizon,
                args.n_agents,
                len_seqs_test,
                args.n_all_agents + 1,
                n_sample,
            )
        )
        macros = np.zeros((args.horizon, args.n_agents, len_seqs_test, n_sample))
        loss_i = [{} for t in range(n_sample)]
        losses = {}
        losses2 = {}
        # prediction_list = []
        for r in range(rep_smp):
            start_time = time.time()
            if r > 0:
                state_dict = torch.load(
                    "{}_best.pth".format(init_pthname, params["model"]),
                    map_location=lambda storage, loc: storage,
                )
                model.load_state_dict(state_dict)

            for batch_idx, (data, macro_intents) in enumerate(loader):
                if args.cuda:
                    data = data.cuda()  # , data_y.cuda()
                    # (batch, agents, time, feat) => (time, agents, batch, feat)
                data = data.permute(2, 1, 0, 3)

                if "MACRO" in args.model:
                    macro_intents = macro_intents.transpose(0, 1)
                    sample, macro, att, output, output2, prediction = model.sample(
                        data,
                        macro_intents,
                        rollout=True,
                        burn_in=args.burn_in,
                        L_att=L_att,
                        CF_pred=False,
                        n_sample=n_smp_b,
                        TEST=True,
                    )
                    att = att.detach().cpu().numpy()
                    # macro = macro.detach().cpu().numpy()
                    x_pre = float(prediction[1][4].item())
                    y_pre = float(prediction[1][5].item())
                    # i += 1
                    # prediction_list.append([x_pre, y_pre])
                    # print(prediction_list)
                    # writer.add_scalar('test/prediction', x_pre, y_pre)
                    # writer.add_scalar('test/prediction_x', x_pre, i)
                    # writer.add_scalar('test/prediction_y', y_pre, i)
                else:
                    sample, _, _, output, output2, prediction = model.sample(
                        data,
                        rollout=True,
                        burn_in=args.burn_in,
                        L_att=L_att,
                        CF_pred=False,
                        n_sample=n_smp_b,
                        TEST=True,
                    )
                    # x_pre = float(prediction[1][4].item())
                    # y_pre = float(prediction[1][5].item())
                    # prediction_list.append([x_pre, ])
                    # print(prediction_list)
                    # i += 1
                    # writer.add_scalar('test/prediction', x_pre, y_pre)
                    # writer.add_scalar('test/prediction_x', x_pre, i)
                    # writer.add_scalar('test/prediction_y', y_pre, i)

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
                    ] = sample0[
                        :-3
                    ]  # なんで:-3？, 3フレームだけなくなる
                    samples_true[r * n_smp_b + i][
                        :,
                        :,
                        batch_idx * batchSize_test : (batch_idx + 1) * batchSize_test,
                    ] = data0[
                        :-3
                    ]  ###ここ実際の軌跡
                    # writer.add_scalar('test/prediction', samples_true[][0], samples_true[][1])
                    # import pdb; pdb.set_trace()
                    if "MACRO" in args.model:
                        hard_att[
                            :,
                            :,
                            batch_idx
                            * batchSize_test : (batch_idx + 1)
                            * batchSize_test,
                            :,
                            r * n_smp_b + i,
                        ] = att[:, :, :, :, i]
                        # macros[:,:,batch_idx*batchSize_test:(batch_idx+1)*batchSize_test,i] = macros[:,:,:,i] # (time,agents,batch,samples)
                del sample, sample0
                for key in output:  # lossesが出力、keyが
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
                    "Test sample "
                    + str(r * n_smp_b + i)
                    + ":\t"
                    + loss_str(loss_i[r * n_smp_b + i])
                )
                writer.add_scalars(
                    "test/loss", loss_i[r * n_smp_b + i], r * n_smp_b + i
                )

            epoch_time = time.time() - start_time
            print("Time:\t {:.3f}".format(epoch_time))  # Sample {} r*n_smp_b,

        # writer.close()
        if (
            True
        ):  # create Mean + SD Tex Table for positions------------------------------------------------
            avgL2_m = {}
            avgL2_sd = {}
            bestL2_m = {}
            bestL2_sd = {}
            for key in losses2:
                mean = np.mean(losses2[key], 0)
                avgL2_m[key] = np.mean(mean)
                avgL2_sd[key] = np.std(mean)
                best = np.min(losses2[key], 0)
                bestL2_m[key] = np.mean(best)
                bestL2_sd[key] = np.std(best)

            print(args.model + "att" + str(args.attention) + " body:" + str(args.body))
            print(
                "(mean):"
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
            )
            print(
                "(best):"
                + " $"
                + "{:.2f}".format(bestL2_m["e_pos"])
                + " \pm "
                + "{:.2f}".format(bestL2_sd["e_pos"])
                + "$ &"
                + " $"
                + "{:.2f}".format(bestL2_m["e_vel"])
                + " \pm "
                + "{:.2f}".format(bestL2_sd["e_vel"])
                + "$ &"
                + " $"
                + "{:.2f}".format(bestL2_m["e_acc"])
                + " \pm "
                + "{:.2f}".format(bestL2_sd["e_acc"])
                + "$ &"
            )
            print(
                "(max):"
                + " $"
                + "{:.2f}".format(avgL2_m["e_pmax"])
                + " \pm "
                + "{:.2f}".format(avgL2_sd["e_pmax"])
                + "$ &"
                + " $"
                + "{:.2f}".format(avgL2_m["e_vmax"])
                + " \pm "
                + "{:.2f}".format(avgL2_sd["e_vmax"])
                + "$ &"
                + " $"
                + "{:.2f}".format(avgL2_m["e_amax"])
                + " \pm "
                + "{:.2f}".format(avgL2_sd["e_amax"])
                + "$ &"
            )

        # Save samples
        experiment_path = "{}/experiments/sample".format(init_filename0)
        if not os.path.exists(experiment_path):
            os.makedirs(experiment_path)
        if False:  # 'MACRO' in args.model:
            pickle.dump(
                [samples, hard_att, losses2, macros],
                open(experiment_path + "/samples.p", "wb"),
                protocol=4,
            )
        else:
            pickle.dump(
                [samples, samples_true, hard_att, losses2],
                open(experiment_path + "/samples.p", "wb"),
                protocol=4,
            )

    if "MACRO" in args.model and not args.wo_macro:
        # Sample trajectory
        samples = [
            np.zeros((args.horizon + 1, args.n_agents, len_seqs_test, featurelen))
            for t in range(n_sample)
        ]
        hard_att = np.zeros(
            (
                args.horizon,
                args.n_agents,
                len_seqs_test,
                args.n_all_agents + 1,
                n_sample,
            )
        )

        for batch_idx, (data, macro_intents) in enumerate(loader):
            if args.cuda:
                data = data.cuda()  # , data_y.cuda()
                # (batch, agents, time, feat) => (time, agents, batch, feat)
            data = data.permute(2, 1, 0, 3)
            for i in range(n_sample):
                samples[i][
                    :, :, batch_idx * batchSize_test : (batch_idx + 1) * batchSize_test
                ] = (data[: args.horizon + 1, :, :, :].detach().cpu().numpy())

        # Save samples
        experiment_path = "{}/experiments/sample_1step".format(init_filename0)
        if not os.path.exists(experiment_path):
            os.makedirs(experiment_path)
        pickle.dump(samples, open(experiment_path + "/samples.p", "wb"), protocol=4)
        print("GT data is saved")
    i = 0
    # counter factural prediction
    CF_pred = True
    if CF_pred and args.attention == 3:
        print("counter factural prediction")
        samples = [
            np.zeros((args.horizon + 1, args.n_agents, len_seqs_test, featurelen))
            for t in range(n_sample)
        ]
        hard_att = np.zeros(
            (
                args.horizon,
                args.n_agents,
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
            if r > 0:
                state_dict = torch.load(
                    "{}_best.pth".format(init_pthname, params["model"]),
                    map_location=lambda storage, loc: storage,
                )
                model.load_state_dict(state_dict)
            for batch_idx, (data, macro_intents) in enumerate(loader):
                if args.cuda:
                    data = data.cuda()  # , data_y.cuda()
                    # (batch, agents, time, feat) => (time, agents, batch, feat)
                data = data.permute(2, 1, 0, 3)
                macro_intents = macro_intents.transpose(0, 1)
                sample, _, att, _, output2, prediction = model.sample(
                    data,
                    macro_intents,
                    rollout=True,
                    burn_in=args.burn_in,
                    L_att=L_att,
                    CF_pred=CF_pred,
                    n_sample=n_smp_b,
                    TEST=True,
                )
                att = att.detach().cpu().numpy()
                # x_pre = float(prediction[1][4].item())
                # y_pre = float(prediction[1][5].item())
                # prediction_list.append([x_pre, y_pre])
                # print(prediction_list)
                # i += 1
                # writer.add_scalar('test/prediction', x_pre, y_pre)
                # writer.add_scalar('test/prediction_x', x_pre, i)
                # writer.add_scalar('test/prediction_y', y_pre, i)
                for i in range(n_smp_b):
                    sample0 = (
                        sample.detach().cpu().numpy()
                        if n_smp_b == 1
                        else sample[i].detach().cpu().numpy()
                    )
                    samples[r * n_smp_b + i][
                        :,
                        :,
                        batch_idx * batchSize_test : (batch_idx + 1) * batchSize_test,
                    ] = sample0[:-3]
                    hard_att[
                        :,
                        :,
                        batch_idx * batchSize_test : (batch_idx + 1) * batchSize_test,
                        :,
                        r * n_smp_b + i,
                    ] = att[:, :, :, :, i]
                del sample, sample0, att
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
                    "CF sample "
                    + str(r * n_smp_b + i)
                    + ":\t"
                    + loss_str(loss_i[r * n_smp_b + i])
                )
            epoch_time = time.time() - start_time
            print("Time:\t {:.3f}".format(epoch_time))  # Sample {} r*n_smp_b,

        experiment_path = "{}/experiments/sample_CF".format(init_filename0)
        if not os.path.exists(experiment_path):
            os.makedirs(experiment_path)
        pickle.dump(
            [samples, hard_att, losses2],
            open(experiment_path + "/samples.p", "wb"),
            protocol=4,
        )

    # for i in range(170):
    #     writer.add_scalar('test/prediction', prediction_list[i][0], prediction_list[i][1])
    # writer.add_scalars('test/prediction', prediction[0][0], prediction[0][1])
    # print(prediction_list)
    writer.close()
