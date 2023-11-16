import matplotlib.pyplot as plt
import numpy as np
import os
import glob
import pandas as pd
import statistics
import json
from natsort import natsorted
from matplotlib.backends.backend_pdf import PdfPages


with open('./weights/sub100_filt_vel_meanHMM_inSimple_acc_0_unnorm/MACRO_VRNN_bat/att_-1_100_796_wo_macro/params.p', 'rb') as f:
    param = np.load(f, allow_pickle=True)
    # print(param)
    predict_time = param['burn_in']

with open('./weights/sub100_filt_vel_meanHMM_inSimple_acc_0_unnorm/MACRO_VRNN_bat/att_-1_100_796_wo_macro/experiments/sample/samples.p', 'rb') as f:
    data = np.load(f, allow_pickle=True)

# import pdb; pdb.set_trace()

pp = PdfPages('./teshima/result/pulse/topview_vrnn.pdf')

count = 0

all_pulse = []
pos_me = []
pos_pre = []
vel_me = []
vel_pre = []
loss_pos = []
loss_vel = []

for episode in range(len(data[0][0][0][0])):
    train_x = []
    train_y = []
    train_x_pre = []
    train_y_pre = []
    test_x = []
    test_y = []
    train_pulse = []
    test_pulse = []

    count += 1

    for step in range(predict_time + 1):
        train_x.append(data[1][0][step][0][episode][0])
        train_y.append(data[1][0][step][0][episode][1])
    for step in range(predict_time, len(data[0][0])):
        train_x_pre.append(data[1][0][step][0][episode][0])
        train_y_pre.append(data[1][0][step][0][episode][1])
        test_x.append(data[0][0][step][0][episode][0])
        test_y.append(data[0][0][step][0][episode][1])
        train_pulse.append(data[1][0][step][0][episode][5])
        test_pulse.append(data[0][0][step][0][episode][5])
        all_pulse.append(data[0][0][step][0][episode][5])

    # print(test_pulse)
    # print(train_pulse.count(0))

    # import pdb; pdb.set_trace()

    fig = plt.figure(figsize=(6, 6))
    ax = fig.add_subplot(111)

####### yubi #######
    obs_point_dict = json.load(open("./make_results_fig/Envs.json", "r"))
    if count <= 15:
        obs_x = obs_point_dict["Env3"]["x"]
        obs_y = obs_point_dict["Env3"]["y"]

        ax.scatter(
            obs_x, obs_y, marker="o", label="chain", color="#ff7f0e", s=8, zorder=1
        )
    elif count >= 16 and count <= 30:
        obs_x = obs_point_dict["Env4"]["x"]
        obs_y = obs_point_dict["Env4"]["y"]

        ax.scatter(
            obs_x, obs_y, marker="o", label="chain", color="#ff7f0e", s=8, zorder=1
        )
    elif count >= 31 and count <= 43:
        obs_x = obs_point_dict["Env5"]["x"]
        obs_y = obs_point_dict["Env5"]["y"]

        ax.scatter(
            obs_x, obs_y, marker="o", label="chain", color="#ff7f0e", s=8, zorder=1
        )
    elif count >= 44 and count <= 53:
        obs_x = obs_point_dict["Env6"]["x"]
        obs_y = obs_point_dict["Env6"]["y"]

        ax.scatter(
            obs_x, obs_y, marker="o", label="chain", color="#ff7f0e", s=8, zorder=1
        )

####### kiku #######
    # if count <= 13:
    #     from result.kiku.obstacle_information.Env4 import obs_x, obs_y
    #     ax.scatter(obs_x, obs_y, marker='o', label='chain', color='#ff7f0e', s=8, zorder=1)
    # elif count >= 14 and count <= 24:
    #     from result.kiku.obstacle_information.Env6 import obs_x, obs_y
    #     ax.scatter(obs_x, obs_y, marker='o', label='chain', color='#ff7f0e', s=8, zorder=1)
    # elif count >= 25:
    #     from result.kiku.obstacle_information.Env7 import obs_x, obs_y
    #     ax.scatter(obs_x, obs_y, marker='o', label='chain', color='#ff7f0e', s=8, zorder=1)

    ax.plot(train_x, train_y, label='input flight path',
            color='#2ca02c', linestyle='--', zorder=2, linewidth=3)
    ax.plot(train_x_pre, train_y_pre, label='measured flight path',
            color='#1f77b4', zorder=2, linewidth=3)
    ax.plot(test_x, test_y, label='predicted flight path',
            color='#d62728', zorder=2, linewidth=3)

    for i in range(len(train_pulse)):
        if train_pulse[i] != 0:
            ax.scatter(train_x_pre[i], train_y_pre[i], label='measured pulse timing', color='w', edgecolors="#1f77b4", s=10, zorder=3)
        if test_pulse[i] >= 0.5:
            ax.scatter(test_x[i], test_y[i], label='predicted pulse timing', color='w', edgecolors="#d62728", s=10, zorder=3)

    plt.xlabel('X [m]')
    plt.ylabel('Y [m]')

    plt.xticks([0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5])
    plt.yticks([0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5])
    plt.tight_layout()
    ax.set_aspect('equal')
    # plt.legend(bbox_to_anchor=(1,1), loc='upper left')

    pp.savefig(fig)

pp.close()

# print("mesured position mean : {}".format(statistics.mean(pos_me)))
# print("mesured position stdev : {}".format(statistics.stdev(pos_me)))
# print("predicted position mean : {}".format(statistics.mean(pos_pre)))
# print("predicted position stdev : {}".format(statistics.stdev(pos_pre)))

# print("mesured velocity mean : {}".format(statistics.mean(vel_me)))
# print("mesured velocity stdev : {}".format(statistics.stdev(vel_me)))
# print("predicted velocity mean : {}".format(statistics.mean(vel_pre)))
# print("predicted velocity stdev : {}".format(statistics.stdev(vel_pre)))

# print("loss position mean : {}".format(statistics.mean(loss_pos)))
# print("loss position stdev : {}".format(statistics.stdev(loss_pos)))
# print("loss velocity mean : {}".format(statistics.mean(loss_vel)))
# print("loss velocity stdev : {}".format(statistics.stdev(loss_vel)))