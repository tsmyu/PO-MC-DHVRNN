import matplotlib.pyplot as plt
import numpy as np
import os
import glob
import pandas as pd
import statistics
from natsort import natsorted
from matplotlib.backends.backend_pdf import PdfPages


with open('./result/yubi->kiku/onlypath/params.p', 'rb') as f:
    param = np.load(f, allow_pickle=True)
    # print(param)
    predict_time = param['burn_in']

with open('./result/yubi->kiku/onlypath/samples.p', 'rb') as f:
    data = np.load(f, allow_pickle=True)

# import pdb; pdb.set_trace()

pp = PdfPages('./result/yubi->kiku/onlypath/topview.pdf')

count = 0

all_pulse = []

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

    for step in range(predict_time+1):
        train_x.append(data[1][0][step][0][episode][0])
        train_y.append(data[1][0][step][0][episode][1])
    for step in range(predict_time, len(data[0][0])):
        train_x_pre.append(data[1][0][step][0][episode][0])
        train_y_pre.append(data[1][0][step][0][episode][1])
        test_x.append(data[0][0][step][0][episode][0])
        test_y.append(data[0][0][step][0][episode][1])
        # train_pulse.append(data[1][0][step][0][episode][5])
        # test_pulse.append(data[0][0][step][0][episode][5])
        # all_pulse.append(data[0][0][step][0][episode][5])
    
    # print(test_pulse)
    # print(train_pulse.count(0))

    # import pdb; pdb.set_trace()
    
    fig = plt.figure(figsize=(6,6))
    ax = fig.add_subplot(111)

#############yubi###############
#############8s###############
    # if count <= 15:
    #     from obstacle_information.Env3 import obs_x, obs_y
    #     ax.scatter(obs_x, obs_y, marker='o', label='chain', color='#ff7f0e', s=8, zorder=1)
    # elif count >= 16 and count <= 30:
    #     from obstacle_information.Env4 import obs_x, obs_y
    #     ax.scatter(obs_x, obs_y, marker='o', label='chain', color='#ff7f0e', s=8, zorder=1)
    # elif count >= 31 and count <= 43:
    #     from obstacle_information.Env5 import obs_x, obs_y
    #     ax.scatter(obs_x, obs_y, marker='o', label='chain', color='#ff7f0e', s=8, zorder=1)
    # elif count >= 44 and count <= 53:
    #     from obstacle_information.Env6 import obs_x, obs_y
    #     ax.scatter(obs_x, obs_y, marker='o', label='chain', color='#ff7f0e', s=8, zorder=1)
    # elif count >= 43 and count <= 57:
    #     from obstacle_information.Env import obs_x, obs_y
    #     ax.scatter(obs_x, obs_y, marker='o', label='chain', color='#ff7f0e', s=8)
    # elif count >= 38 and count <= 47:
    #     from obstacle_information.EnvRegular_20230421 import obs_x, obs_y
    #     ax.scatter(obs_x, obs_y, marker='o', label='chain', color='#ff7f0e', s=8)

    # for i in range(len(train_pulse_pre)):
    #     if train_pulse_pre[i] == 1:
    #         ax.scatter(train_x_pre[i], train_y_pre[i], color='#1f77b4', s=10)

    # for i in range(len(test_pulse)):
    #     if test_pulse[i] != 0:
    #         ax.scatter(test_x[i], test_y[i], color='#d62728', s=10)

#############12s###############
    # if count <= 23:
    #     from obstacle_information.EnvRegular_20230421 import obs_x, obs_y
    #     ax.scatter(obs_x, obs_y, marker='o', label='chain', color='#ff7f0e', s=8)
    # elif count >= 24 and count <= 37:
    #     from obstacle_information.EnvRegular_20221018 import obs_x, obs_y
    #     ax.scatter(obs_x, obs_y, marker='o', label='chain', color='#ff7f0e', s=8)
    # elif count >= 38 and count <= 51:
    #     from obstacle_information.EnvRegular_20230429 import obs_x, obs_y
    #     ax.scatter(obs_x, obs_y, marker='o', label='chain', color='#ff7f0e', s=8)
    # elif count >= 52 and count <= 74:
    #     from obstacle_information.EnvRegular_20230428 import obs_x, obs_y
    #     ax.scatter(obs_x, obs_y, marker='o', label='chain', color='#ff7f0e', s=8)
    # elif count >= 75 and count <= 80:
    #     from obstacle_information.EnvRegular_20221011 import obs_x, obs_y
    #     ax.scatter(obs_x, obs_y, marker='o', label='chain', color='#ff7f0e', s=8)
    # elif count >= 81 and count <= 105:
    #     from obstacle_information.EnvRegular_20221003 import obs_x, obs_y
    #     ax.scatter(obs_x, obs_y, marker='o', label='chain', color='#ff7f0e', s=8)

    
#############kiku###############
#############8s###############
    if count <= 35:
        from obstacle_information.kiku.Env3 import obs_x, obs_y
        ax.scatter(obs_x, obs_y, marker='o', label='chain', color='#ff7f0e', s=8, zorder=1)
    elif count >= 26 and count <= 100:
        from obstacle_information.kiku.Env4 import obs_x, obs_y
        ax.scatter(obs_x, obs_y, marker='o', label='chain', color='#ff7f0e', s=8, zorder=1)
    elif count >= 101 and count <= 174:
        from obstacle_information.kiku.Env5 import obs_x, obs_y
        ax.scatter(obs_x, obs_y, marker='o', label='chain', color='#ff7f0e', s=8, zorder=1)
    elif count >= 175 and count <= 205:
        from obstacle_information.kiku.Env6 import obs_x, obs_y
        ax.scatter(obs_x, obs_y, marker='o', label='chain', color='#ff7f0e', s=8, zorder=1)


    # ax.plot(train_x, train_y, label='mesured flight path', color='#1f77b4')
    ax.plot(train_x_pre, train_y_pre, label='measured flight path', color='#1f77b4', linestyle='--', zorder=2)
    ax.plot(test_x, test_y, label='predicted flight path', color='#d62728', zorder=2)

    for i in range(len(train_pulse)):
        if train_pulse[i] != 0:
            ax.scatter(train_x_pre[i], train_y_pre[i], label='measured pulse timing', color='#1f77b4', s=10, zorder=3)
        if test_pulse[i] != 0:
            ax.scatter(test_x[i], test_y[i], label='predicted pulse timing', color='#d62728', s=10, zorder=3)

    plt.xlabel('X [m]')
    plt.ylabel('Y [m]')

    plt.xticks([0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5])
    plt.yticks([0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5])
    plt.tight_layout()
    ax.set_aspect('equal')
    # plt.legend(bbox_to_anchor=(1,1), loc='upper left')

    pp.savefig(fig)

pp.close()

# print(statistics.mean(all_pulse))
# print(statistics.stdev(all_pulse))