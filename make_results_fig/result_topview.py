import matplotlib.pyplot as plt
import numpy as np
import os
import glob
import pandas as pd
import statistics
import json
from natsort import natsorted
from matplotlib.backends.backend_pdf import PdfPages


with open(
    "./home/user/Desktop/dev_python/imitation_bat/PO-MC-DHVRNN/teshima/result/without_pulse/yubi/epoch200/att_-1_100_796_wo_macro/params.p",
    "rb",
) as f:
    param = np.load(f, allow_pickle=True)
    # print(param)
    predict_time = param["burn_in"]

with open(
    "./home/user/Desktop/dev_python/imitation_bat/PO-MC-DHVRNN/teshima/result/without_pulse/yubi/epoch200/att_-1_100_796_wo_macro/experiments/sample/samples.p",
    "rb",
) as f:
    data = np.load(f, allow_pickle=True)

# import pdb; pdb.set_trace()
os.makedirs("./teshima/result/pulse", exist_ok=True)
pp = PdfPages("./teshima/result/pulse/topview_vrnn_yubi.pdf")

count = 0

all_pulse = []


def get_env_name(env_num):
    if env_num == 1:
        env_name = "Env1"
    elif env_num == 2:
        env_name = "Env2"
    elif env_num == 3:
        env_name = "Env3"
    elif env_num == 4:
        env_name = "Env4"
    elif env_num == 5:
        env_name = "Env5"
    elif env_num == 6:
        env_name = "Env6"
    elif env_num == 7:
        env_name = "Env7"
    elif env_num == 8:
        env_name = "Test"

    return env_name


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
    Env_name = get_env_name(data[0][0][step][0][episode][6])

    #############8s###############
    obs_point_dict = json.load(open("./make_results_fig/Envs.json", "r"))

    obs_x = obs_point_dict[Env_name]["x"]
    obs_y = obs_point_dict[Env_name]["y"]

    ax.scatter(
        obs_x,
        obs_y,
        marker="o",
        label="chain",
        color="#ff7f0e",
        s=8,
        zorder=1,
    )

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

    # ax.plot(train_x, train_y, label='mesured flight path', color='#1f77b4')
    ax.plot(
        train_x_pre,
        train_y_pre,
        label="measured flight path",
        color="#1f77b4",
        linestyle="--",
        zorder=2,
    )
    ax.plot(
        test_x, test_y, label="predicted flight path", color="#d62728", zorder=2
    )

    # for i in range(len(train_pulse)):
    #     if train_pulse[i] >= 0.5:
    #         ax.scatter(
    #             train_x_pre[i],
    #             train_y_pre[i],
    #             label="measured pulse timing",
    #             color="#1f77b4",
    #             s=10,
    #             zorder=3,
    #         )
    #     if test_pulse[i] >= 0.5:
    #         ax.scatter(
    #             test_x[i],
    #             test_y[i],
    #             label="predicted pulse timing",
    #             color="#d62728",
    #             s=10,
    #             zorder=3,
    #         )

    plt.xlabel("X [m]")
    plt.ylabel("Y [m]")

    plt.xticks([0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5])
    plt.yticks([0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5])
    plt.tight_layout()
    ax.set_aspect("equal")
    # plt.legend(bbox_to_anchor=(1,1), loc='upper left')

    pp.savefig(fig)

pp.close()

print(statistics.mean(all_pulse))
print(statistics.stdev(all_pulse))
