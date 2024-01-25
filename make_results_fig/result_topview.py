import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os
import glob
import pandas as pd
import statistics
import json
from natsort import natsorted
from matplotlib.backends.backend_pdf import PdfPages

sns.set(
    "paper",
    "whitegrid",
    "bright",
    font_scale=2.5,
    rc={"lines.linewidth": 3, "grid.linestyle": "--"},
)


with open(
    "./weights/for_paper/kiku/params.p",
    "rb",
) as f:
    param = np.load(f, allow_pickle=True)
    # print(param)
    predict_time = param["burn_in"]
with open(
    "./weights/for_paper/kiku/samples_val.p",
    "rb",
) as f:
    data = np.load(f, allow_pickle=True)

# import pdb; pdb.set_trace()
os.makedirs("./weights/for_paper/kiku/results", exist_ok=True)
pp = PdfPages("./weights/for_paper/kiku/results/topview_vrnn_kiku_val.pdf")

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


train_vel = []
test_vel = []
loss_position = []
loss_velocity = []

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
        train_pos_x = data[1][0][step][0][episode][0]
        train_pos_y = data[1][0][step][0][episode][1]
        test_pos_x = data[0][0][step][0][episode][0]
        test_pos_y = data[0][0][step][0][episode][1]
        train_x_pre.append(train_pos_x)
        train_y_pre.append(train_pos_y)
        test_x.append(test_pos_x)
        test_y.append(test_pos_y)
        train_pulse.append(data[1][0][step][0][episode][5])
        test_pulse.append(data[0][0][step][0][episode][5])
        all_pulse.append(data[0][0][step][0][episode][5])
        loss_position.append(
            np.sqrt(
                (test_pos_y - train_pos_y) ** 2
                + (test_pos_x - train_pos_x) ** 2
            )
        )
        train_vel_x = data[1][0][step][0][episode][2]
        train_vel_y = data[1][0][step][0][episode][3]
        test_vel_x = data[0][0][step][0][episode][2]
        test_vel_y = data[0][0][step][0][episode][3]
        train_vel.append(np.sqrt(train_vel_x**2 + train_vel_y**2))
        test_vel.append(np.sqrt(test_vel_x**2 + test_vel_y**2))
        loss_velocity.append(
            np.sqrt(
                (test_vel_y - train_vel_y) ** 2
                + (test_vel_x - train_vel_x) ** 2
            )
        )
    # print(test_pulse)
    # print(train_pulse.count(0))

    fig = plt.figure(figsize=(6, 6))
    ax = fig.add_subplot(111)
    Env_name = get_env_name(data[0][0][step][0][episode][6])

    #############8s###############
    if data[0][0][step][0][episode][7] >= 200:
        obs_point_dict = json.load(
            open(
                "./calc_states/preprocess_bats/obstacle_information/2023/Envs_kiku.json",
                "r",
            )
        )
    elif (
        data[0][0][step][0][episode][7] >= 100
        and data[0][0][step][0][episode][7] < 200
    ):
        obs_point_dict = json.load(
            open(
                "./calc_states/preprocess_bats/obstacle_information/2023/Envs_yubi.json",
                "r",
            )
        )

    obs_x = obs_point_dict[Env_name]["x"]
    obs_y = obs_point_dict[Env_name]["y"]

    sns.scatterplot(
        x=obs_x,
        y=obs_y,
        marker="s",
        # label="chain",
        color="green",
        s=8,
    )

    sns.lineplot(
        x=train_x_pre,
        y=train_y_pre,
        # label="measured",
        sort=False,
        linestyle="--",
    )
    sns.lineplot(
        x=test_x,
        y=test_y,
        # label="predicted",
        sort=False,
    )

    # for i in range(len(train_pulse)):
    #     if train_pulse[i] >= 0.5:
    #         ax.scatter(
    #             train_x_pre[i],
    #             train_y_pre[i],
    #             label="measured pulse timing",
    #             color="w",
    #             edgecolors="#1f77b4",
    #             s=20,
    #             zorder=3,
    #         )
    #     if test_pulse[i] >= 0.5:
    #         ax.scatter(
    #             test_x[i],
    #             test_y[i],
    #             label="predicted pulse timing",
    #             color="w",
    #             edgecolors="#d62728",
    #             s=20,
    #             zorder=3,
    #         )

    plt.xlabel("X [m]")
    plt.ylabel("Y [m]")

    plt.xticks([0, 1.5, 3.0, 4.5])
    plt.yticks([0, 1.5, 3.0, 4.5])
    plt.tight_layout()
    ax.set_aspect("equal")
    # plt.legend(bbox_to_anchor=(1, 1), ncol=1, fontsize=10)

    pp.savefig(fig)

pp.close()
print(
    f"loss position: mean:{statistics.mean(loss_position)} std:{statistics.stdev(loss_position)}"
)
print(
    f"loss velocity: mean:{statistics.mean(loss_velocity)} std:{statistics.stdev(loss_velocity)}"
)
print(
    f"mesured velocity: mean:{statistics.mean(train_vel)} std:{statistics.stdev(train_vel)}"
)
print(
    f"predicted velocity: mean:{statistics.mean(test_vel)} std:{statistics.stdev(test_vel)}"
)
