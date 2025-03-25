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
from matplotlib.colors import LinearSegmentedColormap
import argparse

sns.set(
    "paper",
    "whitegrid",
    "bright",
    font_scale=2.5,
    rc={"lines.linewidth": 3, "grid.linestyle": "--"},
)

# cmap = LinearSegmentedColormap.from_list("red_blue", ["blue", "red"])
parser = argparse.ArgumentParser()
parser.add_argument("--bat_type", type=str, required=True)
parser.add_argument("--model_type", type=str, default="VRNN")
parser.add_argument("--pulse_pred", type=bool, default=False)
parser.add_argument("--flag_valdata", type=bool, default=False)
args, _ = parser.parse_known_args()
bat_type = args.bat_type
model_type = args.model_type
folder_path = "yubi/train_each/seed_40/with_std"
pulse_flag = args.pulse_pred
flag_val = args.flag_valdata
# folder_path = f"{bat_type}/{model_type}"

with open(
    f"./weights/for_paper/{folder_path}/params.p",
    "rb",
) as f:
    param = np.load(f, allow_pickle=True)
    predict_time = param["burn_in"]
    dt = param["fs"]


with open(
    f"./weights/for_paper/{folder_path}/samples.p",
    "rb",
) as f:
    data_test = np.load(f, allow_pickle=True)

if flag_val:
    with open(
        f"./weights/for_paper/{folder_path}/samples_val.p",
        "rb",
    ) as f:
        data_val = np.load(f, allow_pickle=True)


# with open(
#     f"./weights/for_paper/{bat_type}/{model_type}/params.p",
#     "rb",
# ) as f:
#     param = np.load(f, allow_pickle=True)
#     # print(param)
#     predict_time = param["burn_in"]
# with open(
#     f"./weights/for_paper/{bat_type}/{model_type}/samples.p",
#     "rb",
# ) as f:
#     data_test = np.load(f, allow_pickle=True)

# with open(
#     f"./weights/for_paper/{bat_type}/{model_type}/samples_val.p",
#     "rb",
# ) as f:
#     data_val = np.load(f, allow_pickle=True)

# import pdb; pdb.set_trace()


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


def calc_data(
    data,
    episode,
    predict_time,
):
    pos_x_burnin_list = []
    pos_y_burnin_list = []
    pos_x_measured_list = []
    pos_y_measured_list = []
    pos_x_predicted_list = []
    pos_y_predicted_list = []
    pulse_measured_list = []
    pulse_predicted_list = []
    vel_measured_list = []
    vel_predicted_list = []
    vel_x_std_predicted_list = []
    vel_y_std_predicted_list = []
    loss_position = []
    loss_velocity = []
    env_num = data[0][0][0][0][episode][8]
    bat_name = data[0][0][0][0][episode][9]

    for step in range(len(data[0][0])):
        if step <= predict_time:
            pos_x_burnin_list.append(data[1][0][step][0][episode][0])
            pos_y_burnin_list.append(data[1][0][step][0][episode][1])
        elif step > predict_time:
            pos_x_measured = data[1][0][step][0][episode][0]
            pos_y_measured = data[1][0][step][0][episode][1]
            pos_x_predicted = data[0][0][step][0][episode][0]
            pos_y_predicted = data[0][0][step][0][episode][1]
            pos_x_measured_list.append(pos_x_measured)
            pos_y_measured_list.append(pos_y_measured)
            pos_x_predicted_list.append(pos_x_predicted)
            pos_y_predicted_list.append(pos_y_predicted)
            pulse_measured_list.append(data[1][0][step][0][episode][5])
            pulse_predicted_list.append(data[0][0][step][0][episode][7])
            loss_position.append(
                np.sqrt(
                    (pos_y_predicted - pos_y_measured) ** 2
                    + (pos_x_predicted - pos_x_measured) ** 2
                )
            )
            vel_x_measured = data[1][0][step][0][episode][2]
            vel_y_measured = data[1][0][step][0][episode][3]
            vel_x_predicted = data[0][0][step][0][episode][2]
            vel_y_predicted = data[0][0][step][0][episode][3]
            vel_x_std_predicted = data[0][0][step][0][episode][4]
            vel_y_std_predicted = data[0][0][step][0][episode][5]
            vel_measured_list.append(
                np.sqrt(vel_x_measured**2 + vel_y_measured**2)
            )
            vel_predicted_list.append(
                np.sqrt(vel_x_predicted**2 + vel_y_predicted**2)
            )
            vel_x_std_predicted_list.append(vel_x_std_predicted)
            vel_y_std_predicted_list.append(vel_y_std_predicted)
            loss_velocity.append(
                np.sqrt(
                    (vel_y_predicted - vel_y_measured) ** 2
                    + (vel_x_predicted - vel_x_measured) ** 2
                )
            )

    return (
        env_num,
        bat_name,
        pos_x_measured_list,
        pos_y_measured_list,
        pos_x_predicted_list,
        pos_y_predicted_list,
        pulse_measured_list,
        pulse_predicted_list,
        vel_measured_list,
        vel_predicted_list,
        vel_x_std_predicted_list,
        vel_y_std_predicted_list,
        loss_position,
        loss_velocity,
    )


def calc_obs(data, episode):
    Env_name = get_env_name(data[0][0][0][0][episode][8])

    obs_point_dict = json.load(
        open(
            "./calc_states/preprocess_bats/obstacle_information/2023/Envs_yubi.json",
            "r",
        )
    )

    # if data[0][0][0][0][episode][7] >= 200:
    #     obs_point_dict = json.load(
    #         open(
    #             "./calc_states/preprocess_bats/obstacle_information/2023/Envs_kiku.json",
    #             "r",
    #         )
    #     )
    # elif (
    #     data[0][0][0][0][episode][7] >= 100
    #     and data[0][0][0][0][episode][7] < 200
    # ):
    #     obs_point_dict = json.load(
    #         open(
    #             "./calc_states/preprocess_bats/obstacle_information/2023/Envs_yubi.json",
    #             "r",
    #         )
    #     )

    obs_x = obs_point_dict[Env_name]["x"]
    obs_y = obs_point_dict[Env_name]["y"]

    return obs_x, obs_y


def write_csv(
    env_num,
    bat_name,
    loss_position,
    loss_velocity,
    vel_measured_list,
    vel_predicted_list,
    sample_type,
):
    env_name = get_env_name(env_num)
    env_name_for_loss = [env_name] * len(loss_position)
    bat_name_for_loss = [bat_name] * len(loss_position)
    loss_dataset_for_pd = {
        "env": env_name_for_loss,
        "bat": bat_name_for_loss,
        "loss_position": loss_position,
        "loss_velocity": loss_velocity,
    }
    df_loss = pd.DataFrame(loss_dataset_for_pd)
    env_name_for_vel = [env_name] * len(vel_measured_list)
    bat_name_for_vel = [bat_name] * len(vel_measured_list)
    vel_dataset_for_pd = {
        "env": env_name_for_vel,
        "bat": bat_name_for_vel,
        "vel_measured": vel_measured_list,
        "vel_predicted": vel_predicted_list,
    }
    df_vel = pd.DataFrame(vel_dataset_for_pd)

    if sample_type == "TEST":
        df_loss.to_csv(
            f"./weights/for_paper/{folder_path}/results/loss.csv",
            mode="a",
            index=False,
            header=False,
        )
        df_vel.to_csv(
            f"./weights/for_paper/{folder_path}/results/velocity.csv",
            mode="a",
            index=False,
            header=False,
        )
    elif sample_type == "VAL":
        df_loss.to_csv(
            f"./weights/for_paper/{folder_path}/results/loss_val.csv",
            mode="a",
            index=False,
            header=False,
        )
        df_vel.to_csv(
            f"./weights/for_paper/{folder_path}/results/velocity_val.csv",
            mode="a",
            index=False,
            header=False,
        )


def plot_gaussian(x0, y0, std_x, std_y, ax, levels=10):

    x = np.linspace(x0 - 3 * std_x, x0 + 3 * std_x, 100)
    y = np.linspace(y0 - 3 * std_y, y0 + 3 * std_y, 100)
    X_grid, Y_grid = np.meshgrid(x, y)

    Z = (1 / (2 * np.pi * std_x * std_y)) * np.exp(
        -(
            ((X_grid - x0) ** 2) / (2 * std_x**2)
            + ((Y_grid - y0) ** 2) / (2 * std_y**2)
        )
    )

    levels_array = np.linspace(Z.min(), Z.max(), levels)

    ax.contour(X_grid, Y_grid, Z, levels=levels, colors="blue", alpha=0.7)


def calc(target_data, pp, sample_type):
    loss_postion_list = []
    loss_velocity_list = []
    vel_m_list = []
    vel_p_list = []
    vel_x_std_list = []
    vel_y_std_list = []
    for episode in range(len(target_data[0][0][0][0])):
        (
            env_num,
            bat_name,
            pos_x_measured_list,
            pos_y_measured_list,
            pos_x_predicted_list,
            pos_y_predicted_list,
            pulse_measured_list,
            pulse_predicted_list,
            vel_measured_list,
            vel_predicted_list,
            vel_x_std_predicted_list,
            vel_y_std_predicted_list,
            loss_position,
            loss_velocity,
        ) = calc_data(
            target_data,
            episode,
            predict_time,
        )
        loss_postion_list.append(loss_position)
        loss_velocity_list.append(loss_velocity)
        vel_m_list.append(vel_measured_list)
        vel_p_list.append(vel_predicted_list)
        vel_x_std_list.append(vel_x_std_predicted_list)
        vel_y_std_list.append(vel_y_std_predicted_list)
        write_csv(
            env_num,
            bat_name,
            loss_position,
            loss_velocity,
            vel_measured_list,
            vel_predicted_list,
            sample_type,
        )

        obs_x, obs_y = calc_obs(target_data, episode)
        # make fig
        fig = plt.figure(figsize=(6, 6))
        ax = fig.add_subplot(111)
        sns.scatterplot(
            x=obs_x,
            y=obs_y,
            marker="o",
            label="chain",
            color="#ff7f0e",
            s=20,
        )

        sns.lineplot(
            x=pos_x_measured_list,
            y=pos_y_measured_list,
            label="measured",
            sort=False,
            linestyle="--",
            color="gray",
        )
        for i in range(0, len(vel_x_std_predicted_list)):
            pos_std_x = vel_x_std_predicted_list[i] * dt
            pos_std_y = vel_y_std_predicted_list[i] * dt
            plot_gaussian(
                pos_x_predicted_list[i],
                pos_y_predicted_list[i],
                pos_std_x,
                pos_std_y,
                ax,
            )
        sns.lineplot(
            x=pos_x_predicted_list,
            y=pos_y_predicted_list,
            label="predicted",
            sort=False,
            color="red",
            lw=2,
        )
        if pulse_flag:
            for i in range(len(pulse_measured_list)):
                if pulse_measured_list[i] >= 0.5:
                    sns.scatterplot(
                        x=pos_x_measured_list[i],
                        y=pos_y_measured_list[i],
                        label="measured pulse timing",
                        color="w",
                        edgecolors="#1f77b4",
                        s=20,
                        zorder=3,
                    )
                if pulse_predicted_list[i] >= 0.5:
                    sns.scatterplot(
                        x=pos_x_predicted_list[i],
                        y=pos_y_predicted_list[i],
                        label="predicted pulse timing",
                        color="w",
                        edgecolors="#d62728",
                        s=20,
                        zorder=3,
                    )

        plt.xlabel("X [m]")
        plt.ylabel("Y [m]")

        plt.xticks([0, 1.5, 3.0, 4.5])
        plt.yticks([0, 1.5, 3.0, 4.5])
        plt.xlim(0, 4.5)
        plt.ylim(0, 4.5)
        plt.tight_layout()
        ax.set_aspect("equal")
        plt.legend(loc="upper right", borderaxespad=0, ncol=1, fontsize=10)

        pp.savefig(fig)

    print(
        f"loss position: mean:{statistics.mean(np.concatenate(loss_postion_list))} std:{statistics.stdev(np.concatenate(loss_postion_list))}"
    )
    print(
        f"loss velocity: mean:{statistics.mean(np.concatenate(loss_velocity_list))} std:{statistics.stdev(np.concatenate(loss_velocity_list))}"
    )
    print(
        f"mesured velocity: mean:{statistics.mean(np.concatenate(vel_m_list))} std:{statistics.stdev(np.concatenate(vel_m_list))}"
    )
    print(
        f"predicted velocity: mean:{statistics.mean(np.concatenate(vel_p_list))} std:{statistics.stdev(np.concatenate(vel_p_list))}"
    )
    pp.close()


def main():
    # val data resutls
    if flag_val:
        print("calc validation data")
        os.makedirs(f"./weights/for_paper/{folder_path}/results", exist_ok=True)
        pp = PdfPages(
            f"./weights/for_paper/{folder_path}/results/topview_{model_type}_{bat_type}_val_with_legend.pdf"
        )
        calc(data_val, pp, "VAL")

    # test data resutls
    print("calc test data")
    os.makedirs(f"./weights/for_paper/{folder_path}/results", exist_ok=True)
    pp = PdfPages(
        f"./weights/for_paper/{folder_path}/results/topview_{model_type}_{bat_type}_with_legend.pdf"
    )
    calc(data_test, pp, "TEST")


if __name__ == "__main__":
    main()
