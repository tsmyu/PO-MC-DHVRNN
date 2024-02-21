import glob
import os
import numpy as np
import pandas as pd
import argparse

# from natsort import natsorted

from calculation import (
    vel,
    horizon_angle,
    vertical_angle,
    rotation,
    cross_point
)

from cut_down import cut_for_episode, dwnsmp


def calc_cross_points(target_data, pos_x, pos_z, env_name):
    X = target_data["X"][:-1]
    Y = target_data["Y"][:-1]
    Z = target_data["Z"][:-1]

    cross_list = cross_point(X, Z, pos_x, pos_z, env_name)

    return cross_list


def calc_rotation_point(target_data):
    X = target_data["X"]
    Y = target_data["Y"]
    Z = target_data["Z"]

    rot_all_x, rot_all_z = rotation(X, Z)

    return rot_all_x, rot_all_z


def calc_angles(target_data):
    X = target_data["X"]
    Y = target_data["Y"]
    Z = target_data["Z"]

    theta = horizon_angle(X, Z)
    alpha = vertical_angle(X, Y, Z)

    return theta, alpha


def calc_velocities(target_data):
    Vx = vel(target_data["X"], target_data["dt"])
    Vy = vel(target_data["Y"], target_data["dt"])
    Vz = vel(target_data["Z"], target_data["dt"])

    return Vx, Vy, Vz


def read_data(target_data):
    Time = target_data["Time (Seconds)"]
    dt = Time[1] - Time[0]
    target_data = {
        "Time": target_data["Time (Seconds)"],
        "dt": dt,
        "X": target_data["X"],
        "Y": target_data["Y"],
        "Z": target_data["Z"],
    }

    return target_data


def calc_states(indf, env_name):
    # read data
    print("read data.................")
    target_data = read_data(indf)
    # calc velocity
    print("calc velocity.............")
    Vx, Vy, Vz = calc_velocities(target_data)
    # calc angle
    print("calc angle................")
    theta, alpha = calc_angles(target_data)
    # calc rotation point
    print("calc rotation point.......")
    pos_x, pos_z = calc_rotation_point(target_data)
    # calc cross point
    print("calc cross points.........")
    cross_distance = calc_cross_points(target_data, pos_x, pos_z, env_name)

    return Vx, Vy, Vz, theta, alpha, cross_distance, target_data["dt"], target_data


def calc_actions(indf):
    target_data = read_data(indf)
    actions = []
    for x, y, z in zip(
        target_data["X"][1:], target_data["Y"][1:], target_data["Z"][1:]
    ):
        actions.append([x, y, z])
    return actions


def calc_rewards(indf):
    target_data = read_data(indf)
    rewards = [0]
    theta, _ = calc_angles(target_data)
    for i in range(len(theta) - 1):
        rewards.append(theta[i + 1] - theta[i])
    return rewards


def preprocess_bat(input_data, episode_sec):
    states, actions, rewards, lengths, conditions, vel_abss = [], [], [], [], [], []
    yubi_ID = [100, 101, 102, 103]
    # kiku_ID = [200, 201, 202, 203, 204]
    target_env_list = glob.glob(
        f"{input_data}/*"
    )  # target_data == "OneDrive - 同志社大学\源田会\data\藤井先生\ユビ\2023"　想定
    for idx, target_env in enumerate(target_env_list):
        print(f"target_env:{os.path.split(target_env)[-1]}")
        target_bat_list = glob.glob(f"{target_env}/*")
        env_name = os.path.split(target_env)[-1]
        # testのため一旦障害物なしで
        # env_name = "Test"
        for target_bat, target_bat_ID in zip(target_bat_list, yubi_ID):
            print(f"target_bat:{os.path.split(target_bat)[-1]}")
            target_data_list = glob.glob(f"{target_bat}/*.csv")
            bat_name = os.path.split(target_bat)[-1]
            for target_data in target_data_list:
                fname = os.path.split(target_data)[1].split(".csv")[0]
                indf = pd.read_csv(target_data)
                # import pdb; pdb.set_trace()
                Vx, Vy, Vz, theta, alpha, cross_distance, dt, rawdata = calc_states(
                    indf, env_name
                )
                states.append(cross_distance)
                action = calc_actions(indf)
                reward = calc_rewards(indf)
                length = len(reward)
                actions.append(action)
                rewards.append(reward)
                lengths.append(length)
                conditions.append(idx)
                vel_abss.append(
                    np.sqrt(np.array(Vx) ** 2 + np.array(Vy) ** 2 + np.array(Vz) ** 2)
                )
                print('finish calculation........')

                df = pd.DataFrame()
                df["Time (Seconds)"] = rawdata["Time"][:-2].tolist()
                df["X"] = rawdata["X"][:-2].tolist()
                df["Y"] = rawdata["Y"][:-2].tolist()
                df["Z"] = rawdata["Z"][:-2].tolist()
                print('finish position...........')
                df["Vx"] = Vx[:-2]
                df["Vy"] = Vy[:-2]
                df["Vz"] = Vz[:-2]
                print('finish velocity...........')
                df["theta"] = theta[:-2]
                df["alpha"] = alpha[:-2]
                df["Env"] = [env_name.replace("Env", "") for i in range(len(df["X"]))]
                df["Bat"] = [target_bat_ID for i in range(len(df["X"]))]
                print('finish angle..............')
                # import pdb; pdb.set_trace()
                cross_distance = np.array(cross_distance).T
                cross_distance = cross_distance.tolist()
                for i in range(251):
                    df["distance obs {}".format(i + 1)] = cross_distance[:][i]
                df.to_csv("./calcdata/{}/{}/path/{}_{}_{}.csv".format(env_name, bat_name, env_name, bat_name, fname), index=None)
                print(f"finish {fname} ..........")

    # states_arr = np.array(states)
    # import matplotlib.pyplot as plt
    # plt.plot(states_arr[0][0])
    # plt.savefig("tmp.png")
    # dt = 0.01
    # cut_samples_len = episode_sec // dt
    # states = cut_for_episode(states, cut_samples_len)
    # actions = cut_for_episode(actions, cut_samples_len)
    # rewards = cut_for_episode(rewards, cut_samples_len)
    # vel_abss = cut_for_episode(vel_abss, cut_samples_len)

    return states, actions, rewards, lengths, conditions, vel_abss


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data_path", type=str, default="../AniMARL_data/", help="data path"
    )
    args = parser.parse_args()
    input_data_path = args.data_path  # not fix yet, data is .csv.

    (
        states,
        actions,
        rewards,
        lengths,
        conditions,
        vel_abss,
    ) = preprocess_bat(input_data_path, 8.01)

    # import pdb; pdb.set_trace()

    # df = pd.DataFrame()

    # for i in range(len(states)):
    #     df["distance obs {}".format(i + 1)] = states[i]

    # os.mkdir("./calc/")

    # df.to_csv("./calcdata/yubi/{}.csv".format(path))