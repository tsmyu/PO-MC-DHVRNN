import glob
import os
import numpy as np
import pandas as pd
import argparse
import pickle

# from natsort import natsorted

from preprocess_bats.calculation import (
    vel,
    horizon_angle,
    vertical_angle,
    rotation,
    cross_point
)

from preprocess_bats.cut_down import cut_for_episode, dwnsmp


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

    return Vx, Vy, Vz, theta, alpha, cross_distance, target_data["dt"]


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
    
    if os.path.isfile(f"{input_data}/status.pkl"):
        with open(f"{input_data}/status.pkl", 'rb') as f:
            states, actions, rewards, lengths, conditions, vel_abss = np.load(
                f, allow_pickle=True)
    else:
        states, actions, rewards, lengths, conditions, vel_abss = [], [], [], [], [], []
        target_env_list = glob.glob(
            f"{input_data}/*"
        )  # target_data == "OneDrive - 同志社大学\源田会\data\藤井先生\ユビ\2023"　想定
        print(target_env_list)
        for idx, target_env in enumerate(target_env_list):
            print(f"target_env:{os.path.split(target_env)[-1]}")
            target_bat_list = glob.glob(f"{target_env}/*")
            env_name = os.path.split(target_env)[-1]
            # testのため一旦障害物なしで
            # env_name = "Test"
            for target_bat in target_bat_list:
                print(f"target_bat:{os.path.split(target_bat)[-1]}")
                target_data_list = glob.glob(f"{target_bat}/*.csv")
                for target_data in target_data_list:
                    fname = os.path.split(target_data)[1].split(".csv")[0]
                    indf = pd.read_csv(target_data)
                    # import pdb; pdb.set_trace()
                    Vx, Vy, Vz, theta, alpha, cross_distance, dt = calc_states(
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
                    print(f"finish {fname}")
        states_arr = np.array(states)
        import matplotlib.pyplot as plt
        plt.plot(states_arr[0][0])
        plt.savefig("tmp.png")
        cut_samples_len = episode_sec // dt
        states = cut_for_episode(states, cut_samples_len)
        actions = cut_for_episode(actions, cut_samples_len)
        rewards = cut_for_episode(rewards, cut_samples_len)
        vel_abss = cut_for_episode(vel_abss, cut_samples_len)

        with open(f"{input_data}/status.pkl", 'wb') as f:
            pickle.dump([states, actions, rewards, lengths, conditions, vel_abss], f, protocol=4)

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
    ) = preprocess_bat(input_data_path)

    df = pd.DataFrame()

    for i in range(len(states)):
        df["distance obs {}".format(i + 1)] = states[i]

    os.mkdir("./calc/tmp")

    df.to_csv("./calc/tmp/tmp.csv")
