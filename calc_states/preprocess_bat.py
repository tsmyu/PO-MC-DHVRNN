import glob
import os
import numpy as np
import pandas as pd
import argparse
import matplotlib
import time

matplotlib.use("Agg")

# from natsort import natsorted
print(os.getcwd())

from calc_states.preprocess_bats.calculation import (
    vel,
    horizon_angle,
    vertical_angle,
    rotation,
    cross_point,
)


def calc_cross_points(
    prev_f,
    pos_x,
    pos_z,
    env_name,
    obs_point_dict,
):
    X = [prev_f[0]]
    Y = [prev_f[1]]

    cross_list = cross_point(
        X,
        Y,
        pos_x,
        pos_z,
        env_name,
        obs_point_dict,
    )

    return cross_list


def calc_rotation_point(prev_f, next_point, dim):
    if dim == 2:
        X = [
            prev_f[0],
            next_point[0],
        ]
        Y = [
            prev_f[1],
            next_point[1],
        ]
        rot_all_x, rot_all_y = rotation(X, Y)
    elif dim == 3:
        X = [
            prev_f[0],
            next_point[0],
        ]
        Y = [
            prev_f[1],
            next_point[1],
        ]
        Z = [
            prev_f[2],
            next_point[2],
        ]
        rot_all_x, rot_all_y = rotation(X, Y, Z)

    return rot_all_x, rot_all_y


def calc_angles(prev_f, prev_point, next_point, dim):
    if dim == 2:
        X = [
            prev_point[0],
            prev_f[0],
            next_point[0],
        ]
        Y = [
            prev_point[1],
            prev_f[1],
            next_point[1],
        ]
        theta = horizon_angle(X, Y)
    elif dim == 3:
        X = [
            prev_point[0],
            prev_f[0],
            next_point[0],
        ]
        Y = [
            prev_point[1],
            prev_f[1],
            next_point[1],
        ]
        Z = [
            prev_point[2],
            prev_f[2],
            next_point[2],
        ]
        theta = horizon_angle(X, Y, Z)

    return theta


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


def calc_states(
    prev_f,
    prev_point,
    next_point,
    dim,
    pulse_flag,
    obs_point_dict,
):
    env_name = int(prev_f[6])
    if dim == 3:
        print("not concider in 3-dim yet")
        exit()
    theta = calc_angles(prev_f, prev_point, next_point, dim)
    # calc rotation point
    pos_x, pos_z = calc_rotation_point(prev_f, next_point, dim)
    if pulse_flag:
        # calc cross point
        cross_distance = calc_cross_points(
            prev_f,
            pos_x,
            pos_z,
            env_name,
            obs_point_dict,
        )[0]
    else:
        cross_distance = [2.0 for i in range(len(pos_x[0]))]

    return theta, np.array(cross_distance)


def calc_actions(indf):
    target_data = read_data(indf)
    actions = []
    for x, y, z in zip(
        target_data["X"][1:],
        target_data["Y"][1:],
        target_data["Z"][1:],
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


def calc_bat_states(
    prev_f,
    prev_point,
    next_point,
    dim,
    pulse_flag,
    obs_point_dict,
):
    """
    prev_f = [X, Y, Vx, Vy, θ, pulse_flag, Env, Bat, state(251dim)]
    next_point = [X, Y]
    """
    prev_f = prev_f.to("cpu").detach().numpy().copy()
    prev_point = prev_point.to("cpu").detach().numpy().copy()
    next_point = next_point.to("cpu").detach().numpy().copy()
    theta, cross_distance = calc_states(
        prev_f,
        prev_point,
        next_point,
        dim,
        pulse_flag,
        obs_point_dict,
    )

    return theta, cross_distance


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data_path",
        type=str,
        default="../AniMARL_data/bat/2023",
        help="data path",
    )
    parser.add_argument(
        "--episode_sec",
        type=int,
        default=8,
        help="episode duration [sec]",
    )
    args = parser.parse_args()
    input_data_path = args.data_path  # not fix yet, data is .csv.
    episode_sec = args.episode_sec

    (
        states,
        actions,
        rewards,
        lengths,
        conditions,
        vel_abss,
    ) = preprocess_bat(input_data_path, episode_sec)

    df = pd.DataFrame()

    for i in range(len(states)):
        df["distance obs {}".format(i + 1)] = states[i]

    os.makedirs("./calc/tmp", exist_ok=True)

    df.to_csv("./calc/tmp/tmp.csv")
