import numpy as np
import torch
import json
from calc_states.preprocess_bats.obstacle_information.wall import (
    wall_x,
    wall_y,
)

# from obstacle_information.Regular_20230429 import obs_frame_x, obs_frame_y


def calc_each_point(obs_point_dict_env):
    obs_x = obs_point_dict_env["x"]
    obs_y = obs_point_dict_env["y"]
    obs_frame_x = []
    obs_frame_y = []
    for x, y in zip(obs_x, obs_y):
        x0 = x - 0.015
        y0 = y - 0.015
        x1 = x + 0.015
        y1 = y - 0.015
        x2 = x + 0.015
        y2 = y + 0.015
        x3 = x - 0.015
        y3 = y + 0.015
        frame_x = [x0, x1, x2, x3, x0]
        frame_y = [y0, y1, y2, y3, y0]
        obs_frame_x.append(frame_x)
        obs_frame_y.append(frame_y)

    return obs_frame_x, obs_frame_y


def calc_obs_area(obs_point_dict):
    """
    障害物の座標を算出
    """
    obs_area_dict = {
        "wall": {"x": [], "y": []},
        "Env1": {"x": [], "y": []},
        "Env2": {"x": [], "y": []},
        "Env3": {"x": [], "y": []},
        "Env4": {"x": [], "y": []},
        "Env5": {"x": [], "y": []},
        "Env6": {"x": [], "y": []},
        "Env7": {"x": [], "y": []},
        "Test": {"x": [], "y": []},
    }
    for env in obs_point_dict.keys():
        if env == "wall":
            obs_frame_x = [obs_point_dict[env]["x"]]
            obs_frame_y = [obs_point_dict[env]["y"]]
        else:
            (
                obs_frame_x,
                obs_frame_y,
            ) = calc_each_point(obs_point_dict[env])
        obs_area_dict[env]["x"] = obs_frame_x
        obs_area_dict[env]["y"] = obs_frame_y

    return obs_area_dict


# 2023の障害物情報
obs_point_dict = json.load(
    open(
        "calc_states/preprocess_bats/obstacle_information/2023/Envs_yubi.json",
        "r",
    )
)
obs_area_dict = calc_obs_area(obs_point_dict)


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


def vel(P, dt=0.01):
    """
    速度を算出
    P:list
    """
    vel = []
    for i in range(len(P) - 1):
        vel.append((P[i + 1] - P[i]) / dt)
        if i == len(P) - 2:
            vel.append(0)

    return vel


def horizon_angle(x, y):
    """
    水平方向の角度を算出
    入力のXとYはリストを想定
    """

    vec1 = [x[1] - x[0], y[1] - y[0]]
    vec2 = [x[2] - x[1], y[2] - y[1]]
    if np.linalg.norm(vec1) == 0 or np.linalg.norm(vec2) == 0:
        import pdb

        pdb.set_trace()
    absvec1 = np.linalg.norm(vec1)
    absvec2 = np.linalg.norm(vec2)
    inner = np.dot(vec1, vec2)
    cos_theta = inner / (absvec1 * absvec2)
    rad = np.arccos(np.clip(cos_theta, -1.0, 1.0))
    theta = np.rad2deg(rad)

    return theta


def vertical_angle(x, y, z):
    """
    垂直方向の角度を算出
    入力のX,Y,Zはリストを想定
    higashisalary.com/entry/numpy-angle-calc
    """

    vec1 = [
        x[1] - x[0],
        y[1] - y[0],
        z[1] - z[0],
    ]
    vec2 = [
        x[2] - x[1],
        y[2] - y[1],
        z[2] - z[1],
    ]
    absvec1 = np.linalg.norm(vec1)
    absvec2 = np.linalg.norm(vec2)
    inner = np.dot(vec1, vec2)
    cos_theta = inner / (absvec1 * absvec2)
    rad = np.arccos(np.clip(cos_theta, -1.0, 1.0))
    theta = np.rad2deg(rad)

    return theta


def rotation(x: list, y: list):
    """
    座標を±40°回転
    """
    rads = [np.deg2rad(round(j * 0.01, 1)) for j in range(-4000, 4032, 32)]
    rot_x = []
    rot_y = []
    length = 5
    for i in range(len(x) - 1):
        theta = np.arctan2((y[i + 1] - y[i]), (x[i + 1] - x[i]))
        rads += theta
        cos_list = np.cos(rads)
        sin_list = np.sin(rads)
        r = np.sqrt((x[i + 1] - x[i]) ** 2 + (y[i + 1] - y[i]) ** 2)
        # rot_x.append(((x[i+1] - x[i]) * cos_list) - ((y[i+1] - y[i]) * sin_list)+ x[i])
        # rot_y.append(((x[i+1] - x[i]) * sin_list) + ((y[i+1] - y[i]) * cos_list)+ y[i])
        rot_x.append(length * cos_list + x[i])
        rot_y.append(length * sin_list + y[i])

    return rot_x, rot_y


def calc_cross_point(x1, y1, x2, y2, x3, y3, x4, y4):
    """
    交点を算出
    x1, y1, x2, y2:bat
    x3, y3, x4, y4:obs
    """
    den = (x2 - x1) * (y4 - y3) - (y2 - y1) * (x4 - x3)
    if den == 0:
        dis_norm = np.nan
    else:  # https://www.hiramine.com/programming/graphics/2d_segmentintersection.html
        r = ((y4 - y3) * (x3 - x1) - (x4 - x3) * (y3 - y1)) / den
        s = ((y2 - y1) * (x3 - x1) - (x2 - x1) * (y3 - y1)) / den

        if 0 <= r <= 1 and 0 <= s <= 1:
            dis_norm = r
        else:
            dis_norm = np.nan

    return round(dis_norm, 3)


def cross_point(
    x: list,
    y: list,
    rot_x: list,
    rot_y: list,
    env_num: int,
    bat_species: int,
):
    """
    交点までの距離を算出
    """
    env_name = get_env_name(env_num)
    if bat_species >= 200:
        obs_area_dict = json.load(
            open(
                "./calc_states/preprocess_bats/obstacle_information/2023/Envs_kiku.json",
                "r",
            )
        )
    elif bat_species >= 100 and bat_species < 200:
        obs_area_dict = json.load(
            open(
                "./calc_states/preprocess_bats/obstacle_information/2023/Envs_yubi.json",
                "r",
            )
        )
    obs_x_list = obs_area_dict["wall"]["x"] + obs_area_dict[env_name]["x"]
    obs_y_list = obs_area_dict["wall"]["y"] + obs_area_dict[env_name]["y"]

    cross_alldistance = []
    for i, (x0, y0) in enumerate(zip(x, y)):  ### bat position
        cross_subdistance = []
        for x1, y1 in zip(rot_x[i], rot_y[i]):  ### 251
            first_dis = 10
            for j in range(len(obs_x_list)):
                for k in range(len(obs_x_list[0]) - 1):
                    x2 = obs_x_list[j][k]
                    y2 = obs_y_list[j][k]
                    x3 = obs_x_list[j][k + 1]
                    y3 = obs_y_list[j][k + 1]
                    r = calc_cross_point(
                        x0,
                        y0,
                        x1,
                        y1,
                        x2,
                        y2,
                        x3,
                        y3,
                    )
                    if r < first_dis:
                        first_dis = r
                    else:
                        pass
            if first_dis == 10:
                first_dis = 0
            cross_subdistance.append(first_dis)
        cross_alldistance.append(cross_subdistance)

    return cross_alldistance
