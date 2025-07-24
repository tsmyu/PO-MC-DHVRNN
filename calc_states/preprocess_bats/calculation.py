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


def get_env_name(env_num):
    if env_num == 1:
        env_name = "Env1"
    elif env_num == 2:
        env_name = "Env2"
    elif env_num == 3:
        env_name = "Env3"
    elif env_num == 4:
        env_name = "Env4"
    else:
        raise KeyError(f"env_num={env_num}")

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


def rotation(x: list, y: list, pulse_directions: float):
    """
    座標を±40°回転
    """
    rads = [np.deg2rad(round(j*0.01, 1)) for j in range(-4000, 4032, 32)]
    rot_x = []
    rot_y = []
    length = 5
    for i in range(len(x)-1):
        base_angle_deg = pulse_directions[i]
        # 0°から360°の範囲の角度を-180°から+180°の範囲に変換
        if isinstance(base_angle_deg, torch.Tensor):
            base_angle_deg = base_angle_deg.detach().cpu().numpy()
        base_angle_deg = base_angle_deg - 360 if base_angle_deg > 180 else base_angle_deg
        base_angle_rad = np.deg2rad(base_angle_deg) # 基準角度をラジアンに変換
        rads_rotated = [rad + base_angle_rad for rad in rads]
        cos_list = np.cos(rads_rotated)
        sin_list = np.sin(rads_rotated)
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
    obs_point_dict: dict,
):
    """
    交点までの距離を算出
    """
    env_name = get_env_name(env_num)
    obs_area_dict = calc_obs_area(obs_point_dict)
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

def get_min_distance(input_array, tolerance=0.01):
    """
    251次元のリスト(cross_subdistance)を処理し、連続した障害物情報の値の中で最小値(コウモリの位置から一番短い距離)だけ取り出し、
    その他の値を 2 に置き換える関数
    引数:
        input_array (list): 入力の251次元リスト
        tolerance (float): ほとんど同じ値とみなす許容範囲。 0.01(1つの障害物にある2つの点の距離差以上)に設定
    戻り値:
        cross_subdistance2(list): 出力の251次元リスト
    """
    # 結果を格納するリスト（251次元すべて 2 で初期化）
    cross_subdistance2 = [2] * 251

    # 一時的に連続した値を保持するリスト
    temp = []
    # 現在の連続したの開始インデックスを記録
    start_index = 0

    # 入力配列を順次処理
    for i, value in enumerate(input_array):
        if value != 2:  # 値が 2 ではない場合
            if temp and abs(value - temp[-1]) > tolerance:
                # 許容範囲を超えた場合、現在の連続した値を処理してリセット
                min_value = min(temp)  # 連続した値の最小値だけ取得
                min_index = temp.index(min_value)  # 最小値のインデックスを取得
                for j, v in enumerate(temp):
                    if j == min_index:
                        cross_subdistance2[start_index + j] = min_value  # 最小値を同じインデックスに配置
                    else:
                        cross_subdistance2[start_index + j] = 2  # 他は2に置き換え
                temp = []  # 一時的に溜め込んだ連続した値をリセット
                start_index = i  # 新しい開始インデックスを更新
            temp.append(value)  # 値を一時的なリストに追加
        else: 
            if temp:  
                min_value = min(temp)  
                min_index = temp.index(min_value)  
                for j, v in enumerate(temp):
                    if j == min_index:
                        cross_subdistance2[start_index + j] = min_value  
                    else:
                        cross_subdistance2[start_index + j] = 2  
                temp = [] 
            start_index = i + 1  
            cross_subdistance2[i] = 2  # 2 をそのまま結果リストに追加

    # 最後の処理（ループ終了後）
    if temp:
        min_value = min(temp)  # 最小値を取得
        min_index = temp.index(min_value)  # 最小値のインデックスを取得
        for j, v in enumerate(temp):
            if j == min_index:
                cross_subdistance2[start_index + j] = min_value  # 最小値を同じインデックスに配置
            else:
                cross_subdistance2[start_index + j] = 2  # 他は2に置き換え

    return cross_subdistance2
