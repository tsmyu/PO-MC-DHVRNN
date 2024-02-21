import glob
import os
import numpy as np
import pandas as pd
from natsort import natsorted


def cut_for_episode(data_list, samples):
    """
    step + 1 の倍数だけデータをカット
    input : data_list : すべての個体ごとのデータのlist
            samples：カットするサンプル数
    """
    cut_data_list = []
    for target_data in data_list:
        n_episode = len(target_data) // samples
        for n in np.arange(n_episode):
            cut_data = target_data[n * samples : (n + 1) * samples]
            cut_data_list.append(cut_data)

    return cut_data_list

def episode(seconds, df):
    """
    step + 1 の倍数だけデータをカット
    """
    episode = seconds
    len_df = len(df)
    
    q = len_df // episode
    
    if q != 0:
        out_df = df[:episode*q]
        
    return out_df

def dwnsmp(df):
    """
    100Hz -> 10Hzにダウンサンプリング
    """
    dwn_df = df[::10]

    return dwn_df

if __name__ == '__main__':
    target_env_list = glob.glob("./calcdata/*")
    for idx, target_env in enumerate(target_env_list):
        # print(f"target_env:{os.path.split(target_env)[-1]}")
        target_bat_list = glob.glob(f"{target_env}/*")
        env_name = os.path.split(target_env)[-1]
        for target_bat in target_bat_list:
            # print(f"target_bat:{os.path.split(target_bat)[-1]}")
            target_data_list = glob.glob(f"{target_bat}/dummy_flag/*.csv")
            bat_name = os.path.split(target_bat)[-1]
            for target_data in target_data_list:
                fname = os.path.split(target_data)[1].split(".csv")[0]
                df = pd.read_csv(target_data)
                cutdf = episode(810, df)
                downdf = dwnsmp(cutdf)
                downdf.to_csv('./calcdata/{}/{}/dummy_flag_cutdown/{}.csv'.format(env_name, bat_name, fname), index=None)
                print('finish {}'.format(fname))

    # env_name = "Env7"
    # bat_name = "BatD"
    # fname = "Env7_BatD_no5"

    # df = pd.read_csv(f"./../{env_name}_{bat_name}/dummy_flag/{fname}.csv")
    # cutdf = episode(810, df)
    # downdf = dwnsmp(cutdf)
    # downdf.to_csv(f"./../{env_name}_{bat_name}/dummy_flag_cutdown/{fname}.csv", index=None)
    # print('finish {}'.format(fname))