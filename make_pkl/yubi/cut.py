import glob
import os
import numpy as np
import pandas as pd
from natsort import natsorted

def episode(seconds, df):
    """
    step + 1 の倍数だけデータをカット
    """
    episode = seconds
    len_df = len(df)
    
    q = len_df // episode
    
    if q != 0:
        cut_df = df[:episode*q]
        
    return cut_df, q

if __name__ == '__main__':
    target_env_list = glob.glob("./calcdata/*")
    for idx, target_env in enumerate(target_env_list):
        # print(f"target_env:{os.path.split(target_env)[-1]}")
        target_bat_list = glob.glob(f"{target_env}/*")
        env_name = os.path.split(target_env)[-1]
        for target_bat in target_bat_list:
            # print(f"target_bat:{os.path.split(target_bat)[-1]}")
            target_data_list = glob.glob(f"{target_bat}/flag/*.csv")
            bat_name = os.path.split(target_bat)[-1]
            for target_data in target_data_list:
                fname = os.path.split(target_data)[1].split(".csv")[0]
                df = pd.read_csv(target_data)
                cutdf, epi = episode(801, df)

                cutdf.to_csv('./calcdata/{}/{}/cut/{}.csv'.format(env_name, bat_name, fname), index=None)
                print("finish {}".format(fname))