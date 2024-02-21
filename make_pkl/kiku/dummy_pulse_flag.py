import os, glob
import pandas as pd
import statistics
from natsort import natsorted
import json


target_env_list = glob.glob("./calcdata/*")
for idx, target_env in enumerate(target_env_list):
    print(f"target_env:{os.path.split(target_env)[-1]}")
    target_bat_list = glob.glob(f"{target_env}/*")
    env_name = os.path.split(target_env)[-1]
    for target_bat in target_bat_list:
        print(f"target_bat:{os.path.split(target_bat)[-1]}")
        target_path_list = glob.glob(f"{target_bat}/path/*.csv")
        bat_name = os.path.split(target_bat)[-1]
        for target_path in target_path_list:
            path_fname = os.path.split(target_path)[1].split(".csv")[0]
            path_df = pd.read_csv(f"{target_bat}/path/{path_fname}.csv")
            path_df['pulse'] = 1
            
            path_df.to_csv(f"{target_bat}/dummy_flag/{path_fname}.csv", index=None)