import os, glob
import pandas as pd


target_env_list = glob.glob("./rawdata/*")  # target_data == "OneDrive - 同志社大学\源田会\data\藤井先生\ユビ\2023"　想定
for idx, target_env in enumerate(target_env_list):
    print(f"target_env:{os.path.split(target_env)[-1]}")
    target_bat_list = glob.glob(f"{target_env}/*")
    env_name = os.path.split(target_env)[-1]
    if env_name == "Env3" or env_name == "Env6" or env_name == "Env7":
        for target_bat in target_bat_list:
            print(f"target_bat:{os.path.split(target_bat)[-1]}")
            target_data_list = glob.glob(f"{target_bat}/*.csv")
            bat_name = os.path.split(target_bat)[-1]
            for target_data in target_data_list:
                fname = os.path.split(target_data)[1].split(".csv")[0]
                indf = pd.read_csv(target_data)
                indf["X"] = indf["X"] + 0.24
                indf["Z"] = indf["Z"] + 0.28
                indf.to_csv(f"./rawdata__/{env_name}/{bat_name}/{fname}.csv", index=None)