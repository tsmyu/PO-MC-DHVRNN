import os, glob
import pandas as pd

kiku_ID = [200, 201, 202, 203, 204]

target_env_list = glob.glob("./calcdata/*")  # target_data == "OneDrive - 同志社大学\源田会\data\藤井先生\ユビ\2023"　想定
for idx, target_env in enumerate(target_env_list):
    print(f"target_env:{os.path.split(target_env)[-1]}")
    target_bat_list = glob.glob(f"{target_env}/*")
    env_name = os.path.split(target_env)[-1]
    for target_bat in target_bat_list:
        print(f"target_bat:{os.path.split(target_bat)[-1]}")
        target_data_list = glob.glob(f"{target_bat}/path/*.csv")
        bat_name = os.path.split(target_bat)[-1]
        for target_data in target_data_list:
            fname = os.path.split(target_data)[1].split(".csv")[0]
            indf = pd.read_csv(target_data)
            if bat_name == "BatA":
                indf.loc[:, "Bat"] = 200
            elif bat_name == "BatB":
                indf.loc[:, "Bat"] = 201
            elif bat_name == "BatC":
                indf.loc[:, "Bat"] = 202
            elif bat_name == "BatD":
                indf.loc[:, "Bat"] = 203
            elif bat_name == "BatE":
                indf.loc[:, "Bat"] = 204
            
            indf.to_csv(f"./calcdata/{env_name}/{bat_name}/path/{fname}.csv", index=None)