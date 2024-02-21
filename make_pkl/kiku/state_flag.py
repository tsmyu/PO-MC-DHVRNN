import pandas as pd
import glob, os

target_env_list = glob.glob("./calcdata/*")
for idx, target_env in enumerate(target_env_list):
    # print(f"target_env:{os.path.split(target_env)[-1]}")
    target_bat_list = glob.glob(f"{target_env}/*")
    env_name = os.path.split(target_env)[-1]
    for target_bat in target_bat_list:
        # print(f"target_bat:{os.path.split(target_bat)[-1]}")
        target_data_list = glob.glob(f"{target_bat}/combine/*.csv")
        bat_name = os.path.split(target_bat)[-1]
        for target_data in target_data_list:
            fname = os.path.split(target_data)[1].split(".csv")[0]
            df = pd.read_csv(target_data)

            df.iloc[df["pulse"] == 0, 11:-1] = 2
            # df.mask(df.iloc[:, 11:-1] == 0, 2)

            # print(df.iloc[:, 6:])

            df.to_csv('./calcdata/{}/{}/flag/{}.csv'.format(env_name, bat_name, fname), index=None)
            print("finish {}".format(fname))