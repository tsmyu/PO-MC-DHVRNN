import os, glob
import pandas as pd
import statistics
from natsort import natsorted
import json


# PATH = './dataset/'

# timing = PATH + 'voice/'
# path = PATH + 'path/cut/'

# timing_files = glob.glob(timing + '*.csv')
# path_files = natsorted(glob.glob('./teshima/train/calc/0/cutpath/*.csv'))

# for path_file in path_files:
#     pulse_fname = os.path.split(path_file)[1].split('.csv')[0]
#     padf = pd.read_csv(path_file)
#     vodf = pd.read_csv('./teshima/train/calc/0/voice/{}.csv'.format(pulse_fname), header=None)

#     time = padf['Time (Seconds)'].tolist()
#     pulse = vodf.iloc[:, 0].tolist()

#     padf['pulse'] = 0

#     for i in pulse:    
#         padf.loc[padf['Time (Seconds)']==i, 'pulse'] = 1

#     # print(sum(padf['pulse']))
    
#     padf.to_csv('./teshima/train/calc/0/combine/{}.csv'.format(pulse_fname), index=None)
#     print('finish {}'.format(pulse_fname))


##########
#### fix pulse time !!!!!!!!!!!!
##########

trigger_json = json.load(open("./calcdata/start_time.json", "r"))
target_env_list = glob.glob("./calcdata/*")
for idx, target_env in enumerate(target_env_list):
    print(f"target_env:{os.path.split(target_env)[-1]}")
    target_bat_list = glob.glob(f"{target_env}/*")
    env_name = os.path.split(target_env)[-1]
    for target_bat in target_bat_list:
        print(f"target_bat:{os.path.split(target_bat)[-1]}")
        target_pulse_list = glob.glob(f"{target_bat}/pulse/*.csv")
        target_path_list = glob.glob(f"{target_bat}/path/*.csv")
        bat_name = os.path.split(target_bat)[-1]
        for target_pulse in target_pulse_list:
            pulse_fname = os.path.split(target_pulse)[1].split(".csv")[0]
            pulse_df = pd.read_csv(target_pulse)
            for target_path in target_path_list:
                path_fname = os.path.split(target_pulse)[1].split(".csv")[0]
                if pulse_fname in path_fname:
                    path_df = pd.read_csv(f"{target_bat}/path/{pulse_fname}.csv")
                    pulse = pulse_df['StartTiming[s]'].tolist()
                    trigger_time = trigger_json[f"{pulse_fname}"]
                    pulse_trigger = [i - trigger_time for i in pulse]
                    time = path_df.iloc[:, 0].tolist()
                    
                    path_df['pulse'] = 0

                    for i in pulse_trigger:    
                        path_df.loc[path_df['Time (Seconds)']==i, 'pulse'] = 1
                    
                    path_df.to_csv(f"{target_bat}/combine/{pulse_fname}.csv", index=None)
                    print('finish {}'.format(pulse_fname))
                else:
                    pass