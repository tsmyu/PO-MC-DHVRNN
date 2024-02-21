import pandas as pd
import glob
import os
from natsort import natsorted

data_list = []

path = './dataset/pulse_flag/train'
files = natsorted(glob.glob(path + '/*.csv'))
for file in files:
    data_list.append(pd.read_csv(file))
    print(file)

df = pd.concat(data_list, axis=0)
df.to_csv('./dataset/pulse_flag/train/train.csv', index=None)

# data_list = []
# path = './dataset/dummy_flag/train'
# files = natsorted(glob.glob(path + '/*.csv'))
# for file in files:
#     data_list.append(pd.read_csv(file))
#     print(file)

# df = pd.concat(data_list, axis=0)
# df.to_csv(path + '/train.csv', index=None)