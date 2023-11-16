import pandas as pd
import glob
import os
from natsort import natsorted

Random_data = ['20221025_1', '20221025_2', '20221025_3', '20230413', '20230414', '20230510']
Regular_data = ['20221003', '20221011', '20221018', '20230420', '20230421', '20230428', '20230429']
# Regular_data = ['20230420', '20230421', '20230428', '20230429']

BAT = ['601', '603', '604', '605', '2686', '2690']

data_list = []

#########TRAIN#########
# for data in Regular_data:
#     for bat in BAT:
#         path = './{}/mc/csv/12s_down/{}'.format(data, bat)
#         files = natsorted(glob.glob(path + '/*.csv'))
#         for file in files:
#             data_list.append(pd.read_csv(file))
#             print(file)

# df = pd.concat(data_list, axis=0)
# df.to_csv('./toPKL/TR-allRandom/2d/12s/test/test.csv', index=False)

#########TEST#########
path = './toPKL/TR-allRandom/2d/12s/test/virtual'
files = natsorted(glob.glob(path + '/*.csv'))
for file in files:
    data_list.append(pd.read_csv(file))
    print(file)

df = pd.concat(data_list, axis=0)
df.to_csv(path + '/test.csv')