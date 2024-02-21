import numpy as np
import csv
import pickle
import glob
import pandas as pd 


step = 801 
flag = "pulse_flag"

#####TRAIN#####
episodes = 208 ###pulse_flag:212 dummy_flag:208
# data = np.loadtxt('./dataset/train/train.csv', delimiter=',')
data = pd.read_csv(f'./dataset/{flag}/train/train.csv', header=None)
data = data.values


alldata = data.reshape(1, episodes, step, 259)


with open(f'./dataset/{flag}/train/kiku_train.pkl', 'wb') as f:
    pickle.dump(alldata, f, protocol=4)



# # #####TEST#####
# episodes = 88 ###pulse_flag:88 dummy_flag:135
# # data = np.loadtxt('./dataset/test/test.csv', delimiter=',')
# data = pd.read_csv(f'./dataset/{flag}/test/test.csv', header=None)
# data = data.values


# alldata = data.reshape(1, episodes, step, 259)

# with open(f'./dataset/{flag}/test/kiku_test.pkl', 'wb') as f:
#     pickle.dump(alldata, f, protocol=4)