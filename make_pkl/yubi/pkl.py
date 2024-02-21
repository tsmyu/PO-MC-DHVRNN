import numpy as np
import csv
import pickle
import glob
import pandas as pd 


step = 801

#####TRAIN#####
episodes = 281 ###pulse_flag:281 dummy_flag:856
data = pd.read_csv('./dataset/pulse_flag/train/train.csv', header=None)
data = data.values

alldata = data.reshape(1, episodes, step, 259)

with open('./dataset/pulse_flag/train/yubi_train.pkl', 'wb') as f:
    pickle.dump(alldata, f, protocol=4)



#####TEST#####
episodes = 53 ###pulse_flag:53 dummy_flag:183
data = pd.read_csv('./dataset/pulse_flag/test/test.csv', header=None)
data = data.values

alldata = data.reshape(1, episodes, step, 259)

with open('./dataset/pulse_flag/test/yubi_test.pkl', 'wb') as f:
    pickle.dump(alldata, f, protocol=4)