import pandas as pd
import glob, os

path_files = glob.glob('./teshima/test/calc/combine/*.csv')
for path_file in path_files:
    fname = os.path.split(path_file)[1].split('.csv')[0]
    df = pd.read_csv(path_file)

    for i in range(len(df)-1):
        if df.iloc[i, -1] == df.iloc[i+1, -1]:
            df.iloc[i+1, -1] == 0
    
    df.to_csv('./teshima/test/calc/combine/{}.csv'.format(fname), index=None)
    
    print('finish {}'.format(fname))