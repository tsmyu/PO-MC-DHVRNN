import pandas as pd

flag = "dummy_flag"
tetr = "train"

path = f"./dataset/{flag}/{tetr}/{tetr}.csv"

indf = pd.read_csv(path, header = None)

indf.iloc[:, 8:70] = 2
indf.iloc[:, 196:] = 2

outdf = indf.to_csv(path, index=False, header = False)