import pandas as pd

df = pd.read_csv("./dataset/pulse_flag/train/train.csv")
df_digits = df.iloc[:, 12:].round(3)

DF = pd.concat([df.iloc[:, :12], df_digits], axis=1)


DF.to_csv("./dataset/pulse_flag/train/train.csv", index=None)

# df = pd.DataFrame([[555.1111111, 555.2222222, 100.530555, 50.99922, 0.333333],
#                    [555.1111111, 555.2222222, 100.530555, 50.99922, 0.333333]])
# df_digits = df.iloc[:, 2:].round(3)
# DF = pd.concat([df.iloc[:, :2], df_digits], axis=1)
# # DF.to_csv("temp.csv")
# print(DF)
