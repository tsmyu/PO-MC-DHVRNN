import glob
import os
import numpy as np
import pandas as pd


def cut_for_episode(data_list, samples):
    """
    step + 1 の倍数だけデータをカット
    input : data_list : すべての個体ごとのデータのlist
            samples：カットするサンプル数
    """
    cut_data_list = []
    for target_data in data_list:
        n_episode = round(len(target_data) // samples)
        if n_episode == 0:
            pass
        else:
            for n in range(n_episode):
                cut_data = target_data[int(n * samples) : int((n + 1) * samples)]
                cut_data_list.append(cut_data)

    return cut_data_list


def dwnsmp(df):
    """
    100Hz -> 10Hzにダウンサンプリング
    """
    down_df = df[::10]

    return down_df
