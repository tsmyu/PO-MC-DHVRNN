import glob
import os
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import json
from natsort import natsorted
from matplotlib.backends.backend_pdf import PdfPages

# pp = PdfPages("./flightpath/flightpath.pdf")

target_env_list = glob.glob("./rawdata/*")  # target_data == "OneDrive - 同志社大学\源田会\data\藤井先生\ユビ\2023"　想定
for idx, target_env in enumerate(target_env_list):
    print(f"target_env:{os.path.split(target_env)[-1]}")
    target_bat_list = glob.glob(f"{target_env}/*")
    env_name = os.path.split(target_env)[-1]

    chainDATA = json.load(open("./obstacle_information/Envs.json", "r"))
    # chaindf = pd.read_csv(chainpath)
    chain_x = chainDATA[env_name]["x"]
    chain_y = chainDATA[env_name]["y"]

    fig1 = plt.figure(figsize=(6,6))
    ax1 = fig1.add_subplot(111)
    ax1.scatter(chain_x, chain_y, color='#ff7f0e')
    ax1.set_xlabel('X [m]')
    ax1.set_ylabel('Y [m]')
    ax1.set_xticks([0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5])
    ax1.set_yticks([0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5])
    ax1.set_aspect('equal')

    plt.title(f"{env_name} obstacle position")
    plt.tight_layout()
    plt.savefig(f"./flightpath/{env_name}/obstacle.png")
    # pp.savefig(fig1)

    for target_bat in target_bat_list:
        print(f"target_bat:{os.path.split(target_bat)[-1]}")
        target_data_list = glob.glob(f"{target_bat}/*.csv")
        bat_name = os.path.split(target_bat)[-1]
        X = []
        Y = []
        for target_data in target_data_list:
            fname = os.path.split(target_data)[1].split(".csv")[0]
            indf = pd.read_csv(target_data)

            x = indf['X']
            y = indf['Z']
            X.append(x)
            Y.append(y)

        fig2 = plt.figure(figsize=(6,6))
        ax2 = fig2.add_subplot(111)
        for i in range(len(X)):
            ax2.plot(X[i], Y[i], color='#1f77b4')
        ax2.scatter(chain_x, chain_y, color='#ff7f0e')
        ax2.set_xlabel('X [m]')
        ax2.set_ylabel('Y [m]')
        ax2.set_xticks([0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5])
        ax2.set_yticks([0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5])
        ax2.set_aspect('equal')

        plt.title(f"{env_name} {bat_name} flight path")
        plt.tight_layout()
        plt.savefig(f"./flightpath/{env_name}/{bat_name}.png")
        # pp.savefig(fig2)

        print('finish {} - {}'.format(env_name, bat_name))
    # pp.close()