import matplotlib.pyplot as plt
import numpy as np
import os
import glob
import pandas as pd
import statistics
#from natsort import natsorted
from matplotlib.backends.backend_pdf import PdfPages


path = r"C:\Users\yota-\OneDrive - 同志社大学\PO-MC-DHVRNN\result\20251003"
with open(os.path.join(path, 'params.p'), 'rb') as f: ###########
    param = np.load(f, allow_pickle=True)
    # print(param)
    predict_time = param['burn_in']

with open(os.path.join(path, 'samples.p'), 'rb') as f: ############
    data = np.load(f, allow_pickle=True)

# import pdb; pdb.set_trace()

pp = PdfPages(os.path.join(path, 'topview.pdf')) ##########

count = 0
# ループの前で合計用の変数を初期化
total_zeros = 0
total_ones = 0

all_pulse = []
pos_me = []
pos_pre = []
vel_me = []
vel_pre = []
loss_pos = []
loss_vel = []

for episode in range(len(data[0][0][0][0])):
    train_x = [] ###burn_in
    train_y = [] ###burn_in
    train_x_pre = []
    train_y_pre = []
    test_x = []
    test_y = []
    train_pulse = []
    test_pulse = []

    count += 1

    for step in range(predict_time+1): ###burn_in
        train_x.append(data[1][0][step][0][episode][0])
        train_y.append(data[1][0][step][0][episode][1])
    for step in range(predict_time, len(data[0][0])):
        train_x_pre.append(data[1][0][step][0][episode][0])
        train_y_pre.append(data[1][0][step][0][episode][1])
        test_x.append(data[0][0][step][0][episode][0])
        test_y.append(data[0][0][step][0][episode][1])
        train_pulse.append(data[1][0][step][0][episode][5])
        test_pulse.append(data[0][0][step][0][episode][7])
        all_pulse.append(data[0][0][step][0][episode][7])

        pos_me.append(np.sqrt(data[1][0][step][0][episode][0]** 2 + data[1][0][step][0][episode][1] ** 2))
        pos_pre.append(np.sqrt(data[0][0][step][0][episode][0]** 2 + data[0][0][step][0][episode][1] ** 2))
        vel_me.append(np.sqrt(data[1][0][step][0][episode][2]** 2 + data[1][0][step][0][episode][3] ** 2))
        vel_pre.append(np.sqrt(data[0][0][step][0][episode][2]** 2 + data[0][0][step][0][episode][3] ** 2))
        loss_pos.append(np.sqrt((data[1][0][step][0][episode][0] - data[0][0][step][0][episode][0])** 2 + 
                                (data[1][0][step][0][episode][1] - data[0][0][step][0][episode][1]) ** 2))
        loss_vel.append(np.sqrt((data[1][0][step][0][episode][2] - data[0][0][step][0][episode][2])** 2 + 
                                (data[1][0][step][0][episode][3] - data[0][0][step][0][episode][3]) ** 2))
    #print(data[0][0][step][0][episode][7])
    #print(data[0][0][step][0][episode][7])
    #print("====================")

    # 0.5以上の値を1に変換
    test_pulse_replace = [1 if i >= 0.5 else 0 for i in test_pulse]
    #print(test_pulse_replace)
    # 0と1の数をカウント
    num_zeros = train_pulse.count(0)
    num_ones = train_pulse.count(1)
    #print(f"0の数: {num_zeros}, 1の数: {num_ones}")
    num_zeros = train_pulse.count(0) + test_pulse.count(0)
    num_ones = train_pulse.count(1) + test_pulse.count(1)
    #print(f"Episode {episode}: 0の数: {num_zeros}, 1の数: {num_ones}")
    # 合計に加算
    total_zeros += num_zeros
    total_ones += num_ones

    #パルス放射(1)を抽出
    test_pulse_replace_1 = [i for i , x in enumerate(test_pulse_replace) if x == 1]
    #print(test_pulse_replace_1)
    IPI = [j-i for i, j in zip(test_pulse_replace_1[:-1], test_pulse_replace_1[1:])]
    #print(IPI)



    #print(len(train_pulse_replace_1))



    #print(data[1][0][step][0][episode][7])

    # print(test_pulse)
    # print(train_pulse.count(0))

    # import pdb; pdb.set_trace()

    fig = plt.figure(figsize=(4.5, 7.5))
    ax = fig.add_subplot(111)

####### yubi #######
    #if count <= 3:
    #    from obstacle_information.Env1 import obs_x, obs_y
    #    ax.scatter(obs_x, obs_y, marker='o', label='chain', color='#ff7f0e', s=15, zorder=1)
    #elif count >= 4 and count <= 13:
    #    from obstacle_information.Env2 import obs_x, obs_y
    #    ax.scatter(obs_x, obs_y, marker='o', label='chain', color='#ff7f0e', s=15, zorder=1)
    #elif count >= 14 and count <= 15:
    #    from obstacle_information.Env3 import obs_x, obs_y
    #    ax.scatter(obs_x, obs_y, marker='o', label='chain', color='#ff7f0e', s=15, zorder=1)
    #elif count >= 16:
    #    from obstacle_information.Env4 import obs_x, obs_y
    #    ax.scatter(obs_x, obs_y, marker='o', label='chain', color='#ff7f0e', s=15, zorder=1)
    if count >= 0 and count <= 25:
        from obstacle_information.Env3 import obs_x, obs_y
        ax.scatter(obs_x, obs_y, marker='o', label='chain', color='#ff7f0e', s=15, zorder=1)

####### kiku #######
    #if count <= 35:
    #    from obstacle_information.obstacle.Env4 import obs_x, obs_y
    #    ax.scatter(obs_x, obs_y, marker='o', label='chain', color='#ff7f0e', s=25, zorder=1)
    #elif count >= 36 and count <= 56:
    #    from obstacle_information.obstacle.Env5 import obs_x, obs_y
    #    ax.scatter(obs_x, obs_y, marker='o', label='chain', color='#ff7f0e', s=25, zorder=1)
    #elif count >= 57 and count <= 72:
    #    from obstacle_information.obstacle.Env6 import obs_x, obs_y
    #    ax.scatter(obs_x, obs_y, marker='o', label='chain', color='#ff7f0e', s=25, zorder=1)
    #elif count >= 73:
    #    from obstacle_information.obstacle.Env7 import obs_x, obs_y
    #    ax.scatter(obs_x, obs_y, marker='o', label='chain', color='#ff7f0e', s=25, zorder=1)

    ax.plot(train_x, train_y, label='input flight path',
            color='#2ca02c', linestyle='--', zorder=2, linewidth=2)
    ax.plot(train_x_pre, train_y_pre, label='measured flight path',
            color='#1f77b4', zorder=2, linewidth=3,)
    ax.plot(test_x, test_y, label='predicted flight path',
            color='#d62728', zorder=2, linewidth=3)
    
    for i in range(len(train_x_pre)):
        x = train_x_pre[i]
        y = train_y_pre[i]
        if train_pulse[i] >= 0.5:
            angle_deg = data[1][0][predict_time + i][0][episode][7]  # 実測Pxy (度数)
            angle_rad = np.deg2rad(angle_deg) # 度数からラジアンに変換
            dx = np.cos(angle_rad) * 0.2
            dy = np.sin(angle_rad) * 0.2
            ax.arrow(x, y, dx, dy, head_width=0.1, head_length=0.1, fc='b', ec='b')

    for i in range(len(test_x)):
        x = test_x[i]
        y = test_y[i]
        # パルス放射タイミングのみ描画
        if train_pulse[i] >= 0.5:
            angle_test = data[0][0][predict_time + i][0][episode][9]  # 実測Pxyを使用
            angle_rad_test = np.deg2rad(angle_test) # 度数からラジアンに変換
            dx = np.cos(angle_rad_test) * 0.2
            dy = np.sin(angle_rad_test) * 0.2
            ax.arrow(x, y, dx, dy, head_width=0.1, head_length=0.1, fc='r', ec='r')

    for i in range(len(train_pulse)):
        if train_pulse[i] >= 0.5:
            ax.scatter(train_x_pre[i], train_y_pre[i], label='measured pulse timing', color='k', s=30, zorder=3)
        if test_pulse[i] >= 0.5:
            ax.scatter(test_x[i], test_y[i], label='predicted pulse timing', color='w',edgecolor = '#d62728' ,s=30, zorder=3)
    #print(test_pulse[i])

    plt.xlabel('X [m]')
    plt.ylabel('Y [m]')

    plt.xticks([0, 1.0, 2.0, 3.0, 4.0, 4.5])
    plt.yticks([0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 7.5])
    plt.tight_layout()
    ax.set_aspect('equal')
    # plt.legend(bbox_to_anchor=(1,1), loc='upper left')

    pp.savefig(fig)

pp.close()

# print("mesured position mean : {}".format(statistics.mean(pos_me)))
# print("mesured position stdev : {}".format(statistics.stdev(pos_me)))
# print("predicted position mean : {}".format(statistics.mean(pos_pre)))
# print("predicted position stdev : {}".format(statistics.stdev(pos_pre)))

# print("mesured velocity mean : {}".format(statistics.mean(vel_me)))
# print("mesured velocity stdev : {}".format(statistics.stdev(vel_me)))
# print("predicted velocity mean : {}".format(statistics.mean(vel_pre)))
# print("predicted velocity stdev : {}".format(statistics.stdev(vel_pre)))

print("loss position mean : {}".format(statistics.mean(loss_pos)))
print("loss position stdev : {}".format(statistics.stdev(loss_pos)))
print("loss velocity mean : {}".format(statistics.mean(loss_vel)))
print("loss velocity stdev : {}".format(statistics.stdev(loss_vel)))
# ループ終了後に合計を表示
#print(f"全エピソードの0の合計: {total_zeros}, 1の合計: {total_ones}")