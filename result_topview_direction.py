import matplotlib.pyplot as plt
import numpy as np
import os
import glob
import pandas as pd
import statistics
#from natsort import natsorted
from matplotlib.backends.backend_pdf import PdfPages


path = r"C:\Users\yota-\OneDrive - 同志社大学\PO-MC-DHVRNN_aoki\result\20251121_yubi_vel_pulse"
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
all_angle_diffs = []

# CSV出力用のデータを格納するリスト
results_for_csv = []

for episode in range(len(data[0][0][0][0])):
    train_x = [] ###burn_in
    train_y = [] ###burn_in
    train_x_pre = []
    train_y_pre = []
    test_x = []
    test_y = []
    train_pulse = []
    test_pulse = []
    angle_diffs = []
    travel_vs_pulse_measured_diffs = []
    travel_vs_pulse_predicted_diffs = []

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
    #print(data[0][0][step][0][episode][9])
    #print("====================")

    # print(test_pulse)
    # print(train_pulse.count(0))

    # import pdb; pdb.set_trace()

    fig = plt.figure(figsize=(4.5, 7.5))
    ax = fig.add_subplot(111)

####### yubi #######
    if count <= 3:
        from obstacle_information.Env1 import obs_x, obs_y
        ax.scatter(obs_x, obs_y, marker='o', label='chain', color='#ff7f0e', s=15, zorder=1)
    elif count >= 4 and count <= 13:
        from obstacle_information.Env2 import obs_x, obs_y
        ax.scatter(obs_x, obs_y, marker='o', label='chain', color='#ff7f0e', s=15, zorder=1)
    elif count >= 14 and count <= 15:
        from obstacle_information.Env3 import obs_x, obs_y
        ax.scatter(obs_x, obs_y, marker='o', label='chain', color='#ff7f0e', s=15, zorder=1)
    elif count >= 16:
        from obstacle_information.Env4 import obs_x, obs_y
        ax.scatter(obs_x, obs_y, marker='o', label='chain', color='#ff7f0e', s=15, zorder=1)

    # if count >= 0 and count <= 25:
    #    from obstacle_information.Env3 import obs_x, obs_y
    #    ax.scatter(obs_x, obs_y, marker='o', label='chain', color='#ff7f0e', s=15, zorder=1)

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

    #ax.plot(train_x, train_y, label='input flight path',
    #        color='#2ca02c', linestyle='--', zorder=2, linewidth=2)
    ax.plot(train_x_pre, train_y_pre, label='measured flight path',
            color='#1f77b4', zorder=2, linewidth=3,)
    ax.plot(test_x, test_y, label='predicted flight path',
             color='#1f77b4', zorder=2, linewidth=3)
    # ax.plot(test_x, test_y, label='predicted flight path',
    #         color='#d62728', zorder=2, linewidth=3)

    # train_x_pre, train_y_preから進行方向の角度を算出してプロット
    for i in range(len(train_x_pre) - 1):
        dx = train_x_pre[i+1] - train_x_pre[i]
        dy = train_y_pre[i+1] - train_y_pre[i]
        # 進行方向の角度をラジアンで計算
        travel_angle_rad = np.arctan2(dy, dx)
        # 矢印の描画
        arrow_dx = np.cos(travel_angle_rad) * 0.2
        arrow_dy = np.sin(travel_angle_rad) * 0.2
        #ax.arrow(train_x_pre[i], train_y_pre[i], arrow_dx, arrow_dy, head_width=0.1, head_length=0.1, fc='g', ec='g', zorder=3, alpha=0.5)

    # test_x, test_yから進行方向の角度を算出してプロット
    for i in range(len(test_x) - 1):
        dx = test_x[i+1] - test_x[i]
        dy = test_y[i+1] - test_y[i]
        # 進行方向の角度をラジアンで計算
        travel_angle_rad = np.arctan2(dy, dx)
        travel_angle_deg_predicted = np.rad2deg(travel_angle_rad) # 度数法に変換
        #print(f"Step {i}: Predicted Angle(deg)={travel_angle_deg_predicted:.2f}") # デバッグ用

        # パルス放射がある場合、進行方向とパルス放射方向の角度差を計算
        if train_pulse[i] >= 0.5:
            # 実測値：進行方向と放射方向の差は、Pxyの絶対値そのもの
            angle_deg_measured = data[1][0][predict_time + i][0][episode][7] # 実測Pxy
            diff_measured = abs(angle_deg_measured)
            travel_vs_pulse_measured_diffs.append(diff_measured)

            # 予測値：進行方向と放射方向の差は、予測Pxyの絶対値そのもの
            angle_deg_predicted = data[0][0][predict_time + i][0][episode][9] # 予測Pxy
            diff_predicted = abs(angle_deg_predicted)
            travel_vs_pulse_predicted_diffs.append(diff_predicted)

            # 実測Pxyと予測Pxyの誤差
            angle_diff_val = abs(angle_deg_measured - angle_deg_predicted)
            if angle_diff_val > 180:
                angle_diff_val = 360 - angle_diff_val
            
            env_val = data[1][0][predict_time + i][0][episode][8]
            bat_val = data[1][0][predict_time + i][0][episode][9]

            results_for_csv.append({'count': count, 
                                    'diff_measured': diff_measured, 
                                    'diff_predicted': diff_predicted, 
                                    'angle_diff': angle_diff_val,
                                    'ENV': env_val, 'BAT': bat_val})

    for i in range(len(train_x_pre)):
        # ループの範囲外アクセスを防ぐ
        if i < len(train_x_pre) - 1 and train_pulse[i] >= 0.5:
            # --- 実測値の放射方向 ---
            # 進行方向を計算
            dx_measured = train_x_pre[i+1] - train_x_pre[i]
            dy_measured = train_y_pre[i+1] - train_y_pre[i]
            travel_rad_measured = np.arctan2(dy_measured, dx_measured)
            # 進行方向からのずれ（実測Pxy）
            pulse_offset_rad_measured = np.deg2rad(data[1][0][predict_time + i][0][episode][7])
            # 最終的な放射方向
            final_rad_measured = travel_rad_measured + pulse_offset_rad_measured
            # 矢印を描画
            dx_arrow_measured = np.cos(final_rad_measured) * 0.2
            dy_arrow_measured = np.sin(final_rad_measured) * 0.2
            ax.arrow(train_x_pre[i], train_y_pre[i], dx_arrow_measured, dy_arrow_measured, head_width=0.1, head_length=0.1, fc='b', ec='b', zorder=3)

            # --- 予測値の放射方向 ---
            # 進行方向を計算
            dx_predicted = test_x[i+1] - test_x[i]
            dy_predicted = test_y[i+1] - test_y[i]
            travel_rad_predicted = np.arctan2(dy_predicted, dx_predicted)
            # 進行方向からのずれ（予測Pxy）
            pulse_offset_rad_predicted = np.deg2rad(data[0][0][predict_time + i][0][episode][9])
            # 最終的な放射方向
            final_rad_predicted = travel_rad_predicted + pulse_offset_rad_predicted
            # 矢印を描画
            dx_arrow_predicted = np.cos(final_rad_predicted) * 0.2
            dy_arrow_predicted = np.sin(final_rad_predicted) * 0.2
            ax.arrow(test_x[i], test_y[i], dx_arrow_predicted, dy_arrow_predicted, head_width=0.1, head_length=0.1, fc='r', ec='r', zorder=3)

            # --- 角度の差を計算 ---
            # Pxyは進行方向からの相対角度なので、そのまま差を計算する
            angle_deg = data[1][0][predict_time + i][0][episode][7]  # 実測Pxy
            angle_test = data[0][0][predict_time + i][0][episode][9] # 予測Pxy
            angle_diff = abs(angle_deg - angle_test)
            if angle_diff > 180:
                angle_diff = 360 - angle_diff
            angle_diffs.append(angle_diff)

    for i in range(len(train_pulse)):
        if train_pulse[i] >= 0.5:
            ax.scatter(train_x_pre[i], train_y_pre[i], label='measured pulse timing', color='k', s=20, zorder=4)
        if test_pulse[i] >= 0.5:
            ax.scatter(test_x[i], test_y[i], label='predicted pulse timing', color='w',edgecolor = '#d62728' ,s=20, zorder=4)
    #print(test_pulse[i])

    if angle_diffs:
        mean_diff = np.mean(angle_diffs)
        std_diff = np.std(angle_diffs)
        all_angle_diffs.extend(angle_diffs)
        print(f"count: {count}, 実測 vs 予測 -> Mean Abs Diff: {mean_diff:.2f}, Std Dev: {std_diff:.2f}")
    
    if travel_vs_pulse_measured_diffs:
        mean_measured_diff = np.mean(travel_vs_pulse_measured_diffs)
        std_measured_diff = np.std(travel_vs_pulse_measured_diffs)
        print(f"count: {count}, 進行方向 vs 実測放射方向 -> Mean: {mean_measured_diff:.2f}, Std: {std_measured_diff:.2f}")

    if travel_vs_pulse_predicted_diffs:
        mean_predicted_diff = np.mean(travel_vs_pulse_predicted_diffs)
        std_predicted_diff = np.std(travel_vs_pulse_predicted_diffs)
        print(f"count: {count}, 進行方向 vs 予測放射方向 -> Mean: {mean_predicted_diff:.2f}, Std: {std_predicted_diff:.2f}")
    print("================================================================================")

    plt.xlabel('X [m]')
    plt.ylabel('Y [m]')

    plt.xticks([0, 1.0, 2.0, 3.0, 4.0, 4.5])
    plt.yticks([0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 7.5])
    plt.tight_layout()
    ax.set_aspect('equal')
    # plt.legend(bbox_to_anchor=(1,1), loc='upper left')

    pp.savefig(fig)

pp.close()

# CSVファイルに結果を保存
df_results = pd.DataFrame(results_for_csv)
csv_path = os.path.join(path, 'angle_diffs.csv')
df_results.to_csv(csv_path, index=False)
print(f"\n角度差の結果を {csv_path} に保存しました。")

if all_angle_diffs:
    total_mean_diff = np.mean(all_angle_diffs)
    total_std_diff = np.std(all_angle_diffs)
    print("\n--- Overall Statistics ---")
    print(f"Total Mean Abs Diff: {total_mean_diff:.2f}")
    print(f"Total Std Dev: {total_std_diff:.2f}")

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