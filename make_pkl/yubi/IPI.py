import matplotlib.pyplot as plt
import numpy as np
import os
import glob
import pandas as pd
from natsort import natsorted
from matplotlib.backends.backend_pdf import PdfPages
# from obstacle_information.Random_20230510 import obs_x, obs_y
import statistics

# pp = PdfPages('./toPKL/TR-allRandom/2d/with pulse/8s/result/flag/topview_pulse.pdf')

files_te = natsorted(glob.glob('./dataset/test/combine/flag/*.csv'))
files_tr = natsorted(glob.glob('./dataset/train/combine/flag/*.csv'))
files = files_te+files_tr

# start = 1500
# finish = 1980
# start = [1500, 1500, 1500, 2200]
# finish = [1980, 1850, 1900, 2500]
# start-finish = 400 /// MAX3200
# 601:1500-1980
# 603:1500-1850
# 605:1500-1900
# 2690:2200-2500

all_ipi = []
distance = []
count = 0

for file in files:
    df = pd.read_csv(file)
    fname = os.path.split(file)[1].split('.csv')[0]

    pulse_time = []
    ipi = []
    # x_l = []
    # y_l = []

    # x = df['x']
    # y = df['z']
    pulse = df['pulse']

    # fig = plt.figure()
    # ax = fig.add_subplot(111)

    # ax.plot(x[start:finish], y[start:finish], color='#1f77b4', zorder=1)

    for i in range(len(pulse)):
        if pulse[i] == 1:
            # ax.scatter(x[i], y[i], color='#e41a1c', s=18, zorder=2, label='pulse timing')
            pulse_time.append(df['time'][i])
            # x_l.append(x[i])
            # y_l.append(y[i])
            count += 1

    for i in range(len(pulse_time) - 1):
        ipi.append((pulse_time[i+1] - pulse_time[i]) * 1000)
        all_ipi.append((pulse_time[i+1] - pulse_time[i]) * 1000)
        # distance.append(np.sqrt((x_l[i+1] - x_l[i])**2 + (y_l[i+1] - y_l[i])**2))


# IPI = [i for i in all_ipi if i < 500]

print(count)
print('ipi max {}'.format(max(all_ipi)))
print('ipi min {}'.format(min(all_ipi)))
print('ipi mean {}'.format(statistics.mean(all_ipi)))
print('ipi se {}'.format(statistics.pstdev(all_ipi)))
# print('ipi max {}'.format(max(IPI)))
# print('ipi min {}'.format(min(IPI)))
# print('ipi mean {}'.format(statistics.mean(IPI)))
# print('ipi se {}'.format(statistics.pstdev(IPI)))
# print('distance mean {}'.format(statistics.mean(distance)))
# print('distance se {}'.format(statistics.pstdev(distance)))

plt.hist(all_ipi, bins=80, edgecolor="black")
plt.title("IPI histogram")
plt.xlabel("IPI [ms]")
plt.ylabel("frequnecy")
plt.savefig("./teshima/result/IPI_histgram.png", format="png")

    # ax.scatter(obs_x, obs_y, marker='o', label='chain', color='#ff7f0e', s=8)

    # plt.xlabel('X [m]')
    # plt.ylabel('Y [m]')

    # plt.xticks([0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5])
    # plt.yticks([0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5])

    # ax.set_aspect('equal')
    # plt.tight_layout()
    # plt.legend(bbox_to_anchor=(1,1), loc='upper left')
    # plt.savefig('./for plot pulsetiming/fig/{}.png'.format(fname))
    # plt.show()
#     pp.savefig(fig)

# pp.close()