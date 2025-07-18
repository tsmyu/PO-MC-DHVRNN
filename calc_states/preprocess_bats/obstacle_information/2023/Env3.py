obs_x = [0.934525,
         1.335974,
         3.254925,
         2.900872,
         2.505039,
         2.142174,
         1.732515
         ]

obs_y = [3.31334,
         2.792345,
         0.585745,
         1.019103,
         1.47251,
         1.921288,
         2.376273
         ]


obs_frame_x = []
obs_frame_y = []

for x, y in zip(obs_x, obs_y):
    x0 = x - 0.015
    y0 = y - 0.015
    x1 = x + 0.015
    y1 = y - 0.015
    x2 = x + 0.015
    y2 = y + 0.015
    x3 = x - 0.015
    y3 = y + 0.015
    frame_x = [x0, x1, x2, x3, x0]
    frame_y = [y0, y1, y2, y3, y0]
    obs_frame_x.append(frame_x)
    obs_frame_y.append(frame_y)
