obs_x = [2.079539,
         2.050287,
         2.05266,
         2.048295,
         2.033162,
         2.014463,
         1.980089
         ]

obs_y = [0.501269,
         0.99784,
         1.455863,
         1.976081,
         2.476076,
         2.992291,
         3.472948
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
