obs_x = [0.9,
         2.0,
         0.6,
         0.65,
         2.0,
         1.5,
         2.7,
         3.15,
         2.75,
         3.9,
         2.65,
         2.5
         ]

obs_y = [3.7,
         3.9,
         2.35,
         0.9,
         0.9,
         1.9,
         0.55,
         1.5,
         1.9,
         3.4,
         3.6,
         3.5
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