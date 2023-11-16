obs_x = [1.6,
         0.9,
         1.95,
         1.0,
         1.0,
         2.2,
         2.6,
         3.3,
         3.65,
         3.5,
         2.9,
         2.65
         ]

obs_y = [3.95,
         3.8,
         3.05,
         1.35,
         1.05,
         1.55,
         0.9,
         0.7,
         1.35,
         2.5,
         2.9,
         3.6
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