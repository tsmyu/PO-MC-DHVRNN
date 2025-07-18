obs_x = [1.6,
         0.9,
         1.05,
         1.05,
         1.8,
         3.4,
         2.7,
         3.7,
         3.4,
         2.1,
         3.0,
         3.5
         ]

obs_y = [3.9,
         2.9,
         1.35,
         1.0,
         1.85,
         3.55,
         3.25,
         2.6,
         2.35,
         2.2,
         1.0,
         2.0
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