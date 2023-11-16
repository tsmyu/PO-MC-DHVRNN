obs_x = [0.615833,
         1.492721,
         1.910963,
         2.47458,
         3.646028,
         3.631695,
         3.458596,
         2.479156,
         1.674436,
         0.712584,
         1.205884
         ]

obs_y = [0.855146,
         0.422255,
         1.420122,
         1.170984,
         1.137113,
         2.950053,
         3.55672,
         3.319134,
         2.456405,
         2.101615,
         3.446665
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
