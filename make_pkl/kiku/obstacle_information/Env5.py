obs_x = [
	   2.347244,
	   2.36658,
	   2.319276,
	   2.310381,
	   2.370136,
	   2.332053,
	   2.327425
         ]

obs_y = [
	   0.819634,
	   1.814846,
	   2.823416,
	   3.813048,
	   1.308726,
	   2.331246,
	   3.336422
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
