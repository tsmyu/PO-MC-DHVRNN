obs_x = [
	   3.337976,
	   1.955404,
	   2.298488,
	   2.99858,
	   1.589509,
	   2.596363,
	   1.262466
         ]

obs_y = [
	   1.277734,
	   2.693686,
	   2.334811,
	   1.621028,
	   3.043048,
	   1.993195,
	   3.435733
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
