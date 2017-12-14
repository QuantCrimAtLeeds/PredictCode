from common import *

import datetime, itertools
import open_cp.kde

with scripted.Data(load_points, load_geometry,
        start=datetime.datetime(2016,1,1)) as state:
    
    time_range = scripted.TimeRange(datetime.datetime(2016,10,1),
            datetime.datetime(2017,1,1), datetime.timedelta(days=1))

    for bw in [30, 40, 50, 60, 70, 100]:
        tk = open_cp.kde.ExponentialTimeKernel(10)
        sk = open_cp.kde.GaussianFixedBandwidthProvider(bw)
        state.add_prediction(scripted.KDEProvider(tk, sk), time_range)

    state.score(scripted.HitCountEvaluator)
    state.process(scripted.HitCountSave("kde_opt.csv"))
