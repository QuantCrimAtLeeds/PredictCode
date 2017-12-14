from common import *

import datetime, itertools
import open_cp.prohotspot

with scripted.Data(load_points, load_geometry,
        start=datetime.datetime(2016,1,1)) as state:
    
    time_range = scripted.TimeRange(datetime.datetime(2016,10,1),
            datetime.datetime(2017,1,1), datetime.timedelta(days=1))
    for tb, sb in itertools.product([6,7,8,9], [2,3,4]):
        weight = open_cp.prohotspot.ClassicWeight(time_bandwidth=tb, space_bandwidth=sb)
        distance = open_cp.prohotspot.DistanceCircle()
        state.add_prediction(scripted.ProHotspotProvider(weight, distance), time_range)

    state.score(scripted.HitCountEvaluator)
    state.process(scripted.HitCountSave("pro_grid_opt.csv"))
