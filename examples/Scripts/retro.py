from common import *

import datetime
import open_cp.retrohotspot

with scripted.Data(load_points, load_geometry,
        start=datetime.datetime(2016,1,1)) as state:
    
    time_range = scripted.TimeRange(datetime.datetime(2016,10,1),
            datetime.datetime(2017,1,1), datetime.timedelta(days=1))
    weight = open_cp.retrohotspot.Quartic()
    state.add_prediction(scripted.RetroHotspotProvider(weight), time_range)
    state.add_prediction(scripted.RetroHotspotCtsProvider(weight), time_range)

    state.score(scripted.HitCountEvaluator)

    state.save_predictions("retro_preds.pic.xz")

    state.process(scripted.HitCountSave("retro.csv"))
