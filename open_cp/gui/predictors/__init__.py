from . import predictor

# Add new prediction modules here ###########################################

from . import naive
from . import grid
from . import lonlat

# End #######################################################################

import sys as _sys
_fp = predictor.FindPredictors(_sys.modules[__name__])
all_predictors = list(_fp.predictors)