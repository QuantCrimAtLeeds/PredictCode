
# Add new prediction modules here ###########################################

from . import predictor
from . import naive

# End #######################################################################

import sys as _sys
_fp = predictor.FindPredictors(_sys.modules[__name__])
all_predictors = list(_fp.predictors)