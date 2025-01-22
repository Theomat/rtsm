from rtsm.predictors.predictor import Predictor
from rtsm.predictors.no_predictor import NoPredictor
from rtsm.predictors.logistic_boolean_predictor import LogisticBooleanPredictor
from rtsm.predictors.linear_predictor import LinearRegressionPredictor
import sklearn

sklearn.set_config(
    assume_finite=True,  # disable validation
    skip_parameter_validation=True,
)
