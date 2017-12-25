from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer
import logging


# setting logging configuration
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class BuildPipeline(object):
    """
    Try to merge all steps into a complete pipeline.
    """
    def __init__(self, imputer, engineer, selector, scaler, reducer, model, split_date):
        self.imputer = imputer
        self.engineer = engineer
        self.selector = selector
        self.scaler = scaler
        self.reducer = reducer
        self.model = model
        self.split_date = split_date

    def split_train_test(self, X, y):
        """This function is to split the X by split_date, so that we can get
        X and y for train and test respectively."""
        return X[X.index <= self.split_date], y[y.index <= self.split_date]

    def build(self):
        """Finally, we still failed here. This method will be deprecated in later versions."""
        pipeline = Pipeline([
            ("imputer", self.imputer),
            ("engineer", self.engineer),
            ("splitter", FunctionTransformer(self.split_train_test)),
            ("selector", self.selector),
            ("scaler", self.scaler),
            ("reducer", self.reducer),
            ("model", self.model)
        ])
        return pipeline

    def build_before_selector(self):
        pipeline = Pipeline([
            ("imputer", self.imputer),
            ("engineer", self.engineer),
            # ("selector", self.selector),
        ])
        return pipeline

    def build_after_selector(self):
        pipeline = Pipeline([
            # ("selector", self.selector),
            ("scaler", self.scaler),
            # ("reducer", self.reducer),  # temporarily not used
            ("model", self.model)
        ])
        return pipeline

