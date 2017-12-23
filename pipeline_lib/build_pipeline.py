from sklearn.pipeline import Pipeline, FeatureUnion
import logging


# setting logging configuration
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class BuildPipeline(object):
    """
    Try to merge all steps into a complete pipeline.
    """
    def __init__(self, imputer, scaler, engineer, selector, model):
        self.imputer = imputer
        self.scaler = scaler
        self.engineer = engineer
        self.selector = selector
        self.model = model

    def build(self):
        pipeline = Pipeline([
            ("imputer", self.imputer),
            ("scaler", self.scaler),
            ("engineer", self.engineer),
            ("selector", self.selector),
            ("model", self.model)
        ])
        return pipeline

