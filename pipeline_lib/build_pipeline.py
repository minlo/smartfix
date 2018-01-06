from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.preprocessing import FunctionTransformer
import logging


# setting logging configuration
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class BuildPipeline(object):
    """
    Try to merge all steps into a complete pipeline.
    """
    def __init__(self, imputer, engineer, selector, scaler, reducer, model):
        self.imputer = imputer
        self.engineer = engineer
        self.selector = selector
        self.scaler = scaler
        self.reducer = reducer
        self.model = model

        # self.magic_feature_extractor = magic_feature_extractor

    def build(self):
        """Finally, we still failed here. This method will be deprecated in later versions."""
        pipeline = Pipeline([
            ("imputer", self.imputer),
            # ("magic_features_union", FeatureUnion([
            #     ("ordinary_features", Pipeline([
            #
            #     ])),
            #     ("magic_features", Pipeline([
            #         ("magic_features_extractor", self.magic_feature_extractor)
            #     ]))
            # ])),
            ("engineer", self.engineer),
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
            ("selector", self.selector),
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

