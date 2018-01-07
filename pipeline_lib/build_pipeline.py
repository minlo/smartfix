from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.preprocessing import FunctionTransformer
import logging
from tempfile import mkdtemp
from shutil import rmtree
from sklearn.externals.joblib import Memory


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
        # for caching pipeline objects
        # It is sometimes worthwhile storing the state of a specific transformer since it could be used again.
        # Using a pipeline in GridSearchCV triggers such situations. Therefore, we use the argument memory
        # to enable caching.
        # See details here:
        # http://scikit-learn.org/stable/auto_examples/plot_compare_reduction.html#caching-transformers-within-a-pipeline
        self.cache_dir = mkdtemp()
        self.memory = Memory(cachedir=self.cache_dir, verbose=10)

    def build(self):
        """Finally, we still failed here. This method will be deprecated in later versions."""
        pipeline = Pipeline(
            [
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
            ],
            memory=self.memory
        )
        return pipeline

    def destroy_cache_memory(self):
        """
        Destroy the cached memory after running the grid search.
        """
        rmtree(self.cache_dir)

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

