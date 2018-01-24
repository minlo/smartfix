from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV, TimeSeriesSplit, train_test_split
import pandas as pd
import datetime
import logging


# setting logging configuration
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class TrainTestForTimeSeries:
    """
    Provide a general purpose train and test split method.
    """
    def __init__(self, data, test_year, feature_engineering_pipeline, date_column='date', split_date=None):
        self.data = data
        self.test_year = test_year
        self.feature_engineering_pipeline = feature_engineering_pipeline
        self.date_column = date_column

        self.split_date = None
        if split_date is not None:
            self.split_date = split_date

    @staticmethod
    def train_test_split_year(data, split_date, date_column='date'):
        try:
            data[date_column] = pd.to_datetime(data[date_column])
            data[date_column] = data[date_column].dt.date
        except Exception:
            print(data[date_column][0])
            pass

        data_train = data[data[date_column] < split_date]
        data_test = data[(data[date_column] >= split_date) &
                         (data[date_column] < datetime.date(split_date.year + 1, 1, 1))]

        print('data: {}, data_train: {}, data_test: {}. '.format(data.shape, data_train.shape, data_test.shape))
        return data_train, data_test

    def train_split_pipeline_diff_features(self):
        data_imputed = self.feature_engineering_pipeline.fit_transform(self.data)
        return self.train_test_split_year(data_imputed, self.split_date, self.date_column)

    def train_split_pipeline_diff_features_by_year(self):
        data_imputed = self.feature_engineering_pipeline.fit_transform(self.data)
        return self.train_test_split_year(data_imputed, datetime.date(self.test_year, 1, 1), self.date_column)


# various pipeline modes
class GeneratePipeline:
    """
    Provide a prototype to write your own pipeline.

    pipeline_mode: str, "single" by default
        If "single", return pipeline for a single model.
        If "random_search", return pipeline for RandomizedSearch.
        If "grid_search", return pipeline for GridSearch.
    """
    def __init__(self, scaler, selector, model, pipeline_mode, param_grid, n_iter, verbose, scoring_criterion="roc_auc"):
        self.scaler = scaler
        self.selector = selector
        self.model = model
        self.pipeline_mode = pipeline_mode
        self.param_grid = param_grid
        self.n_iter = n_iter
        self.verbose = verbose
        self.scoring_criterion = scoring_criterion

    def get_pipeline(self):
        model_pipeline = Pipeline([
                ('feature_scaling', self.scaler),
                ('feature_selecting', self.selector),
                ('modelling', self.model)
            ])
        if self.pipeline_mode == "single":
            return model_pipeline
        elif self.pipeline_mode == "grid_search":
            return GridSearchCV(model_pipeline, cv=TimeSeriesSplit(), n_jobs=-1, param_grid=self.param_grid, verbose=5, scoring=self.scoring_criterion)
        elif self.pipeline_mode == "random_search":
            return RandomizedSearchCV(model_pipeline,
                                      cv=TimeSeriesSplit(),
                                      n_jobs=-1,
                                      param_distributions=self.param_grid,
                                      n_iter=self.n_iter,
                                      verbose=self.verbose)

