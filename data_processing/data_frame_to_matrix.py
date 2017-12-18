import logging
import warnings
from sklearn.base import BaseEstimator, TransformerMixin
import pandas as pd


# setting logging configuration
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DataFrameToMatrix(BaseEstimator, TransformerMixin):
    """Convert pandas data frame into numpy ndarray by calling .as_matrix() method.

    Parameters
    ----------
    X: pandas data frame

    immune_column_list: list
        All columns in the data frame except immune_column_list would be converted into matrix form.

    """
    def __init__(self, immune_column_list):
        self.immune_column_list = immune_column_list

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        column_list_set = [column_i for column_i in list(X.columns) if column_i not in self.immune_column_list]
        if not list(X.columns):
            warnings.warn('Provided column list is empty, return empty array!')
            return pd.DataFrame().as_matrix()

        logger.info('Convert data frame to matrix...')
        return X.as_matrix(list(column_list_set))

