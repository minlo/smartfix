import logging
import warnings
from sklearn.base import BaseEstimator, TransformerMixin
import pandas as pd


# setting logging configuration
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ImputeNumericDataFrame(BaseEstimator, TransformerMixin):
    """Design our own imputation strategy.

    This class takes a pandas data frame as input and return pandas data frame.

    Parameters
    ==========
    X: a pandas data frame

    column_list: list
        A list of desired features which are required to be part of the data frame columns.
        Otherwise KeyError exception would be raised.

    convert_type: str, 'numeric' by default
        If 'numeric', pd.to_numeric() would be done in transform() method.
        If 'time', pd.to_datetime() would be done in transform() method.
        If 'str', astype('str') would be done in transform() method.

    numeric_impute_method: str, 'median' by default
        If 'median', median value of a continuous column would be used to impute the missing values.
        If 'mean', mean value of a continuous column would be used to impute the mussing values.

    immune_columns: list, empty by default
        This list indicates the columns which are not to be operated in this transform process.
    """
    def __init__(self, column_list, convert_type, numeric_impute_method, immune_columns):
        self.column_list = column_list
        self.convert_type = convert_type
        self.numeric_impute_method = numeric_impute_method
        self.immune_columns = immune_columns

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        if not self.column_list:
            warnings.warn('Provided column_list is empty, return empty pandas data frame!')
            return pd.DataFrame()

        data_columns = set(list(X.columns))
        column_list_set = set(self.column_list)
        if not column_list_set.issubset(data_columns):
            raise KeyError('Key {} not present in the data frame!'.format(list(column_list_set - data_columns)[0]))

        logger.info("Starting to convert columns into desired '{}' format...".format(self.convert_type))
        X_copy = X.copy()
        for column_i in column_list_set:
            if column_i in self.immune_columns:
                continue
            logger.debug(column_i)
            if self.convert_type == 'numeric':
                X_copy[column_i] = pd.to_numeric(X_copy[column_i], errors="coerce")
                if self.numeric_impute_method == 'median':
                    X_copy[column_i].fillna(X_copy[column_i].median(), inplace=True)
                elif self.numeric_impute_method == 'mean':
                    X_copy[column_i].fillna(X_copy[column_i].mean(), inplace=True)
                else:
                    pass  # here, we can supply our own design impute method
            elif self.convert_type == 'time':
                X_copy[column_i] = pd.to_datetime(X_copy[column_i], errors="coerce")
            else:
                try:
                    X_copy[column_i] = X_copy[column_i].astype('str')
                except Exception:
                    logger.error('Failed to convert {} to as type str.'.format(column_i), exc_info=True)
                    raise RuntimeError("Unable to convert '{}' when converting it to str.".format(column_i))

        return X_copy[list(column_list_set)]

