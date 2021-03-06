from sklearn.base import BaseEstimator, TransformerMixin
# import pandas as pd
import statsmodels.api as sm
import logging
import operator
from scipy.stats import norm
import numpy as np
import time
from sklearn.feature_selection import SelectKBest, SelectFromModel
from sklearn.linear_model import Lasso

# setting logging configuration
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class HardThresholdSelector(BaseEstimator, TransformerMixin):
    """Select features according to their t-statistics in single regression versus response variables.

    Using this class, we are able to select top-K features which have the biggest t-statistic values.

    We plan to provide two ways:
    1. top-K features
    2. features with t-statistic value bigger than specified confidence level 1 - alpha, where alpha would be a
        parameter specified by the user.
    3. As for magic features, we simply do not regress on them, and we will add them later to selected features.
    4. If k is -1, we select all features.
    """
    def __init__(self, target_column="y", k=10, alpha=0.05, date_column="date", select_top_k=True, print_top_k=False):
        self.target_column = target_column
        self.k = k
        self.alpha = alpha
        self.date_column = date_column
        self.select_top_k = select_top_k
        self.print_top_k = print_top_k

        self.significant_value = norm.ppf(1 - self.alpha / 2)

        self.selected_features = None

    @staticmethod
    def regression_t_statistic(data, target_column, ctrl_columns, feature_column):
        """
        Given ctrl_columns, we run regression of target_column on feature_column.
        After regression, we return the t-statistic.
        """
        column_x_list = ctrl_columns.copy()
        column_x_list.append(feature_column)
        # data.replace([np.inf, -np.inf], np.nan, inplace=True)
        # logger.info("Before dropping na, data: {}".format(data.shape))
        # data = data.dropna()
        # logger.info("After dropping na, data: {}".format(data.shape))
        if data.empty:
            raise ValueError("data is empty!")

        x = data.as_matrix([column_x_list])
        y = data.as_matrix([target_column])
        y = np.ravel(y)
        # logger.info("target_column: {}, column_x_list: {}".format(target_column, column_x_list))
        # logger.info("X: {}, y: {}".format(X.shape, y.shape))
        # logger.info("X: {}, y: {}".format(X[0], y))

        # run regression
        x2 = sm.add_constant(x)
        est = sm.OLS(y, x2)
        est2 = est.fit()
        # logger.info("est2 summary: {}".format(est2.summary()))
        return est2.summary().tables[1].data[-1][3]

    def generate_ctrl_columns(self, data):
        hard_thres_test_columns = []
        hard_thres_ctrl_columns = []
        for column_i in list(data.columns):
            if column_i in [self.target_column, self.date_column] or "forward" in column_i or "magic" in column_i:
                continue
            if self.target_column + "_lag_" in column_i:
                hard_thres_ctrl_columns.append(column_i)
            else:
                hard_thres_test_columns.append(column_i)
        logger.info("There are in total {} columns, while we only need {} columns".format(
            len(hard_thres_test_columns),
            len(hard_thres_ctrl_columns)
        ))
        return hard_thres_ctrl_columns, hard_thres_test_columns

    def generate_t_statistic(self, data):
        hard_thres_test_t_stats = {}
        ctrl_columns, test_columns = self.generate_ctrl_columns(data)

        for column_i in test_columns:
            # time_start = time.time()
            # logger.info("regression on {}".format(column_i))
            t_i = self.regression_t_statistic(data.copy(), self.target_column, ctrl_columns, column_i)
            hard_thres_test_t_stats[column_i] = abs(float(t_i))
            # logger.info("for column_i: {}, it takes {:.2f} seconds to run regression for it!".format(
            # column_i, time.time() - time_start))

        sorted_hard_thres_test_t_stats = sorted(
            hard_thres_test_t_stats.items(),
            key=operator.itemgetter(1),
            reverse=True
        )

        return sorted_hard_thres_test_t_stats

    def select_top_k_hard(self, data):
        init_time = time.time()
        selected_features = []
        for column_i in list(data.columns):
            if "magic" in column_i and self.k > 0:
                selected_features.append(column_i)
            elif self.k < 0:
                selected_features.append(column_i)
        if self.k > 0:
            sorted_hard_thres_t_stats = self.generate_t_statistic(data.copy())
            logger.info("\nIn total, it takes {:.2f} seconds to run regression for {} columns".format(
                time.time() - init_time,
                len(sorted_hard_thres_t_stats)
            ))
            index = 0
            for item_i in sorted_hard_thres_t_stats:
                if self.select_top_k and index < self.k:
                    selected_features.append(item_i[0])
                elif not self.select_top_k and item_i[1] >= self.significant_value:
                    selected_features.append(item_i[1])
                index += 1
        if self.print_top_k:
            logger.info("We select {} features by hard thresholding: \n".format(len(selected_features)))
            logger.info("{}".format("\n".join(selected_features)))

        # selected_features.extend([self.date_column, self.target_column])
        # We may deal with X features only, and leave y as parameter.
        # selected_features.extend([self.target_column, "forward_y"])
        return selected_features

    def partial_fit(self, X, y=None):
        """Online computation of min and max on X for later selecting."""
        self.selected_features = self.select_top_k_hard(X.copy())
        return self

    def fit(self, X, y=None):
        return self.partial_fit(X, y)

    def transform(self, X, y=None):
        if self.selected_features is None:
            raise ValueError("Watch out, self.selected_features is None, you must fit before transforming!!!")
        unseen_columns = [column_i for column_i in self.selected_features if column_i not in list(X.columns)]
        if len(unseen_columns) > 0:
            logger.info("print out columns which may cause key error: ")
            for column_i in unseen_columns:
                logger.info(column_i + "\n")
            logger.info("The end of this print.")
            self.partial_fit(X, y)
        data_hard = X[self.selected_features]
        data_hard.reset_index(drop=True, inplace=True)
        return data_hard.as_matrix()


class FeatureSelector(BaseEstimator, TransformerMixin):
    def __init__(self, select_method="hard", target_column="y", k=200, alpha=0.05,
                 date_column="date", select_top_k=True, print_top_k=False, is_training=True):
        self.select_method = select_method
        self.target_column = target_column
        self.k = k
        self.alpha = alpha
        self.date_column = date_column
        self.select_top_k = select_top_k
        self.print_top_k = print_top_k
        self.is_training = is_training

        self.check_select_method()
        self.selector = self._choose_selector()

    def check_select_method(self):
        if self.select_method not in ["hard", "soft"]:
            raise ValueError("Select method must be one of ['hard', 'soft']")

    def _preprocess_data(self, data):
        # logger.info("select_method: {}, data_type: {}".format(self.select_method, type(data)))
        if self.select_method == "soft":
            data.reset_index(drop=True, inplace=True)
            # del data[self.target_column], data[self.date_column]
            data_values = data.as_matrix()
            return data_values
        else:
            # print("before converting to numpy", data.shape, data.dropna().shape)
            return data

    def _choose_selector(self):
        if self.select_method == "hard":
            selector = HardThresholdSelector(
                target_column=self.target_column,
                k=self.k,
                alpha=self.alpha,
                date_column=self.date_column,
                select_top_k=self.select_top_k,
                print_top_k=self.print_top_k
            )
        else:
            selector = SelectFromModel(Lasso(alpha=0.001), prefit=False)
        return selector

    def fit(self, X, y=None):
        # print("!!!Before soft thresholding fitting, X shape: {}".format(X.shape))
        X = self._preprocess_data(X)
        self.selector.fit(X, y)
        # print("!!!After soft thresholding fitting, X shape: {}".format(X.shape))
        return self

    def transform(self, X, y=None):
        X = self._preprocess_data(X)
        # print("!!!Before soft thresholding transforming, X shape: {}".format(X.shape))
        X = self.selector.transform(X)
        # print("!!!After soft thresholding transforming, X shape: {}".format(X.shape))

        return X


