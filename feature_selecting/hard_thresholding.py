from sklearn.base import BaseEstimator, TransformerMixin
import pandas as pd
import statsmodels.api as sm
import logging
import operator


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
    """
    def __init__(self, target_column, k, alpha, date_column="date", select_top_k=True):
        self.target_column = target_column
        self.k = k
        self.alpha = alpha
        self.date_column = date_column
        self.select_top_k = select_top_k

        self.significant_value = 1.96

    @staticmethod
    def regression_t_statistic(data, target_column, ctrl_columns, feature_column):
        """
        Given ctrl_columns, we run regression of target_column on feature_column.
        After regression, we return the t-statistic.
        """
        column_x_list = ctrl_columns.copy()
        column_x_list.append(feature_column)

        X = data[column_x_list]
        y = data[target_column]

        # run regression
        X2 = sm.add_constant(X)
        est = sm.OLS(y, X2)
        est2 = est.fit()
        return est2.summary().tables[1].data[-1][3]

    def generate_ctrl_columns(self, data):
        hard_thres_test_columns = []
        hard_thres_ctrl_columns = []
        for column_i in list(data.columns):
            if column_i in [self.target_column, self.date_column] or "forward" in column_i:
                continue
            if self.target_column + "_back_0_order_diff" in column_i:
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
            t_i = self.regression_t_statistic(data, self.target_column, ctrl_columns, column_i)
            hard_thres_test_t_stats[column_i] = abs(float(t_i))

        sorted_hard_thres_test_t_stats = sorted(hard_thres_test_t_stats.items(), key=operator.itemgetter(1), reverse=True)

        return sorted_hard_thres_test_t_stats

    def select_top_k_hard(self, data):
        selected_features = []
        sorted_hard_thres_t_stats = self.generate_t_statistic(data)
        index = 0
        for item_i in sorted_hard_thres_t_stats:
            if self.select_top_k and index < self.k:
                selected_features.append(item_i[0])
            elif not self.select_top_k and item_i[1] >= self.significant_value :
                selected_features.append(item_i[1])
            index += 1

        return selected_features

    @staticmethod
    def generate_diff_features(data,
                               target_column,
                               look_backward_days,
                               diff_order):
        """Generate the specified diff_order features.

        :param data:
        :param target_column:
        :param look_backward_days:
        :param diff_order:
        :param date_column:
        :return:
        """
        data[target_column + '_back_0_order_diff_0'] = data[target_column]
        if diff_order == 0:
            shift_column_specifier = target_column
        else:
            shift_column_specifier = target_column + '_back_' + str(diff_order - 1) + '_order_diff_'
        column_specifier = target_column + '_back_' + str(diff_order) + '_order_diff_'

        if diff_order == 0:
            # shift only when diff order is 0, because we would not do differencing
            for backward_i in range(1, look_backward_days + 1):
                data[column_specifier + str(backward_i)] = data[shift_column_specifier].shift(backward_i)
        else:
            # differencing when diff_order is at least zero
            for backward_i in range(diff_order, look_backward_days + 1):
                data[column_specifier + str(backward_i)] = data[shift_column_specifier + str(backward_i - 1)] - \
                                                           data[shift_column_specifier + str(backward_i)]
        del data[target_column + '_back_0_order_diff_0']
        return data

    def generate_additional_time_features(self, data, look_backward_days, look_forward_days, date_column="date"):
        """ Generate month, weekday and day features according to provided column date_column.
        Besides, we would also trim data.

        :param data:
        :param look_backward_days:
        :param look_forward_days:
        :param date_column:
        :return:
        """
        data.reset_index(drop=True, inplace=True)
        data[date_column] = pd.to_datetime(data[date_column])
        if self.additional_time_features:
            data['weekday'] = data[date_column].dt.weekday
            data['month'] = data[date_column].dt.month
            data['day'] = data[date_column].dt.day
        data[date_column] = data[date_column].dt.date

        data = data[(data[date_column] >= data[date_column][look_backward_days])]  # &  # problem here
        # (data[date_column] < data[date_column][data.shape[0] - look_forward_days])]
        data.reset_index(drop=True, inplace=True)
        return data

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        X_simplified_diff = self.generate_look_forward_features(data=X.copy(),
                                                                target_column=self.target_column,
                                                                look_forward_days=self.look_forward_days)

        if self.diff_order < 0:
            pass
        else:
            for diff_order_i in range(self.diff_order + 1):
                X_simplified_diff = self.generate_diff_features(X_simplified_diff,
                                                                self.target_column,
                                                                self.look_backward_days,
                                                                diff_order_i)
        X_enhanced_diff = self.generate_additional_time_features(
            X_simplified_diff,
            self.look_backward_days,
            self.look_forward_days
        )

        return X_enhanced_diff

