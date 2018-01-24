from sklearn.base import BaseEstimator, TransformerMixin
import pandas as pd


class GenerateNDiffFeatures(BaseEstimator, TransformerMixin):
    """Fit and transform data to get diff features.

    target_column: str
        We are gonna generate diff features on this column.

    look_forward_days: int
        Number of days we are going to predict. This argument is pretty important as we calculate the real response
        variable from it. So it is necessary rather than optional.

    look_backward_days: int
        Number of days we are going to include as features.

    diff_order: int
        Number of orders we are going to do diff.

    additional_time_difference: bool, False by default
        If true, we would generate 'month', 'weekday', 'day' as additional features.

    date_column: str, 'date' by default
        The date column of the pandas data frame.
    """
    def __init__(self, target_column, look_forward_days, look_backward_days, diff_order,
                 addition_time_features=False, date_column="date"):
        self.target_column = target_column
        self.look_forward_days = look_forward_days
        self.look_backward_days = look_backward_days
        self.diff_order = diff_order
        self.additional_time_features = addition_time_features
        self.date_column = date_column

    @staticmethod
    def generate_look_forward_features(data, target_column, look_forward_days):
        """Generate separate pandas data frame for different look forward days.

        :param self:
        :param data:
        :param target_column:
        :param look_forward_days:
        :return:
        """
        data[target_column + '_forward_' + str(look_forward_days)] = data[target_column].shift((-1) * look_forward_days)
        data.reset_index(drop=True, inplace=True)
        return data

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

