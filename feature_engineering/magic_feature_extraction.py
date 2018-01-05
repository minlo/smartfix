import numpy as np
import pandas as pd
import math
from sklearn.base import BaseEstimator, TransformerMixin


class MagicFeatureExtract(BaseEstimator, TransformerMixin):
    """
    This class extract the magic features of origin data, like cross special date feature, holiday feature

    This class will be used in the FeatureUnion part of our pipeline
    """

    def __init__(self):
        pass

    @staticmethod
    def date_is_which(date, look_forward_days):
        """
        determine whether a date is end of month? end of season? end of year?
        :param date: pd.datetime
        :param look_forward_days: int
        :return: whether end of month, end of season, end of year
        """
        year = date.year
        month = date.month
        day = date.day
        if year % 100 == 0 or year % 4 == 0:
            month_list = [31, 29, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]
        else:
            month_list = [31, 29, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]

        if day + look_forward_days > month_list[month - 1]:
            end_of_month = 1
        else:
            end_of_month = 0

        if (month == 3 or month == 6 or month == 9 or month == 12) and (
                day + look_forward_days > month_list[month - 1]):
            end_of_season = 1
        else:
            end_of_season = 0

        if month == 12 and (day + look_forward_days > month_list[month - 1]):
            end_of_year = 1
        else:
            end_of_year = 0

        return end_of_month, end_of_season, end_of_year

    def fit(self, X, y=None):
        """

        :param X: DataFrame
        :param y: DataFrame
        :return: self
        """
        return self

    def transform(self, X, y=None):
        # add important time point feature
        data = X.copy()
        data_index = data.index
        weekdays = []
        end_of_month = []
        end_of_season = []
        end_of_year = []
        for i in data_index:
            weekdays.append(i.dayofweek)
            is_end_of_month, is_end_of_season, is_end_of_year = self.date_is_which(i, self.look_forward_days)
            end_of_month.append(is_end_of_month)
            end_of_season.append(is_end_of_season)
            end_of_year.append(is_end_of_year)
        data['weekdays'] = np.array(weekdays)
        data['end_of_month'] = np.array(end_of_month)
        data['end_of_season'] = np.array(end_of_season)
        data['end_of_year'] = np.array(end_of_year)
        return data

