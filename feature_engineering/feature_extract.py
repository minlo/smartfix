import numpy as np
import pandas as pd
import math
from sklearn.base import BaseEstimator, TransformerMixin

"""
work tonight:
1. data.value.copy() ok
2. lag of y ok
3. time point ok
"""


class FeatureExtract(BaseEstimator, TransformerMixin):
    """
    Extract different kind of features of original data, MA, Log, Index, ChangeRate, Diff,etc
    return all the feature extracted merged together

    """
    def __init__(self, ma=[1, 2, 3, 4, 5], log=True, ind=[2, 3], changerate=True,
                 diff=True, lag=10, look_forward_days=1):
        """
        :param ma: int list, what ma days need to calculate
        :param log: bool, whether to calculate log
        :param ind: int list, what index need to extract from raw data
        :param changerate: bool, whether to calculate changerate
        :param diff: bool, whether diff need to calculate
        """

        # self.method_list = method_list
        self.ma = ma
        self.log = log
        self.ind = ind
        self.changerate = changerate
        self.diff = diff
        self.lag = lag
        self.look_forward_days = look_forward_days

    @staticmethod
    def generate_ma(data, ma_days):
        """
        :param data: np.array
        :param ma_days: int
        :return: np.array
        """
        # print(type(data))
        # print(data)
        new_data_list = []
        for i in range(0, data.shape[1]):
            ma_list = []
            for k in range(0, ma_days):
                ma_list.append(None)
            for j in range(ma_days, data.shape[0]):
                ma_add = 0
                for k in range(j-ma_days, j):
                    # print(ma_add, data[k][i])
                    ma_add += data[k][i]
                ma_list.append(ma_add / ma_days)
            new_data_list.append(ma_list)
        new_data = np.array(new_data_list).transpose()
        return new_data

    @staticmethod
    def generate_log(data):
        """
        :param data: np.array
        :return: np.array
        """

        new_data_list = []
        for i in range(0, data.shape[1]):
            log_list = []
            for j in range(0, data.shape[0]):
                num = abs(data[j][i])
                if num != 0:
                    log_list.append(math.log(num))
                else:
                    log_list.append(np.finfo(np.float64).min)
            new_data_list.append(log_list)
        new_data = np.array(new_data_list).transpose()
        return new_data

    @staticmethod
    def generate_index(data, index):
        """
        :param data: np.array
        :param index: int
        :return: np.array
        """

        new_data_list = []
        for i in range(0, data.shape[1]):
            index_list = []
            for j in range(0, data.shape[0]):
                index_list.append(pow(data[j][i], index))
            new_data_list.append(index_list)
        new_data = np.array(new_data_list).transpose()
        return new_data

    @staticmethod
    def generate_change_rate(data):
        """
        :param data: np.array
        :return: np.array
        """
        new_data_list = []
        for i in range(0, data.shape[1]):
            rate_list = []
            rate_list.append(None)
            for j in range(1, data.shape[0]):
                if data[j-1][i] != 0:
                    rate_list.append(abs((data[j][i] - data[j-1][i]) / data[j-1][i]))
                else:
                    rate_list.append(np.finfo(np.float64).max)
            new_data_list.append(rate_list)
        new_data = np.array(new_data_list).transpose()
        return new_data

    @staticmethod
    def generate_diff(data):
        """
        :param data: np.array
        :return: np.array
        """
        new_data_list = []
        for i in range(0, data.shape[1]):
            diff_list = []
            diff_list.append(None)
            for j in range(1, data.shape[0]):
                diff_list.append(data[j][i]-data[j-1][i])
            diff_list[0] = diff_list[1]
            new_data_list.append(diff_list)
        new_data = np.array(new_data_list).transpose()
        return new_data

    @staticmethod
    def generate_lag(y, look_back):
        """
        :param y: np.array
        :param look_back: int
        :return: np.array
        """

        lag_y = []
        for i in range(1, look_back+1):
            lag = []
            for j in range(0, i):
                lag.append(None)
            for j in range(i, len(y)):
                lag.append(y[j-i])
            lag_y.append(lag)
        new_data = np.array(lag_y).transpose()
        return new_data

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

        if day+look_forward_days > month_list[month-1]:
            end_of_month = 1
        else:
            end_of_month = 0

        if(month == 3 or month == 6 or month == 9 or month == 12) and (day+look_forward_days > month_list[month-1]):
            end_of_season = 1
        else:
            end_of_season = 0

        if month == 12 and (day+look_forward_days > month_list[month-1]):
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
        """

        :param X: DataFrame
        :param y: DataFrame
        :return: DataFrame
        """
        
        data = X.copy()
        # data.index = pd.to_datetime(data.index)
        data_value = data.values
        data_index = data.index
        data_columns = data.columns
        # print(data_value)

        # add important time point feature
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
        data_time_point = []
        data_time_point.append(weekdays)
        data_time_point.append(end_of_month)
        data_time_point.append(end_of_season)
        data_time_point.append(end_of_year)
        data_array_time_point = np.array(data_time_point).transpose()
        time_point_columns = ['weekday', 'end_of_month', 'end_of_season', 'end_of_year']
        df_time_point = pd.DataFrame(data_array_time_point, index = data_index, columns = time_point_columns)
        data = pd.concat([data, df_time_point], axis=1)

        # add y lag into X
        data_y = data_value[:, -1].copy()
        y_lag_columns = []
        for i in range(0, self.lag):
            y_lag_columns.append('y_lag_'+str(i+1))
        data_lag = self.generate_lag(data_y.copy(), self.lag)
        df_lag = pd.DataFrame(data_lag, index=data_index, columns=y_lag_columns)
        data = pd.concat([data, df_lag], axis=1)

        # # predict y_lookforward
        # forward_y = []
        # for i in range(0, len(data_y)-self.look_forward_days):
        #     forward_y.append(data_y[i+self.look_forward_days])
        # for i in range(len(data_y)-self.look_forward_days, len(data_y)):
        #     forward_y.append(None)
        # # df_y = pd.DataFrame(np.array(forward_y).transpose(), index=data_index, columns=['forward_y'])

        if len(self.ma) != 0:
            for ma_days in self.ma:
                data_ma_5 = self.generate_ma(data_value.copy()[:, :-1], ma_days)
                data_ma_columns = []
                for j in data_columns[:-1]:
                    data_ma_columns.append(j+'_ma_'+str(ma_days))
                df_ma = pd.DataFrame(data_ma_5, index=data_index, columns=data_ma_columns)
                data = pd.concat([data, df_ma], axis=1)

        if self.log:
            data_log = self.generate_log(data_value.copy()[:, :-1])
            data_log_columns = []
            for j in data_columns[:-1]:
                data_log_columns.append(j+'_log')
            df_log = pd.DataFrame(data_log, index=data_index, columns=data_log_columns)
            data = pd.concat([data, df_log], axis=1)

        if len(self.ind) != 0:
            for ind in self.ind:
                data_ind = self.generate_index(data_value.copy()[:, :-1], ind)
                data_ind_columns = []
                for j in data_columns[:-1]:
                    data_ind_columns.append(j+'_ind_'+str(ind))
                df_ind = pd.DataFrame(data_ind, index=data_index, columns=data_ind_columns)
                data = pd.concat([data, df_ind], axis=1)

        if self.changerate:
            data_changerate = self.generate_change_rate(data_value.copy()[:, :-1])
            data_changerate_columns = []
            for j in data_columns[:-1]:
                data_changerate_columns.append(j+'_changerate')
            df_changerate = pd.DataFrame(data_changerate, index=data_index, columns=data_changerate_columns)
            data = pd.concat([data, df_changerate], axis=1)

        if self.diff:
            data_diff = self.generate_diff(data_value.copy()[:, :-1])
            data_diff_columns = []
            for j in data_columns[:-1]:
                data_diff_columns.append(j+'_diff')
            df_diff = pd.DataFrame(data_diff, index=data_index, columns=data_diff_columns)
            data = pd.concat([data, df_diff], axis=1)

        data['forward_y'] = np.array(forward_y)
        # del data['银行间质押式回购加权利率:7天'] # print(data['forward_y'])
        # print(data['银行间质押式回购加权利率:7天'])
        # data.rename({"银行间质押式回购加权利率:7天": "y"}, inplace=True)
        # print(data['y'])
        return data




