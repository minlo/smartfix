import numpy as np
import pandas as pd
import math
from sklearn.base import BaseEstimator, TransformerMixin


class FeatureExtract(BaseEstimator, TransformerMixin):
    """
    Extract different kind of features of original data, MA, Log, Index, ChangeRate, Diff,etc
    return all the feature extracted merged together

    """
    def __init__(self, method_list, ma=[5], log=True, ind=[2], changerate=True,
                 diff=True, lag=10):
        """

        :param method_list:
        :param ma: int list, what ma days need to calculate
        :param log: bool, whether to calculate log
        :param ind: int list, what index need to extract from raw data
        :param changerate: bool, whether to calculate changerate
        :param diff: bool, whether diff need to calculate
        """

        self.method_list = method_list
        self.ma = ma
        self.log = log
        self.ind = ind
        self.changerate = changerate
        self.diff = diff

    @staticmethod
    def generate_ma(self, data, ma_days):
        """

        :param self:
        :param data: np.array
        :param ma_days: int
        :return: np.array
        """

        new_data_list = []
        for i in range(0, data.shape[1]):
            ma_list = []
            for j in range(0, data.shape[0] - ma_days):
                ma_add = 0
                for k in range(j, j + ma_days):
                    ma_add += data[k][i]
                ma_list.append(ma_add / ma_days)
                ma_last = ma_add / ma_days
            for w in range(data.shape[0] - ma_days, data.shape[0]):
                ma_list.append(ma_last)
            new_data_list.append(ma_list)

        new_data = np.array(new_data_list).transpose()
        return new_data

    @staticmethod
    def genrate_log(self, data):
        """

        :param self:
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
                    log_list.append(None)
            new_data_list.append(log_list)
        new_data = np.array(new_data_list).transpose()
        return new_data

    @staticmethod
    def generate_index(self, data, index):
        """

        :param self:
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
    def generate_change_rate(self, data):
        """

        :param self:
        :param data: np.array
        :return: np.array
        """
        new_data_list = []
        day_before = 0.0
        day_now = 0.0
        for i in range(0, data.shape[1]):
            day_before = data[0][i]
            rate_list = []
            rate_list.append(abs((data[1][i] - day_before) / day_before))
            for j in range(1, data.shape[0]):
                day_now = data[j][i]
                rate_list.append(abs((day_now - day_before) / day_before))
            new_data_list.append(rate_list)
        new_data = np.array(new_data_list).transpose()
        return new_data

    @staticmethod
    def generate_diff(self, data):
        """

        :param self:
        :param data: np.array
        :return: np.array
        """
        new_data_list = []
        for i in range(0, data.shape[1]):
            diff_list = []
            diff_list.append(0)
            for j in range(1, data.shape[0]):
                diff_list.append(data[j][i]-data[j-1][i])
            diff_list[0] = diff_list[1]
            new_data_list.append(diff_list)
        new_data = np.array(new_data_list).transpose()
        return new_data


    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        """

        :param X: DataFrame
        :param y:
        :return: DataFrame
        """

        data = X.copy()
        data_value = data.values
        data_index = data.index
        data_columns = data.columns
        #for i in self.method_list:
        if len(self.ma) != 0:
            for ma_days in self.ma:
                data_ma_5 = self.generate_ma(data_value, ma_days)
                data_ma_columns = []
                for j in data_columns:
                    data_ma_columns.append(j+'_ma_'+str(ma_days))
                df_ma = pd.DataFrame(data_ma_5, index=data_index, columns=data_ma_columns)
                data = pd.concat([data, df_ma], axis=1)

        if self.log == True:
            data_log = self.genrate_log(data_value)
            data_log_columns = []
            for j in data_columns:
                data_log_columns.append(j+'_log')
            df_log = pd.DataFrame(data_log, index=data_index, columns=data_log_columns)
            data = pd.concat([data, df_log], axis=1)

        if len(self.ind) != 0:
            for ind in self.ind:
                data_ind = self.generate_index(data_value, ind)
                data_ind_columns = []
                for j in data_columns:
                    data_ind_columns.append(j+'_ind_'+str(ind))
                df_ind = pd.DataFrame(data_ind, index=data_index, columns=data_ind_columns)
                data = pd.concat([data, df_ind], axis=1)

        if self.changerate == True:
            data_changerate = self.generate_change_rate(data_value)
            data_changerate_columns = []
            for j in data_columns:
                data_changerate_columns.append(j+'_changerate')
            df_changerate = pd.DataFrame(data_changerate, index=data_index, columns=data_changerate_columns)
            data = pd.concat([data, df_changerate], axis=1)

        if self.diff == True :
            data_diff = self.generate_diff(data_value)
            data_diff_columns = []
            for j in data_columns:
                data_diff_columns.append(j+'_diff')
            df_diff = pd.DataFrame(data_diff, index=data_index, columns=data_diff_columns)
            data = pd.concat([data, df_diff], axis=1)

        return data




