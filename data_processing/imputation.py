from sklearn.base import BaseEstimator, TransformerMixin
import pandas as pd
import numpy as np
from scipy import interpolate


class ImputationMethod (BaseEstimator, TransformerMixin):
    """
    make merged data(which have day,month,year frequency data) imputated by different method

    """
    def __init__(self, method):
        self.method = method

    @staticmethod
    def remove_weekend(self, data):
        """
        :parameter data :DataFrame
        :return:
        """
        data.index = pd.to_datetime(data.index)
        for i in data.index:
            if (i.dayofweek == 5 or i.dayofweek == 6):
                data.drop(i, inplace=True)
        return data

    @staticmethod
    def direct_impute(self, data):
        """
        :return:
        """
        data_directly = data.fillna(method='pad')
        data_directly = data_directly.fillna(method='bfill')
        return data_directly

    @staticmethod
    def method_imputed(self, data):
        """
        :return: DataFrame after imputated
        """
        data_value = data.values
        list_columns_fill = []
        for i in range(0, data_value.shape[1]):
            cnt_nan = 0.0
            for j in range(0, data_value.shape[0]):
                if (np.isnan(data_value[j][i])):
                    cnt_nan += 1
            if (cnt_nan / data_value.shape[0] >= 0.1):
                list_columns_fill.append(i)

        # method = "nearest","zero","slinear","quadratic","cubic"

        data_copy = np.copy(data_value)
        for i in list_columns_fill:
            data_filled_value = []
            data_filled_position = []
            for j in range(0, data.shape[0]):
                if (not (np.isnan(data_copy[j][i]))):
                    data_filled_value.append(data_copy[j][i])
                    data_filled_position.append(j)

            x = data_filled_position
            y = data_filled_value
            # f=interpolate.CubicSpline(x,y)
            xnew = np.linspace(x[0], x[-1], x[-1] - x[0] + 1)
            if (self.method == 'cubic'):
                f = interpolate.CubicSpline(x, y)
            elif (self.method == 'quadratic'):
                f = interpolate.interp1d(x, y, kind=self.method)
            else:
                f = interpolate.interp1d(x, y, kind=self.method)
            ynew = f(xnew)
            # print(xnew)
            # print(ynew)
            for k in range(x[0], x[-1]):
                data[k, i] = ynew[k - x[0]]

        data_after_imputation = pd.DataFrame(np.array(data_copy), index=data.index, columns=data.columns)
        data_imputated_distributed = data_after_imputation.fillna(method='pad')
        data_imputated_distributed = data_imputated_distributed.fillna(method='bfill')
        return data_imputated_distributed

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):

        data_after_remove_weekend = self.remove_weekend(X)

        if self.method == 'directly':
            return self.direct_impute(data_after_remove_weekend)
        else:
            return self.diffmethod_imputed(data_after_remove_weekend)
