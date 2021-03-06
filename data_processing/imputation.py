from sklearn.base import BaseEstimator, TransformerMixin
import pandas as pd
import numpy as np
from scipy import interpolate


"""
Bugs to fix:

Notes on 2018-01-06:
1. problem hinges in function method_imputed()
    When we call "cubic" method, a ValueError would be raised concerning the index of x. Specific code is this: 
            xnew = np.linspace(x[0], x[-1], x[-1] - x[0] + 1)

"""



class ImputationMethod (BaseEstimator, TransformerMixin):
    """
    make merged data(which have day,month,year frequency data) imputated by different method
    
    remove argument self when the method declared as staticmethod on 2017-12-23 21:36.
    """
    def __init__(self, method):
        self.method = method

    @staticmethod
    def remove_missing_y(data):
        pass

    @staticmethod
    def remove_weekend(data):
        """
        :parameter data :DataFrame
        :return:
        """
        # the old version of clean data by weekend
        # data.index = pd.to_datetime(data.index)
        # for i in data.index:
            # if i.dayofweek == 5 or i.dayofweek == 6:
                # data.drop(i, inplace=True)
        # return data

        data.index = pd.to_datetime(data.index)
        data = data[~np.isnan(data['y'])]

        return data


    @staticmethod
    def direct_impute(data):
        """
        :return:
        """
        # print("Before filling na, data: {}, dropna {}".format(data.shape, data.dropna().shape))
        data_directly = data.fillna(method='pad')
        # print("After padding, data: {}, dropna: {}".format(data_directly.shape, data_directly.dropna().shape))
        data_directly = data_directly.fillna(method='bfill')
        # print("After bfilling, data: {}, dropna: {}".format(data_directly.shape, data_directly.dropna().shape))
        # exit(1)
        # Drop those columns where we cannot find any valid value in both backward and 
        # forward directions to fill the column. That is, we simply drop these columns with no sympathy.

        return data_directly

    # @staticmethod
    def method_imputed(self, data):
        """
        :return: DataFrame after imputed
        """
        data_value = data.values
        list_columns_fill = []
        for i in range(0, data_value.shape[1]-1):
            cnt_nan = 0.0
            for j in range(0, data_value.shape[0]):
                if np.isnan(data_value[j][i]):
                    cnt_nan += 1
            if cnt_nan / data_value.shape[0] >= 0.1:
                list_columns_fill.append(i)

        # method = "nearest","zero","slinear","quadratic","cubic"

        for i in list_columns_fill:
            data_filled_value = []
            data_filled_position = []
            for j in range(0, data.shape[0]):
                if not (np.isnan(data_value[j][i])):
                    data_filled_value.append(data_value[j][i])
                    data_filled_position.append(j)

            x = data_filled_position
            y = data_filled_value
            # f=interpolate.CubicSpline(x,y)
            print("length of x: ", len(x))
            print("x0: ", x[0])
            xnew = np.linspace(x[0], x[-1], x[-1] - x[0] + 1)
            if self.method == 'cubic':
                f = interpolate.CubicSpline(x, y)
            elif self.method == 'quadratic':
                f = interpolate.interp1d(x, y, kind=self.method)
            else:
                f = interpolate.interp1d(x, y, kind=self.method)
            ynew = f(xnew)
            # print(xnew)
            # print(ynew)
            for k in range(x[0], x[-1]):
                data_value[k, i] = ynew[k - x[0]]

        data_after_imputation = pd.DataFrame(np.array(data_value), index=data.index, columns=data.columns)
        data_imputated_distributed = data_after_imputation.fillna(method='pad')
        data_imputated_distributed = data_imputated_distributed.fillna(method='bfill')
        return data_imputated_distributed

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        # print(X)
        # X = self.remove_weekend(X)
        print("before imputation", X.shape, X.dropna().shape)

        if self.method == 'directly':
            # return self.direct_impute(X), y
            X = self.direct_impute(X)
        else:
            # return self.diffmethod_imputed(X), y
            X = self.method_imputed(X)
        X = X.dropna(axis=1, how="any")      
        # print("After imputation", X.shape, X.dropna().shape)
        # print("After imputation, X: {}".format(X[X.isnull()].shape))
        # print(X[X.isnull()].reset_index().loc[:5])
        return X

