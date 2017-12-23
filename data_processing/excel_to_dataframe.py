import pandas as pd
import numpy as np


class GenerateDataFrame:
    """
    The class process on the origin excel dataset, it has two functions
    1. transform excel dataset to dataframe
    2. make the columns name to dictionaries
    """
    def __init__(self, raw_feature_url, r007_url, warning_url):
        self.raw_feature_url = raw_feature_url
        self.r007_url = r007_url
        self.warning_url = warning_url

    def feature_to_dataframe(self):
        """

        :return: DataFrame
        """
        data = pd.read_excel(self.raw_feature_url, index_col='指标名称')
        data.index = data.index.rename('date')
        data.index = pd.to_datetime(data.index)
        old_colum_list = data.columns
        new_column_list = []
        for i in range(data.shape[1]-1):
            new_column_list.append('x' + str(i + 1))
        new_column_list.append('y')
        data.columns = new_column_list

        return data

    def r007_to_dataframe(self):
        """
        :return: DataFrame
        """
        r007 = pd.read_excel(self.r007_url, index_col='指标名称')
        r007.index.rename('date', inplace=True)
        r007.index = pd.to_datetime(r007.index)
        r007.columns = ['R007']

        return r007

    def warning_to_dataframe(self):
        """

        :return: DataFrame
        """
        warning = pd.read_excel(self.warning_url, index_col='指标名称')
        warning.index.rename('date', inplace=True)
        warning.index = pd.to_datetime(warning.index)
        warning.columns = ['warning']

        return warning



