# coding: utf-8


import logging
import pandas as pd
import numpy as np


class Fluctuation:
    """
    Definition of Fluctuation Point:
    Pt->Pt+1: |ln(Pt+1_predict)-ln(pt_true)|>=0.1
    pt->pt+7: |ln(pt+7_predict)-ln(pt_true)|>=0.15
    pt->t+30: |ln(pt+30_predict)-ln(pt_true)|>=0.2

    parameters:
    data: pandas dataframe; original dataframe with y column as the repurcharsing rate, which is pt in the above example
    mode: string; "univariate"(models using only y to train and test, e.g. ARCH) or "multivariate"(models using y and x
                    to train, e.g. random forest).

    to_compare_column:string; only for "multivariate" mode; "y_forward_1", "y_forward_7", "y_forward_30" are anticipated.
    to_predict: list; the days need to be predict; default:[1,7,30].
    criterion: list; the definition of Fluctuation Point; default: [0.1,0.15,0.2].

    return:
    fluctuation_index: two dimension matrix, each row representing the index of fluctation point(starting from 0) for the corresponding
                        day to predict.
    or date of fluctuation points:two dimension matrix, each row representing the date of fluctation point(starting from 0) for the corresponding
                        day to predict.
    """

    def __init__(self, data, mode="univariate", to_compare_column=None, to_predict=[1, 7, 30],
                 criterion=[0.1, 0.15, 0.2]):
        # transform the repurchasing rate to p.float type
        self.mode = mode.lower()
        self.raw_data = data.copy()
        if self.mode == "univariate":
            self.data = data['y'].astype(np.float)
            self.data.index = pd.to_datetime(self.data.index)
            self.data = self.data.values

            self.to_predict = to_predict
            self.criterion = criterion
            if len(to_predict) != len(criterion):
                raise ValueError('Shapes of to_predict and criterion are different.')

        elif self.mode == "multivariate":
            anticipated_column_names = ["y_forward_1", "y_forward_7", "y_forward_30"]
            if to_compare_column not in anticipated_column_names:
                raise ValueError("Column Name must be one of y_forward_1,y_forward_7,y_forward_30.")

            self.data_to_predict, self.data_to_compare = data['y'].values, data[to_compare_column].values
            self.to_predict = to_predict[anticipated_column_names.index(to_compare_column)]
            self.criterion = criterion[anticipated_column_names.index(to_compare_column)]
        else:
            raise ValueError("Unsupported mode, it should be one of univariate or multivariate.")

    def selecting(self):
        """
        the main process to identify fluctuation points
        """
        fluctuation_index = []
        if self.mode == "univariate":
            for i, k in enumerate(self.to_predict):
                fluctuation_index.append(
                    np.where(np.abs(np.log(self.data[k:]) - np.log(self.data[:-k])) >= self.criterion[i])[0] + k)
        else:
            fluctuation_index = np.where(np.abs(np.log(self.data_to_predict) - np.log(self.data_to_compare)) >= self.criterion)[0]
        return fluctuation_index

    def selecting_date(self):
        fluctuation_index = self.selecting()
        fluctuation_date = []
        for i in range(3):
            fluctuation_date.append(self.raw_data.index.values[fluctuation_index[i]])
        return fluctuation_date

