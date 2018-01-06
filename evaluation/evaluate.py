import numpy as np
import pandas as pd


class Evaluate:

    """
    given predict_y, true_y and error method, calculate accuracy and fluctuation point accuracy
    """
    def __init__(self, predict_y, true_y, error):
        self.predict_y = predict_y
        self.true_y = true_y
        self.error = error

    def accuracy(self):
        """
        calculate the accuracy by error
        :return:  float, the %accuracy of predict_y and true_y
        """
        cnt = 0
        for i, j in zip(self.predict_y, self.true_y):
            # if abs(i-j)/j <= error:
            if abs(i - j) < self.error:
                cnt += 1
        acc = cnt/len(self.predict_y)*100

        return acc

    def fluctuation_accuracy(self):
        """
        calculate the fluctuation point accuracy by error
        :return: float, the fluctuation %accuracy of predict_y and true_y
        """
        pass
