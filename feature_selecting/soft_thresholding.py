from sklearn.feature_selection import SelectFromModel
from sklearn.linear_model import Lasso


class SoftThresholdSelector(object):
    """
    Offer a way to package the sklearn SelectFromModel class.
    User could provide their own model, for instance, Lasso to call soft_filter to generate their own
    SelectFromModel object.
    """
    def __init__(self, model=Lasso(alpha=0.1)):
        self.model = model

    def soft_filter(self):
        return SelectFromModel(self.model, prefit=False)

