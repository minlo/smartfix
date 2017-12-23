import os
import pandas as pd
import numpy as np
import time
import datetime
import sys
import operator

import matplotlib.pyplot as plt
sys.path.append('./../')
from pipeline_lib import TrainTestForTimeSeries, GeneratePipeline, BuildPipeline
from sklearn.externals import joblib


def train(imputer, engineer, selector, scaler, reducer, model, X, y, model_id=""):
    """Train historical data and save the model into pickle file."""
    pipeline = BuildPipeline(
        imputer=imputer,
        engineer=engineer,
        selector=selector,
        scaler=scaler,
        reducer=reducer,
        model=model
    )
    pipeline.fit(X, y)
    model_save_path = os.path.join("../results/models/", "model_" + model_id + ".pkl")
    joblib.dump(pipeline, model_save_path)


def test(X, model_id=""):
    model_load_path = os.path.join("../results/models/", "model_" + model_id + ".pkl")
    if not os.path.exists(model_load_path):
        raise ValueError("model_id {} does not exist.".format(model_id))

    pipeline = joblib.load(model_load_path)
    y_predict = pipeline.predict(X)

    return y_predict

