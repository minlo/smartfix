"""
This module is common for all train, validation and testing procedures.
Besides, it will be also general in the sense that it could be applied to predict multiple target variables
with the same interface.

We provide following functions to be utilized from outside modules.
    1) generate_grid_search()
        Call this function to generate a grid search prototype of a pipeline.
    2) train()
        Call this function to train a model on historical data.
    3) test()
        Call this function to test on new data or historical data not included in training set.
    4) evaluate()
        Call this function to use functionality provided in Evaluate class.
    5) train_test_split_by_position()
        Split train and test by date index position.
    6) save_pipeline()
        Save the fine tuned pipeline into file for later use.
    7) check_inf_nan()
        Check if infinity or nan exists in data.
    8) search_model_ml()
        The main wrapper to grid search on historical data to select the best model.
"""
# Authors: xuhaonan <>, zhaoyizhaoyi <zhaoyizhaoyi@sjtu.edu.cn>


from __future__ import absolute_import
import os
import pandas as pd
import numpy as np
import datetime
import time
import sys
import uuid

sys.path.append('./../')
from pipeline_lib import BuildPipeline
from feature_engineering import FeatureExtract
from feature_selecting import FeatureSelector
from data_processing import ImputationMethod
from evaluation import Evaluate

from sklearn.externals import joblib
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV, TimeSeriesSplit
# from sklearn.linear_model import Lasso
# from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import MinMaxScaler
from sklearn.decomposition import PCA
from xgboost import XGBRegressor, XGBClassifier
import logging
# import argparse
# import json


# setting logging configuration
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


def generate_grid_search(search_pipeline, pipeline_mode, param_grid, n_iter=5, verbose=3,
                         scoring_criterion="neg_mean_squared_error"):
    if pipeline_mode == "random":
        return RandomizedSearchCV(
            search_pipeline,
            cv=TimeSeriesSplit(),
            n_jobs=-1,
            param_distributions=param_grid,
            n_iter=n_iter,
            verbose=verbose,
            scoring=scoring_criterion
        )
    elif pipeline_mode == "grid":
        return GridSearchCV(
            search_pipeline,
            cv=TimeSeriesSplit(),
            n_jobs=-1,
            param_grid=param_grid,
            verbose=verbose,
            scoring=scoring_criterion
        )
    elif pipeline_mode == "single":
        return search_pipeline
    else:
        raise ValueError("pipeline mode must be chosen from ('random', 'grid', 'single')!")


def train(imputer, engineer, selector, scaler, reducer, model, x_train, y_train,
          pipeline_mode, pipeline_param_grid, scoring_criterion):
    """Train historical data and save the model into pickle file."""
    time_init = time.time()
    pipeline_builder = BuildPipeline(
        imputer=imputer,
        engineer=engineer,
        selector=selector,
        scaler=scaler,
        reducer=reducer,
        model=model
    )
    pipeline = pipeline_builder.build()

    pipeline_grid_search = generate_grid_search(
        search_pipeline=pipeline,
        pipeline_mode=pipeline_mode,
        param_grid=pipeline_param_grid,
        scoring_criterion=scoring_criterion
    )
    pipeline_grid_search.fit(x_train, y_train)
    logger.info("It takes {:.2f} seconds to train this model.".format(time.time() - time_init))

    # destroy the cached memory for this specific pipeline
    pipeline_builder.destroy_cache_memory()

    if pipeline_mode != "single":
        pipeline_grid_search = pipeline_grid_search.best_estimator_

    return pipeline_grid_search


def save_pipeline(pipeline_combined, model_id, model_save_dir):
    model_save_path = os.path.join(model_save_dir, "model_" + model_id + ".pkl")
    joblib.dump(pipeline_combined, model_save_path)
    logger.info("Model dumped into {}.".format(model_save_path))


def test(x_test, model_id="", pipeline=None):
    time_init = time.time()
    if pipeline is None:
        model_load_path = os.path.join("../results/models/", "model_" + model_id + ".pkl")
        if not os.path.exists(model_load_path):
            raise ValueError("model_id {} does not exist.".format(model_id))

        pipeline = joblib.load(model_load_path)

    # predict
    y_test_predict = pipeline.predict(x_test)
    logger.info("It takes {:.2f} seconds to predict.".format(time.time() - time_init))

    return y_test_predict


def evaluate(y_test_true, y_test_predict, error=0.10):
    return Evaluate(y_test_predict, y_test_true, error).accuracy()


def train_test_split_by_position(data, split_position):
    """Given split_position, subset records from last up to the split_position."""
    # train_len = data.shape[0] - split_position
    split_date = data.index[data.shape[0] - split_position]
    data_train = data[data.index < split_date]
    data_val = data[data.index >= split_date]
    y_train = data_train.as_matrix(["forward_y"])
    y_train = np.ravel(y_train)
    y_val = data_val.as_matrix(["forward_y"])
    y_val = np.ravel(y_val)
    del data_train["forward_y"]
    del data_val["forward_y"]
    return data_train, y_train, data_val, y_val


def check_inf_nan(np_array, array_name):
    if np.any(np.isnan(np_array)):
        logger.info("There are np.nan in {}".format(array_name))
    if np.any(np.isinf(np_array)):
        logger.info("There are np.inf in {}".format(array_name))


def search_model_ml(data_train, save_k_best, look_ahead_day, split_date, validation_period_length,
                    model_save_dir, model_training_failed_models_path,model_training_eval_results_path,
                    imputer_param_grid, engineer_param_grid, selector_param_grid, reducer_param_grid,
                    model_dict, model_param_grid_dict, model_pipeline_mode_dict,
                    scoring_criterion="neg_mean_squared_error"):
    results = pd.DataFrame(columns=["model_id", "split_date", "model_name", "eval_metric",
                                    "update_date", "timestamp"])
    failed_models_data = pd.DataFrame(columns=['model_name', 'split_date', 'error_message', 'update_date', 'timestamp'])

    model_count = 0
    time_init = time.time()
    save_model_dict = {}
    logger.info("Tuning models, {}: ".format(datetime.date.today().strftime("%Y%m%d")))

    x_train, y_train, x_val, y_val = train_test_split_by_position(data_train, validation_period_length)
    check_inf_nan(y_train, "y_train")
    check_inf_nan(y_val, "y_val")
    logger.info("x_train: {}, y_train: {}, x_val: {}, y_val: {}".format(
        x_train.shape, y_train.shape, x_val.shape, y_val.shape
    ))

    for model_name in model_dict.keys():
        logger.info("\n\n\n\n\nmodel: {}\n".format(model_name))
        time_start = time.time()
        pipeline_param_grid = model_param_grid_dict[model_name].copy()
        pipeline_param_grid.update(imputer_param_grid)
        pipeline_param_grid.update(engineer_param_grid)
        pipeline_param_grid.update(selector_param_grid)
        # pipeline_param_grid.update(reducer_param_grid)

        try:
            pipeline = train(
                imputer=ImputationMethod(method="directly"),
                engineer=FeatureExtract(lag=10, look_forward_days=look_ahead_day),
                selector=FeatureSelector(k=10, select_method="hard"),  # selector_dict[model_selector],
                scaler=MinMaxScaler(),
                reducer=PCA(n_components=10),  # temporarily not in use
                model=model_dict[model_name],
                x_train=x_train,
                y_train=y_train,
                pipeline_mode=model_pipeline_mode_dict[model_name],
                pipeline_param_grid=pipeline_param_grid,
                scoring_criterion=scoring_criterion
            )
            model_id = str(uuid.uuid4())
            y_test_predict = test(
                x_test=x_val,
                model_id=model_id,
                pipeline=pipeline
            )
            eval_metric = evaluate(np.array(y_test_predict), np.array(y_val))
            results.loc[len(results.index)] = [model_id, split_date,
                                               model_name, eval_metric,
                                               datetime.date.today().strftime("%Y%m%d"),
                                               int(1000 * time.time())]
            save_model_dict[model_id] = pipeline

            model_count += 1
            logger.info(
                "Model {}, "
                "model_id: {},"
                "model_name: {}, "
                "eval_metric: {}, "
                "using {:.2f} seconds".format(
                    model_count,
                    model_id,
                    model_name,
                    eval_metric,
                    time.time() - time_start
                )
            )
        except Exception as e:
            failed_models_data.loc[len(failed_models_data.index)] = [
                model_name,
                split_date,
                str(e),
                datetime.date.today().strftime("%Y%m%d"),
                int(1000 * time.time())
            ]
            logger.info(str(e))
    logger.info("We have run {} models, with {} failed, using {:.2f} seconds".format(
        model_count + 1,
        len(failed_models_data),
        time.time() - time_init
    ))
    logger.info("\n\nSaving all the failed models into file...\n")
    failed_model_file = model_training_failed_models_path + str(look_ahead_day) + ".csv"
    # save failed models into a csv
    if not os.path.exists(failed_model_file):
        failed_models_data.to_csv(failed_model_file, encoding="utf-8", header=True, index=None)
    else:
        failed_models_data.to_csv(failed_model_file, encoding="utf-8", header=False, index=None, mode="a")

    # sort the results
    results.sort_values(["model_name", "eval_metric"], ascending=[True, False], inplace=True)
    results.reset_index(drop=True, inplace=True)

    results = results.groupby("model_name", group_keys=False).apply(lambda g: g.nlargest(save_k_best, "eval_metric"))
    # results.loc[results.index < save_k_best]

    # save the best models during this training process
    for index_i in range(len(results.index)):
        model_id_i = results['model_id'][index_i]
        logger.info("Saving {} to disk...".format(model_id_i))
        save_pipeline(save_model_dict[model_id_i], model_id_i, model_save_dir)

        # save model params to json file
        # model_params_dict[model_id_i] = {
        # "pipeline": save_model_dict[model_id_i].get_params()
        # }

    model_eval_results_file = model_training_eval_results_path + str(look_ahead_day) + ".csv"
    # save results into a csv
    if not os.path.exists(model_eval_results_file):
        results.to_csv(model_eval_results_file, encoding="utf-8", header=True, index=None)
    else:
        results.to_csv(model_eval_results_file, encoding="utf-8", header=False, index=None, mode="a")

    # save model params
    # with open(MODEL_PARAMS_JSON_PATH, 'w') as fp:
    #     json.dump(model_params_dict, fp)


if __name__ == "__main__":
    logger.info("hello!")

