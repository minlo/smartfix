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
from data_processing import ImputationMethod, GenerateDataFrame
from evaluation import Evaluate

from sklearn.externals import joblib
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV, TimeSeriesSplit
from sklearn.linear_model import Lasso
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import MinMaxScaler
from sklearn.decomposition import PCA
from xgboost import XGBRegressor, XGBClassifier
import logging
import argparse
import json


"""
Features to be added, checked on 2017-12-26:
1. refit when predicting
    Now there exists some bugs concerning this functionality, changes have to be made to make this feature usable.

2. pipeline params dumps to json file
    It seems that some objects or steps in the pipeline is not json serializable. Problems have to be fixed to 
    normalize the functionality. 

3. combine two fragmented pipelines into one complete pipeline
    Currently, the problems hinges where we have to split the train and test manually. Ideally, if we can impute the 
    y_forward at the end of test_data, we may make this process one part of the feature_engineering module. 
    Then we can build one complete pipeline based on this.

4. Combine all variables and soft-thresholding into the picture
    Currently, we just overlook this two methods as hard thresholding only accepts dataframe while the other 
    two accepts numpy array. How to combine these two into one step, one has to redesign the related modules.


ChangeLog on 2018-01-05:
1. hard thresholding computation
    I would like to make t-statistics computation efficient by computing only once when first encountered. Later on, 
    whenever I have to use the t-statistics, I just check if it already existed in memory to avoid repeated computation,
     which is the source of inefficiency.

2. one complete pipeline for all steps
    Currently, the pipeline is still fragmented. From now on, I would try to build a single pipeline which is 
    convenient for parameter grid search.
    
3. FeatureUnion for magic features
    According to Prof. Zhu, we have to add some magic features, which would supposedly skip the feature engineering
    and feature selecting steps in the pipeline. We plan to realize this by using FeatureUnion.
    
4. to be added...


Progress on 2018-01-06:
1. one complete pipeline for all steps
    After several rounds of debugging, we have finished this part. Currently, it is still under test.

2. store hard thresholding results in memory
    We want to store the immediate results in memory for later access so that we can avoid keep repeating.
    It seems that sklearn provides an official method to cache in version 0.19.1. 
    See details in:
    http://scikit-learn.org/stable/auto_examples/plot_compare_reduction.html#caching-transformers-within-a-pipeline

3. add soft thresholding and null thresholding (all variables) (to be finished)
    We would like to add these two kinds of selecting methods during feature_selecting step.

"""


# setting logging configuration
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# define constants
MODEL_PARAMS_JSON_PATH = "./../results/model_history/regression_model_params.json"
MODEL_TRAINING_EVAL_RESULTS_PATH = "./../results/model_history/regression_model_train_eval_history_"
MODEL_TRAINING_FAILED_MODELS_PATH = "./../results/model_history/regression_model_failed_history_"
BEST_MODEL_HISTORY_PATH = "./../results/model_history/regression_best_model_history_"
PREDICTION_RESULTS_ALL_MODELS_PATH = "./../results/predict/regression_predict_all_models_step_"
PREDICTION_RESULTS_BEST_MODEL_PATH = "./../results/predict/regression_predict_best_model_step_"


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
          pipeline_mode, pipeline_param_grid):
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
        param_grid=pipeline_param_grid
    )
    pipeline_grid_search.fit(x_train, y_train)
    logger.info("It takes {:.2f} seconds to train this model.".format(time.time() - time_init))

    # destroy the cached memory for this specific pipeline
    pipeline_builder.destroy_cache_memory()

    if pipeline_mode != "single":
        pipeline_grid_search = pipeline_grid_search.best_estimator_

    return pipeline_grid_search


def save_pipeline(pipeline_combined, model_id):
    model_save_path = os.path.join("../results/models/", "model_" + model_id + ".pkl")
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


def search_regression_ml(data_train, save_k_best, look_ahead_day, split_date, validation_period_length):
    imputer_param_grid = {
        "imputer__method": ["directly"]
    }
    engineer_param_grid = {
        "engineer__lag": [10, 20, 30, 40, 50, 60]
    }

    selector_param_grid = {
        "selector__k": [10, 20, 30, 40, 50],
        "selector__select_method": ["hard"],  # "soft", "all"]
    }
    # temporarily not used
    reducer_param_grid = {
        "reducer__n_components": [10]
    }
    model_dict = {
        "random_forest": RandomForestRegressor(n_estimators=1000, n_jobs=-1, random_state=1234),
        "xgboost": XGBRegressor(),
        "lasso": Lasso(alpha=0.01, random_state=1234),
    }
    model_param_grid_dict = {
        "random_forest": {
            "model__n_estimators": [1000]
        },
        "xgboost": {
            "model__max_depth": range(2, 12, 2),
            "model__min_child_weight": range(2, 10, 2),
            "model__subsample": [i / 10.0 for i in range(6, 10)],
            "model__colsample_bytree": [0.1, 0.2, 0.3, 0.4, 0.5],
            "model__learning_rate": [0.01, 0.1]
        },
        "lasso": {
            "model__alpha": [1 / (10**x) for x in range(2, 10)]
        }
    }
    model_pipeline_mode_dict = {
        "random_forest": "grid",
        "xgboost": "random",
        "lasso": "random"
    }
    results = pd.DataFrame(columns=["model_id", "split_date", "model_name", "eval_metric",
                                    "update_date", "timestamp"])
    
    failed_models_data = pd.DataFrame(columns=['model_name', 'split_date', 'update_date', 'timestamp'])
    # if os.path.exists(MODEL_PARAMS_JSON_PATH):
    #     model_params_dict = json.loads(MODEL_PARAMS_JSON_PATH)
    # else:
    #     model_params_dict = {}

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
                pipeline_param_grid=pipeline_param_grid
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
    failed_model_file = MODEL_TRAINING_FAILED_MODELS_PATH + str(look_ahead_day) + ".csv"
    # save failed models into a csv
    if not os.path.exists(failed_model_file):
        failed_models_data.to_csv(failed_model_file, encoding="utf-8", header=True, index=None)
    else:
        failed_models_data.to_csv(failed_model_file, encoding="utf-8", header=False, index=None, mode="a")

    # sort the results
    results.sort_values(["model_name", "eval_metric"], ascending=[True, False], inplace=True)
    results.reset_index(drop=True, inplace=True)

    results = results.loc[:save_k_best]

    # save the best models during this training process
    for index_i in range(len(results.index)):
        model_id_i = results['model_id'][index_i]
        logger.info("Saving {} to disk...".format(model_id_i))
        save_pipeline(save_model_dict[model_id_i], model_id_i)

        # save model params to json file
        # model_params_dict[model_id_i] = {
            # "pipeline": save_model_dict[model_id_i].get_params()
        # }

    model_eval_results_file = MODEL_TRAINING_EVAL_RESULTS_PATH + str(look_ahead_day) + ".csv"
    # save results into a csv
    if not os.path.exists(model_eval_results_file):
        results.to_csv(model_eval_results_file, encoding="utf-8", header=True, index=None)
    else:
        results.to_csv(model_eval_results_file, encoding="utf-8", header=False, index=None, mode="a")

    # save model params
    # with open(MODEL_PARAMS_JSON_PATH, 'w') as fp:
    #     json.dump(model_params_dict, fp)


if __name__ == "__main__":
    total_start_time = time.time()
    parser = argparse.ArgumentParser()
    parser.add_argument("--split_date", help="date to split training and testing, must be in YYYYmmdd format",
                        required=False, default=datetime.date.today().strftime("%Y%m%d"), type=str)
    parser.add_argument("--look_forward_days", help="how many days to look forward",
                        required=False, default=1, type=int)
    parser.add_argument("--data_path", help="select which data to import for training or testing",
                        required=False, default="raw_data_20171222.xls", type=str)
    parser.add_argument("--is_training", help="if true, we run the grid search and "
                                              "select the best models in the validation period",
                        required=False, default=False, type=lambda x: (str(x).lower() == "true"))
    parser.add_argument("--validation_period_length", help="run validation on how many natural days before split date",
                        required=False, default=30, type=int)
    parser.add_argument("--dynamic_selecting", help="if true, we see which model did best in last period"
                                                    " (dynamic_eval_last_days). Otherwise, "
                                                    "we just use the best model to predict ",
                        required=False, default=False, type=lambda x: (str(x).lower() == "true"))
    parser.add_argument("--dynamic_eval_last_days", help="period to run dynamic evaluation",
                        required=False, default=7, type=int)
    parser.add_argument("--save_k_best", help="if training, select k best models to save after training",
                        required=False, default=1, type=int)
    args = parser.parse_args()

    data = GenerateDataFrame(
        raw_data_url=os.path.join("./../data/data_live/", args.data_path)
    ).data_to_dataframe()
    data.rename(columns={data.columns[-1]: "y"}, inplace=True)

    # generate forward_y as response variable
    data["forward_y"] = data["y"].shift((-1) * args.look_forward_days)
    split_date = datetime.datetime.strptime(args.split_date, "%Y%m%d").date()
    data.index = data.index.date
    data_train = data[data.index < split_date]
    data_test = data.copy()
    x_test = data_test.copy()
    del x_test["forward_y"]
    logger.info("data_train: {}, data_test: {}".format(data_train.shape, data_test.shape))

    # if is_training is True, we run grid search on multiple combinations of the pipeline to select the best one
    if args.is_training:
        logger.info("Setting is_training to be true, we run grid search on look_forward_days to be {}".format(
            args.look_forward_days
        ))
        search_regression_ml(
            data_train=data_train,
            save_k_best=args.save_k_best,
            look_ahead_day=args.look_forward_days,
            split_date=split_date,
            validation_period_length=args.validation_period_length
        )

    # set the model results path
    model_history_file = MODEL_TRAINING_EVAL_RESULTS_PATH + str(args.look_forward_days) + ".csv"
    model_results = pd.read_csv(model_history_file, encoding="utf-8")

    # select the relevant models by split_date and eval_metric, so that we are using the best model trained with
    # the most recent updated data
    model_results['split_date'] = pd.to_datetime(model_results['split_date'])
    model_results['split_date'] = model_results['split_date'].dt.date
    most_recent_split_date = model_results['split_date'].max()
    model_results = model_results[model_results['split_date'] == most_recent_split_date]
    best_eval_metric = model_results['eval_metric'].max()
    model_results = model_results[model_results['eval_metric'] == best_eval_metric]
    model_results.reset_index(drop=True, inplace=True)
    # print(most_recent_split_date, best_eval_metric)
    if model_results.shape[0] == 0:
        raise ValueError("No model left after filtering the best model recently!")

    logger.info(
        "By filtering the best eval_metric, "
        "we select the only model {}, with eval_metric: {}".format(
            model_results['model_id'][0],
            model_results['eval_metric'][0]        
        )
    )

    predict_results_all_models_data_all_models_file = PREDICTION_RESULTS_ALL_MODELS_PATH + str(args.look_forward_days) + ".csv"
    predict_results_all_models_data = pd.DataFrame(columns=["date", "model_name", "model_id", "y", "forward_y",
                                                            "predict_y", "prediction_date", "timestamp"])

    for index_i in range(len(model_results.index)):
        try:
            logger.info("Model: {}, model_name: {}, metric: {}".format(
                model_results['model_id'][index_i],
                model_results['model_name'][index_i],
                model_results['eval_metric'][index_i]
            ))

            # save prediction results into file
            predict_results_all_models_data_i = pd.DataFrame(columns=["date", "model_name", "model_id", "y",
                                                                      "forward_y", "predict_y", "prediction_date",
                                                                      "timestamp"])
            predict_results_all_models_data_i['date'] = data_test.index
            predict_results_all_models_data_i['date'] = pd.to_datetime(predict_results_all_models_data_i['date'])
            predict_results_all_models_data_i['date'] = predict_results_all_models_data_i['date'].dt.date
            predict_results_all_models_data_i["model_name"] = model_results['model_name'][index_i]
            predict_results_all_models_data_i["model_id"] = model_results['model_id'][index_i]
            predict_results_all_models_data_i["y"] = data_test["y"]
            predict_results_all_models_data_i["forward_y"] = data_test["forward_y"]
            predict_results_all_models_data_i["predict_y"] = test(
                x_test=x_test,
                model_id=model_results['model_id'][index_i]
            )
            predict_results_all_models_data_i["prediction_date"] = datetime.date.today().strftime("%Y%m%d")
            predict_results_all_models_data_i["timestamp"] = int(time.time() * 1000)
            predict_results_all_models_data_i = predict_results_all_models_data_i[predict_results_all_models_data_i['date'] >= split_date]
            predict_results_all_models_data_i = predict_results_all_models_data_i[["date", "model_name", "model_id",
                                                                                   "y", "forward_y", "predict_y",
                                                                                   "prediction_date", "timestamp"]]
            predict_results_all_models_data = predict_results_all_models_data.append(predict_results_all_models_data_i)
        except ValueError as e:
            logger.info("model {} does not exists".format(model_results['model_id'][index_i]))
            logger.info(str(e))

    if not os.path.exists(predict_results_all_models_data_all_models_file):
        predict_results_all_models_data.to_csv(predict_results_all_models_data_all_models_file, encoding="utf-8",
                                               index=None, mode="a", header=True)
    else:
        predict_results_all_models_data.to_csv(predict_results_all_models_data_all_models_file, encoding="utf-8",
                                               index=None, mode="a", header=False)

    # If specified, we would run dynamic selecting of best model, and update the best model in BEST_MODEL_HISTORY_PATH.
    # Besides, we would also select the prediction results of it into another file.
    best_model_file = BEST_MODEL_HISTORY_PATH + str(args.look_forward_days) + ".csv"
    if args.dynamic_selecting:
        logger.info("Dynamically selecting the best model in last {} days.".format(args.dynamic_eval_last_days))
        predict_results_all_models_history = pd.read_csv(predict_results_all_models_data_all_models_file,
                                                         encoding="utf-8")
        predict_results_all_models_history['date'] = pd.to_datetime(predict_results_all_models_history['date'])
        predict_results_all_models_history['date'] = predict_results_all_models_history['date'].dt.date
        predict_results_all_models_history.sort_values(['date', 'model_name'], ascending=[True, True], inplace=True)
        predict_all_models_dates = sorted(predict_results_all_models_history[predict_results_all_models_history['date']
                                                                             < split_date]['date'].unique().tolist())

        # specify start and end dates for eval dynamic period
        start_dynamic_eval_date = split_date
        try:
            end_dynamic_eval_date = predict_all_models_dates[(-1) * args.dynamic_eval_last_days]
        except IndexError:
            end_dynamic_eval_date = predict_all_models_dates[0]

        predict_results_all_models_history = predict_results_all_models_history[
            (predict_results_all_models_history['date'] < start_dynamic_eval_date) &
            (predict_results_all_models_history['date'] >= end_dynamic_eval_date)
        ]

        # initialize best model data
        best_model_data = pd.DataFrame(columns=["model_name", "model_id", "dynamic_eval_metric", "start_eval_date",
                                                "end_eval_date", "update_date", "timestamp"])
        model_list = predict_results_all_models_history['model_id'].unique().tolist()

        y_true_history = np.array(data_test[(data_test.index >= start_dynamic_eval_date) &
                                            (data_test.index < end_dynamic_eval_date)]["forward_y"].tolist())
        for model_id_i in model_list:
            y_test_predict = np.array(predict_results_all_models_history[(predict_results_all_models_history["model_id"] == model_id_i)]["forward_y"].tolist())
            dynamic_eval_metric = Evaluate(y_test_predict, y_true_history, error=0.10).accuracy()
            model_name_i = predict_results_all_models_history[predict_results_all_models_history["model_id"] == model_id_i]["model_name"].unique().tolist()[0]
            best_model_data.loc[len(best_model_data.index)] = [model_name_i, model_id_i, dynamic_eval_metric,
                                                               start_dynamic_eval_date,
                                                               end_dynamic_eval_date,
                                                               datetime.date.today().strftime("%Y%m%d"),
                                                               int(1000 * time.time())]

        best_model_data.sort_values(["dynamic_eval_metric"], ascending=[False], inplace=True)
        best_model_data = best_model_data.loc[0]

        if not os.path.exists(best_model_file):
            best_model_data.to_csv(best_model_file, encoding="utf-8", index=None, mode="a", header=True)
        else:
            best_model_data.to_csv(best_model_file, encoding="utf-8", index=None, mode="a", header=False)

    # fetch the best model and save its predictions into PREDICTION_RESULTS_BEST_MODEL_PATH
    predict_best_model_file = PREDICTION_RESULTS_BEST_MODEL_PATH + str(args.look_forward_days) + ".csv"
    best_model_history = pd.read_csv(best_model_file, encoding="utf-8")
    best_model_history.sort_values(["timestamp"], acending=[False], inplace=True)
    best_model_history.reset_index(drop=True, inplace=True)
    best_model_id = best_model_history['model_id'][0]
    best_model_name = best_model_history['model_name'][0]
    logger.info("We have selected best model, model_id: {}, model_name: {}".format(best_model_id, best_model_name))
    predict_results_best_models_data = predict_results_all_models_data[predict_results_all_models_data["model_id"] == best_model_id]

    # save it into file
    if not os.path.exists(predict_best_model_file):
        predict_results_best_models_data.to_csv(predict_best_model_file, encoding="utf-8",
                                                index=None, mode="a", header=True)
    else:
        predict_results_best_models_data.to_csv(predict_best_model_file, encoding="utf-8",
                                                index=None, mode="a", header=False)

    logger.info("It takes {:.2f} seconds to run this time.".format(time.time() - total_start_time))
    logger.info("Prediction finished!!!")
    print("Done! This will be removed later on.")

