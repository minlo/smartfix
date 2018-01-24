from __future__ import absolute_import

import os
import pandas as pd
import numpy as np
import datetime
import time
import sys

sys.path.append('./../')
from data_processing import GenerateDataFrame
from evaluation import Evaluate
from definitions import MODEL_SAVE_DIR, MODEL_PARAMS_JSON_DIR, MODEL_TRAINING_EVAL_RESULTS_DIR, \
    MODEL_TRAINING_FAILED_MODELS_DIR, BEST_MODEL_HISTORY_DIR, PREDICTION_RESULTS_ALL_MODELS_DIR, \
    PREDICTION_RESULTS_BEST_MODEL_DIR, DATA_DIR
from config.warning_model_config import imputer_param_grid, engineer_param_grid, selector_param_grid, \
    reducer_param_grid, model_dict, model_param_grid_dict, model_pipeline_mode_dict
from utils import search_model_ml, test

import logging
import argparse
# import json


# setting logging configuration
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# define constants
MODEL_PARAMS_JSON_PATH = MODEL_PARAMS_JSON_DIR + "warning_model_params.json"
MODEL_TRAINING_EVAL_RESULTS_PATH = MODEL_TRAINING_EVAL_RESULTS_DIR + "warning_model_train_eval_history_"
MODEL_TRAINING_FAILED_MODELS_PATH = MODEL_TRAINING_FAILED_MODELS_DIR + "warning_model_failed_history_"
BEST_MODEL_HISTORY_PATH = BEST_MODEL_HISTORY_DIR + "warning_best_model_history_"
PREDICTION_RESULTS_ALL_MODELS_PATH = PREDICTION_RESULTS_ALL_MODELS_DIR + "warning_predict_all_models_step_"
PREDICTION_RESULTS_BEST_MODEL_PATH = PREDICTION_RESULTS_BEST_MODEL_DIR + "warning_predict_best_model_step_"


if __name__ == "__main__":
    total_start_time = time.time()
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--split_date", help="date to split training and testing, must be in YYYYmmdd format",
                        required=False, default=datetime.date.today().strftime("%Y%m%d"), type=str)
    parser.add_argument("--look_forward_days", help="how many days to look forward",
                        required=False, default=1, type=int)
    parser.add_argument("--data_path", help="select which data to import for training or testing",
                        required=False, default="raw_data_2018-01-05.csv", type=str)
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
    parser.add_argument("--scoring_criterion", help="scoring criterion when running grid search",
                        required=False, default="roc_auc", type=str)
    args = parser.parse_args()

    # import data
    data = GenerateDataFrame(
        raw_data_url=os.path.join(DATA_DIR, args.data_path)
    ).data_to_dataframe()
    data.rename(columns={data.columns[-1]: "y"}, inplace=True)

    # generate forward_y as response variable
    data["forward_y"] = data["y"].shift((-1) * args.look_forward_days)
    # convert forward_y to be integer
    # data["forward_y"] = data["forward_y"].astype("int")

    split_date = datetime.datetime.strptime(args.split_date, "%Y%m%d").date()
    data.index = data.index.date
    data_train = data[data.index < split_date]
    data_train["forward_y"] = data_train["forward_y"].astype("int")
    data_test = data.copy()
    x_test = data_test.copy()
    del x_test["forward_y"]
    logger.info("data_train: {}, data_test: {}".format(data_train.shape, data_test.shape))

    # if is_training is True, we run grid search on multiple combinations of the pipeline to select the best one
    if args.is_training:
        logger.info("Setting is_training to be true, we run grid search on look_forward_days to be {}".format(
            args.look_forward_days
        ))
        search_model_ml(
            data_train=data_train,
            save_k_best=args.save_k_best,
            look_ahead_day=args.look_forward_days,
            split_date=split_date,
            validation_period_length=args.validation_period_length,
            model_save_dir=MODEL_SAVE_DIR,
            model_training_failed_models_path=MODEL_TRAINING_FAILED_MODELS_PATH,
            model_training_eval_results_path=MODEL_TRAINING_EVAL_RESULTS_PATH,
            imputer_param_grid=imputer_param_grid,
            engineer_param_grid=engineer_param_grid,
            selector_param_grid=selector_param_grid,
            reducer_param_grid=reducer_param_grid,
            model_dict=model_dict,
            model_param_grid_dict=model_param_grid_dict,
            model_pipeline_mode_dict=model_pipeline_mode_dict,
            scoring_criterion=args.scoring_criterion
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
    model_results = model_results.drop_duplicates(['model_id'])  # save best models for each model family
    model_results.reset_index(drop=True, inplace=True)

    logger.info(
        "By filtering the best eval_metric, "
        "we select the only model {}, with eval_metric: {}".format(
            model_results['model_id'][0],
            model_results['eval_metric'][0]
        )
    )

    predict_results_all_models_data_all_models_file = PREDICTION_RESULTS_ALL_MODELS_PATH + str(
        args.look_forward_days) + ".csv"
    predict_results_all_models_data = pd.DataFrame(columns=["date", "look_forward_days", "model_name", "model_id", "y",
                                                            "forward_y", "predict_y", "prediction_date", "timestamp"])

    for index_i in range(len(model_results.index)):
        try:
            logger.info("Model: {}, model_name: {}, metric: {}".format(
                model_results['model_id'][index_i],
                model_results['model_name'][index_i],
                model_results['eval_metric'][index_i]
            ))

            predict_results_all_models_data_i = data_test[["y", "forward_y"]]
            predict_results_all_models_data_i['look_forward_days'] = args.look_forward_days
            predict_results_all_models_data_i['date'] = data_test.index
            predict_results_all_models_data_i['date'] = pd.to_datetime(predict_results_all_models_data_i['date'])
            predict_results_all_models_data_i['date'] = predict_results_all_models_data_i['date'].dt.date
            predict_results_all_models_data_i["model_name"] = model_results['model_name'][index_i]
            predict_results_all_models_data_i["model_id"] = model_results['model_id'][index_i]
            predict_results_all_models_data_i["predict_y"] = test(
                x_test=x_test,
                model_id=model_results['model_id'][index_i]
            )
            predict_results_all_models_data_i["prediction_date"] = datetime.date.today().strftime("%Y%m%d")
            predict_results_all_models_data_i["timestamp"] = int(time.time() * 1000)
            predict_results_all_models_data_i = predict_results_all_models_data_i[
                predict_results_all_models_data_i['date'] >= split_date]

            predict_results_all_models_data_i = predict_results_all_models_data_i[["date", "look_forward_days",
                                                                                   "model_name", "model_id",
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

        print("All model dates: ")
        print(predict_all_models_dates)
        # specify start and end dates for eval dynamic period
        end_dynamic_eval_date = split_date
        try:
            start_dynamic_eval_date = predict_all_models_dates[(-1) * args.dynamic_eval_last_days]
        except:
            start_dynamic_eval_date = predict_all_models_dates[0]

        predict_results_all_models_history = predict_results_all_models_history[
            (predict_results_all_models_history['date'] >= start_dynamic_eval_date) &
            (predict_results_all_models_history['date'] < end_dynamic_eval_date)]

        # initialize best model data
        best_model_data = pd.DataFrame(columns=["look_forward_days", "model_name", "model_id", "dynamic_eval_metric",
                                                "start_eval_date", "end_eval_date", "update_date", "timestamp"])
        model_list = predict_results_all_models_history['model_id'].unique().tolist()

        y_true_history = np.array(data_test[(data_test.index >= start_dynamic_eval_date) &
                                            (data_test.index < end_dynamic_eval_date)]["forward_y"].tolist())
        for model_id_i in model_list:
            y_test_predict = np.array(
                predict_results_all_models_history[(predict_results_all_models_history["model_id"] == model_id_i)][
                    "forward_y"].tolist())
            dynamic_eval_metric = Evaluate(y_test_predict, y_true_history, error=0.10).accuracy()
            model_name_i = predict_results_all_models_history[predict_results_all_models_history["model_id"] == model_id_i]["model_name"].unique().tolist()[0]
            best_model_data.loc[len(best_model_data.index)] = [args.look_forward_days, model_name_i, model_id_i,
                                                               dynamic_eval_metric,
                                                               start_dynamic_eval_date,
                                                               end_dynamic_eval_date,
                                                               datetime.date.today().strftime("%Y%m%d"),
                                                               int(1000 * time.time())]

        best_model_data.sort_values(["dynamic_eval_metric"], ascending=[False], inplace=True)
        best_model_data.reset_index(drop=True, inplace=True)
        best_model_data = best_model_data.loc[best_model_data.index == 0]

        if not os.path.exists(best_model_file):
            best_model_data.to_csv(best_model_file, encoding="utf-8", index=None, mode="a", header=True)
        else:
            best_model_data.to_csv(best_model_file, encoding="utf-8", index=None, mode="a", header=False)

    # fetch the best model and save its predictions into PREDICTION_RESULTS_BEST_MODEL_PATH
    predict_best_model_file = PREDICTION_RESULTS_BEST_MODEL_PATH + str(args.look_forward_days) + ".csv"
    try:
        best_model_history = pd.read_csv(best_model_file, encoding="utf-8")
    except:
        best_model_history = pd.read_csv(model_history_file, encoding="utf-8")
    best_model_history.sort_values(["timestamp"], ascending=[False], inplace=True)
    best_model_history.reset_index(drop=True, inplace=True)
    best_model_id = best_model_history['model_id'][0]
    best_model_name = best_model_history['model_name'][0]
    logger.info("We have selected best model, model_id: {}, model_name: {}".format(best_model_id, best_model_name))
    predict_results_best_models_data = predict_results_all_models_data[
        predict_results_all_models_data["model_id"] == best_model_id]

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

