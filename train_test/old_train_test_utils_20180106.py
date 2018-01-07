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
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV, TimeSeriesSplit
from sklearn.linear_model import Lasso, Ridge, ElasticNet, least_angle, LassoLars, LinearRegression
# from sklearn.feature_selection import chi2, SelectKBest, VarianceThreshold, SelectFromModel
from sklearn.svm import LinearSVC, SVR, SVC
from sklearn.ensemble import ExtraTreesRegressor, RandomForestClassifier, RandomForestRegressor
from sklearn.preprocessing import MinMaxScaler, StandardScaler, Imputer
from sklearn.decomposition import PCA
from xgboost import XGBRegressor, XGBClassifier
import logging
import logging.config
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

2. store hard thresholding results in memory (to be finished)
    We want to store the immediate results in memory for later access so that we can avoid keep repeating.

3. add soft thresholding and null thresholding (all variables) (to be finished)
    We would like to add these two kinds of selecting methods during feature_selecting step.

"""


# setting logging configuration
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


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

    # pipeline_1 = pipeline.build_before_selector()
    # pipeline_2 = pipeline.build_after_selector()
    #
    # X = pipeline_1.fit_transform(X)
    # # logger.info("X head after the feature engineering: {}".format(X.head()))
    # # logger.info(X.shape)
    # logger.info("It takes {:.2f} seconds to fit and transform in the first line!".format(time.time() - time_init))
    #
    # # delete all values after split_date
    # X.replace([np.inf, -np.inf], np.nan, inplace=True)
    # X = X.dropna()
    # # logger.info("X head after dropping na: {}".format(X.head()))
    # X.index = pd.to_datetime(X.index)
    # X.index = X.index.date
    # X = X[X.index < split_date]
    #
    # # logger.info("X head: {}".format(X.head()))
    # # maybe change to X = X.drop_na()
    # x_columns = list(X.columns)
    # x_columns.remove('forward_y')
    # # logger.info("x_columns: {}, unique: {}".format(len(x_columns), len(set(x_columns))))
    # # x_columns_set = []
    # # for index_i in range(len(x_columns)):
    # #     if x_columns[index_i] not in x_columns_set:
    # #         x_columns_set.append(x_columns[index_i])
    # #     else:
    # #         logger.info("duplicate column: {}".format(x_columns[index_i]))
    # #         break
    #
    # X_train = X.as_matrix(x_columns)
    # y_train = X.as_matrix(['forward_y'])
    # y_train = np.ravel(y_train)
    # logger.info("X_train: {}, y: {}".format(X_train.shape, y_train.shape))
    # # logger.info(np.where(X.values >= np.finfo(np.float64).max))
    # # logger.info(X.ix[339, 462])
    #
    # pipeline_2 = generate_grid_search(
    #     search_pipeline=pipeline_2,
    #     pipeline_mode=pipeline_mode,
    #     param_grid=pipeline_param_grid
    # )
    #
    # pipeline_2_init_time = time.time()
    # pipeline_2.fit(X_train, y_train)
    # logger.info("It takes {:.2f} seconds to fit in the second pipeline!".format(time.time() - pipeline_2_init_time))
    # logger.info("It takes {:.2f} seconds to train this model.".format(time.time() - time_init))
    # if pipeline_mode != "single":
    #     pipeline_2 = pipeline_2.best_estimator_
    #
    # pipeline_combined = Pipeline([
    #     ("pipeline_before_selector", pipeline_1),
    #     ("pipeline_after_selector", pipeline_2)
    # ])
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
    # X_copy = data_test[["y", "forward_y"]]
    # X_copy['predict_y'] = y_predict
    logger.info("It takes {:.2f} seconds to predict.".format(time.time() - time_init))
    # print(y_test_predict)
    # # before selector
    # pipeline_before_selector = pipeline_combined.named_steps["pipeline_before_selector"]
    # pipeline_after_selector = pipeline_combined.named_steps["pipeline_after_selector"]
    #
    # pipeline_before_selector_init_time = time.time()
    # X = pipeline_before_selector.transform(X)
    # logger.info("It takes {:.2f} seconds to fit in the first pipeline!".format(
    #     time.time() - pipeline_before_selector_init_time)
    # )
    #
    # # delete all values after split_date
    # X.replace([np.inf, -np.inf], np.nan, inplace=True)
    # X.index = pd.to_datetime(X.index)
    # X.index = X.index.date
    #
    # x_columns = list(X.columns)
    # x_columns.remove('forward_y')
    #
    # if refit:
    #     refit_init_time = time.time()
    #     data_train = X[X.index < split_date]
    #     data_train.dropna(inplace=True)
    #     X_train = data_train.as_matrix(x_columns)
    #     y_train = data_train.as_matrix(['forward_y'])
    #     y_train = np.ravel(y_train)
    #     logger.info("X_train: {}, y_train: {}".format(X_train.shape, y_train.shape))
    #     pipeline_after_selector.fit(X_train, y_train)
    #     logger.info("It takes {:.2f} seconds to refit the model.".format(time.time() - refit_init_time))
    #
    # # predict
    # data_test = X[X.index >= split_date]
    # X_test = data_test.as_matrix(x_columns)
    # logger.info("X_test: {}".format(X_test.shape))
    #
    # # predict
    # y_predict = pipeline_after_selector.predict(X_test)
    # X_copy = data_test[["y", "forward_y"]]
    # X_copy['predict_y'] = y_predict
    # logger.info("It takes {:.2f} seconds to predict.".format(time.time() - time_init))

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
        "engineer__lag": [10],  # [10, 20, 30, 40, 50, 60]
    }

    selector_param_grid = {
        "selector__k": [10],  # [10, 20, 30, 40, 50]
        "selector__select_method": ["hard"],  # ["hard", "soft", "all"]
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
    # logger.info("data: {}".format(data.head(5)))
    # logger.info("data value: {}".format(data.values))

    results = pd.DataFrame(columns=["model_id", "split_date", "model_name", "eval_metric",
                                    "update_date", "timestamp"])
    model_params_json_path = "./../results/model_history/model_params.json"
    failed_models = []
    # if os.path.exists(model_params_json_path):
    #     model_params_dict = json.loads(model_params_json_path)
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
            failed_models.append({
                "model_name": model_name
            })
            logger.info(str(e))
    logger.info("We have run {} models, with {} failed, using {:.2f} seconds".format(
        model_count + 1,
        len(failed_models),
        time.time() - time_init
    ))
    logger.info("\n\nList all the failed models:\n")
    for failed_model_i in failed_models:
        logger.info(
            "failed model_name: {}".format(
                failed_model_i["model_name"]
            )
        )

    # sort the results
    results.sort_values(["eval_metric"], ascending=[False], inplace=True)
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

    # save results into a csv
    results_path = os.path.join("./../results/model_history/",
                                "regression_results_" + str(look_ahead_day) + ".csv")
    if not os.path.exists(results_path):
        results.to_csv(results_path, encoding="utf-8", header=True, index=None)
    else:
        results.to_csv(results_path, encoding="utf-8", header=False, index=None, mode="a")

    # save model params
    # with open(model_params_json_path, 'w') as fp:
    #     json.dump(model_params_dict, fp)


if __name__ == "__main__":
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
    parser.add_argument("--save_k_best", help="if training, select k best models to save after training",
                        required=False, default=1, type=int)
    args = parser.parse_args()

    # global variable to hold hard thresholding results
    hard_thres_t_statistics = {}
    
    # see if there exists conflicts between split date and look_forward_days
    # add code here later, on 2017-12-26 17:27, by Zhao Yi, hopefully to be fixed by Xu Haonan

    # X_train = pd.read_excel("./../data/data_live/data_20171221.xls", encoding="utf-8", index_col="指标名称")
    # y_train = pd.read_excel("./../data/data_live/r007_20171221.xls", encoding="utf-8", index_col="指标名称")
    # logger.info(y_train.columns)
    # y_train.rename(columns={y_train.columns[0]: "y"}, inplace=True)
    # logger.info(y_train.columns)
    # data = pd.concat([X_train, y_train], axis=1)

    # import data
    # data = pd.read_excel("./../data/data_live/raw_data_20171222.xls", encoding="utf-8", index_col="指标名称")
    # data = pd.read_excel(args.data_path, encoding="utf-8", index_col="指标名称")
    data = GenerateDataFrame(
        raw_data_url=os.path.join("./../data/data_live/", args.data_path)
    ).data_to_dataframe()
    data.rename(columns={data.columns[-1]: "y"}, inplace=True)
    # data = data.loc[:data.shape[0]-2, :]
    # import data
    # data = GenerateDataFrame(
    #     raw_data_url="./../data/data_live/raw_data_20171222.xls",
    #     r007_url=None,
    #     warning_url=None
    # ).data_to_dataframe()
    data["forward_y"] = data["y"].shift((-1) * args.look_forward_days)
    split_date = datetime.datetime.strptime(args.split_date, "%Y%m%d").date()
    data.index = data.index.date
    data_train = data[data.index < split_date]
    # data_train.reset_index(drop=True, inplace=True)
    data_test = data.copy() # [data.index >= split_date]
    # data_test.reset_index(drop=True, inplace=True)
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
    results_path = os.path.join("./../results/model_history/",
                                "regression_results_" + str(args.look_forward_days) + ".csv")
    model_results = pd.read_csv(results_path, encoding="utf-8")
    # print(model_results)
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

    predict_results_dir = "./../results/predict/"
    predict_results_path = os.path.join(
        predict_results_dir,
        "regression_predict_step_" + str(args.look_forward_days) + ".csv"
    )
    predict_results = pd.DataFrame(columns=["y", "forward_y", "predict_y",
                                            "model_name", "model_id", "prediction_date", "timestamp"])

    for index_i in range(len(model_results.index)):
        try:
            predict_results_i = pd.DataFrame(columns=predict_results.columns)
            # print(data_test.index[:5])
            predict_results_i['date'] = data_test.index
            predict_results_i['date'] = pd.to_datetime(predict_results_i['date'])
            predict_results_i['date'] = predict_results_i['date'].dt.date
            predict_results_i["y"] = data_test["y"]
            predict_results_i["forward_y"] = data_test["forward_y"]
            predict_results_i["predict_y"] = test(
                x_test=x_test,
                model_id=model_results['model_id'][index_i]
            )
            logger.info("Model: {}, model_name: {}, metric: {}".format(
                model_results['model_id'][index_i],
                model_results['model_name'][index_i],
                model_results['eval_metric'][index_i]
            ))
            predict_results_i["model_name"] = model_results['model_name'][index_i]
            predict_results_i["model_id"] = model_results['model_id'][index_i]
            predict_results_i["prediction_date"] = datetime.date.today().strftime("%Y%m%d")
            predict_results_i["timestamp"] = int(time.time() * 1000)
            predict_results_i = predict_results_i[predict_results_i['date'] >= split_date]
            predict_results = predict_results.append(predict_results_i)
        except ValueError as e:
            logger.info("model {} does not exists".format(model_results['model_id'][index_i]))
            logger.info(str(e))

    if not os.path.exists(predict_results_dir):
        os.makedirs(predict_results_dir)
    if not os.path.exists(predict_results_path):
        predict_results.to_csv(predict_results_path, encoding="utf-8", index=False, mode="a", header=True)
    else:
        predict_results.to_csv(predict_results_path, encoding="utf-8", index=False, mode="a", header=False)

    logger.info("Prediction finished!!!")
    print("Done! This will be removed later on.")

