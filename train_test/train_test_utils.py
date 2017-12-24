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
from feature_selecting import HardThresholdSelector, SoftThresholdSelector
from data_processing import ImputationMethod, GenerateDataFrame

from sklearn.externals import joblib
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV, TimeSeriesSplit
from sklearn.linear_model import Lasso, Ridge, ElasticNet, least_angle, LassoLars, LinearRegression
from sklearn.feature_selection import chi2, SelectKBest, VarianceThreshold, SelectFromModel
from sklearn.svm import LinearSVC, SVR, SVC
from sklearn.ensemble import ExtraTreesRegressor, RandomForestClassifier, RandomForestRegressor
from sklearn.preprocessing import MinMaxScaler, StandardScaler, Imputer
from sklearn.decomposition import PCA
from xgboost import XGBRegressor, XGBClassifier
import logging


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


def train(imputer, engineer, selector, scaler, reducer, model, X, split_date,
          pipeline_mode, pipeline_param_grid):
    """Train historical data and save the model into pickle file."""
    pipeline = BuildPipeline(
        imputer=imputer,
        engineer=engineer,
        selector=selector,
        scaler=scaler,
        reducer=reducer,
        model=model,
        split_date=split_date
    )
    pipeline_1 = pipeline.build_before_selector()
    pipeline_2 = pipeline.build_after_selector()

    time_init = time.time()
    X = pipeline_1.fit_transform(X)
    # logger.info("X head after the feature engineering: {}".format(X.head()))

    # delete all values after split_date
    X.replace([np.inf, -np.inf], np.nan, inplace=True)
    X = X.dropna()
    # logger.info("X head after dropping na: {}".format(X.head()))
    X.index = pd.to_datetime(X.index)
    X.index = X.index.date
    X = X[X.index < split_date]

    # logger.info("X head: {}".format(X.head()))
    # maybe change to X = X.drop_na()
    x_columns = list(X.columns)
    x_columns.remove('forward_y')
    logger.info("x_columns: {}, unique: {}".format(len(x_columns), len(set(x_columns))))
    x_columns_set = []
    for index_i in range(len(x_columns)):
        if x_columns[index_i] not in x_columns_set:
            x_columns_set.append(x_columns[index_i])
        else:
            logger.info("duplicate column: {}".format(x_columns[index_i]))
            break

    X_train = X.as_matrix(x_columns)
    y_train = X.as_matrix(['forward_y'])
    y_train = np.ravel(y_train)
    logger.info("X_train: {}, y: {}".format(X_train.shape, y_train.shape))
    logger.info(np.where(X.values >= np.finfo(np.float64).max))
    logger.info(X.ix[339, 462])

    pipeline_2 = generate_grid_search(
        search_pipeline=pipeline_2,
        pipeline_mode=pipeline_mode,
        param_grid=pipeline_param_grid
    )

    pipeline_2.fit(X_train, y_train)
    logger.info("It takes {:.2f} seconds to train this model.".format(time.time() - time_init))
    if pipeline_mode != "single":
        pipeline_2 = pipeline_2.best_estimator_

    pipeline_combined = Pipeline([
        ("pipeline_before_selector", pipeline_1),
        ("pipeline_after_selector", pipeline_2)
    ])
    return pipeline_combined


def save_pipeline(pipeline_combined, model_id):
    model_save_path = os.path.join("../results/models/", "model_" + model_id + ".pkl")
    joblib.dump(pipeline_combined, model_save_path)
    logger.info("Model dumped into {}.".format(model_save_path))


def test(X, y, split_date, model_id=""):
    model_load_path = os.path.join("../results/models/", "model_" + model_id + ".pkl")
    if not os.path.exists(model_load_path):
        raise ValueError("model_id {} does not exist.".format(model_id))

    pipeline_combined = joblib.load(model_load_path)

    # before selector
    pipeline_before_selector = pipeline_combined.named_steps["pipeline_before_selector"]
    pipeline_after_selector = pipeline_combined.name_steps["pipeline_after_selector"]

    X = pipeline_before_selector.transform(X)

    # split X, y
    X = X[X.index >= split_date]

    # predict
    y_predict = pipeline_after_selector.predict(X)

    return y_predict


def evaluate(y_test_true, y_test_predict):
    return 1.0


def search_regression_ml(save_k_best, look_ahead_day, split_date):
    imputer_param_grid = {
        "imputer__method": ["directly", "quadratic", "slinear", "cubic"]
    }
    # Actually, we would not do grid search on this part right now
    engineer_param_grid = {
        "engineer__lag": [10],  # [10, 20, 30, 40, 50, 60]
    }
    selector_dict = {
        # "soft_selector": SoftThresholdSelector(),
        # "hard_selector": HardThresholdSelector(),
        "all_selector": SelectKBest(k="all")
    }
    hard_selector_param_grid = {
        "selector__k": [10],  # [10, 20, 30, 40, 50]
    }
    # temporarily not used
    reducer_param_grid = {
        "reducer__n_components": [10, 20, 30, 40]
    }
    model_dict = {
        "random_forest": RandomForestRegressor(n_estimators=1000, n_jobs=-1, random_state=1234),
        # "xgboost": XGBRegressor(),
        # "lasso": Lasso(alpha=0.01, random_state=1234),
    }
    model_param_grid_dict = {
        "random_forest": {
            "model__n_estimators": [500]  # [100, 500, 1000]
        },
        # "xgboost": {
        #     "model__max_depth": range(2, 12, 2),
        #     "model__min_child_weight": range(2, 10, 2),
        #     "model__subsample": [i / 10.0 for i in range(6, 10)],
        #     "model__colsample_bytree": [0, 0.1, 0.2, 0.3, 0.4, 0.5],
        #     "model__learning_rate": [0.01, 0.1]
        # },
        # "lasso": {
        #     "model__alpha": [1 / (10**x) for x in range(2, 10)]
        # }
    }
    model_pipeline_mode_dict = {
        "random_forest": "grid",
        # "xgboost": "random",
        # "lasso": "random"
    }

    X_train = pd.read_excel("./../data/data_live/data_20171221.xls", encoding="utf-8", index_col="指标名称")
    y_train = pd.read_excel("./../data/data_live/r007_20171221.xls", encoding="utf-8", index_col="指标名称")
    # logger.info(y_train.columns)
    y_train.rename(columns={y_train.columns[0]: "y"}, inplace=True)
    logger.info(y_train.columns)
    data = pd.concat([X_train, y_train], axis=1)
    # logger.info("data: {}".format(data.head(5)))
    # logger.info("data value: {}".format(data.values))

    results = pd.DataFrame(columns=["model_id", "split_date", "model_name", "impute_method",
                                    "model_selector", "eval_metric", "model_params",
                                    "updating_date", "timestamp"])

    model_count = 0
    time_init = time.time()
    save_model_dict = {}
    logger.info("Tuning models, {}: ".format(datetime.date.today().strftime("%Y%m%d")))
    for impute_method in ["directly", "slinear", "cubic", "zero"]:
        for model_selector in selector_dict.keys():
            for model_name in model_dict.keys():
                time_start = time.time()
                pipeline_param_grid = model_param_grid_dict[model_name].copy()
                # pipeline_param_grid.update(engineer_param_grid)
                if model_selector == "hard_selector":
                    pipeline_param_grid.update(hard_selector_param_grid)
                pipeline = train(
                    imputer=ImputationMethod(method=impute_method),
                    engineer=FeatureExtract(lag=10, look_forward_days=look_ahead_day),
                    selector=selector_dict[model_selector],
                    scaler=MinMaxScaler(),
                    reducer=PCA(n_components=10),  # temporarily not in use
                    model=model_dict[model_name],
                    X=data.copy(),
                    split_date=split_date,
                    pipeline_mode=model_pipeline_mode_dict[model_name],
                    pipeline_param_grid=pipeline_param_grid
                )

                # y_test_predict = pipeline.predict(X_train)

                eval_metric = 1.0# eval_metric = evaluate(y_train, y_test_predict)

                model_id = str(uuid.uuid4())
                results.loc[len(results.index)] = [model_id, split_date,
                                                   model_name, impute_method, model_selector,
                                                   eval_metric, pipeline.get_params(),
                                                   datetime.date.today().strftime("%Y%m%d"),
                                                   int(1000 * time.time())]
                save_model_dict[model_id] = pipeline

                model_count += 1
                logger.info(
                    "Model {}, "
                    "model_id: {},"
                    "model_name: {}, "
                    "impute_method: {}, "
                    "model_selector: {}, "
                    "using {:.2f} seconds".format(
                        model_count,
                        model_id,
                        model_name,
                        impute_method,
                        model_selector,
                        time.time() - time_start
                    )
                )
    logger.info("We have run {} models, using {:.2f} seconds".format(
        model_count + 1,
        time.time() - time_init
    ))

    # sort the results
    results.sort_values(["eval_metric"], ascending=[False], inplace=True)
    results.reset_index(drop=True, inplace=True)

    results = results.loc[:save_k_best]

    for index_i in range(len(results.index)):
        model_id_i = results['model_id'][index_i]
        logger.info("Saving {} to disk...".format(model_id_i))
        save_pipeline(save_model_dict[model_id_i], model_id_i)

    # save results into a csv
    results_path = os.path.join("./../results/model_history/",
                                "regression_results_" + str(look_ahead_day) + ".csv")
    results.to_csv(results_path, encoding="utf-8", header=True, index=None)


if __name__ == "__main__":
    search_regression_ml(5, 1, datetime.date(2017, 12, 24))

