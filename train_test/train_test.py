import os
import pandas as pd
import numpy as np
import time
import datetime
import sys
import operator
import argparse
import matplotlib.pyplot as plt
sys.path.append('./../')


from feature_engineering import GenerateNDiffFeatures
from data_processing import DataFrameToMatrix, ImputeNumericDataFrame, Fluctuation

from pipeline_lib import TrainTestForTimeSeries, GeneratePipeline
from sklearn.linear_model import Lasso, Ridge, ElasticNet, least_angle, LassoLars, LinearRegression

from sklearn.feature_selection import chi2, SelectKBest, VarianceThreshold, SelectFromModel
from sklearn.svm import LinearSVC, SVR, SVC
from sklearn.ensemble import ExtraTreesRegressor, RandomForestClassifier, RandomForestRegressor
from sklearn.preprocessing import MinMaxScaler, StandardScaler, Imputer
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.feature_selection import SelectFromModel

# from scipy import stats
import statsmodels.api as sm
# from xgboost import XGBRegressor, XGBClassifier


def regression_t_statistic(data, y_column, ctrl_columns, x_column):
    """
    Regress y_column on x_column given control_column_list.
    """
    column_x_list = ctrl_columns.copy()
    column_x_list.append(x_column)

    # X = data.as_matrix(column_x_list)
    # y = data.as_matrix([y_column])
    # y = np.ravel(y)
    X = data[column_x_list]
    y = data[[y_column]]

    X2 = sm.add_constant(X)
    est = sm.OLS(y, X2)
    est2 = est.fit()
    return est2.summary().tables[1].data[-1][3]


def test_accuracy(y_predict, y_test):
    count_10 = 0
    count_15 = 0
    count_20 = 0
    count_30 = 0
    for i in range(len(y_predict)):
        error_i = abs(y_predict[i] - y_test[i]) / y_test[i]
        if error_i <= 0.1:
            count_10 += 1
        if error_i <= 0.15:
            count_15 += 1
        if error_i <= 0.2:
            count_20 += 1
        if error_i <= 0.3:
            count_30 += 1

    error_10 = count_10 / len(y_predict)
    error_15 = count_15 / len(y_predict)
    error_20 = count_20 / len(y_predict)
    error_30 = count_30 / len(y_predict)

    print("There are total {} instances in test set.".format(y_predict.shape[0]))
    print("Error within 10% percent: {:.2f}%".format(error_10 * 100))
    print("Error within 15% percent: {:.2f}%".format(error_15 * 100))
    print("Error within 20% percent: {:.2f}%".format(error_20 * 100))
    print("Error within 30% percent: {:.2f}%".format(error_30 * 100))

    return error_10, error_15, error_20, error_30


def train_test_model_pipeline(data, test_year, split_date, target_column, response_column, imputer, feature_engineer, scaler,
                              selector, model, pipeline_mode="single", param_grid=None, n_iter=5, verbose=1,
                              selected_features_list=None, look_forward_days=1):
    """
    Train and test a single model pipeline.
    """
    data_train, data_test = TrainTestForTimeSeries(
        data.copy(),
        test_year,
        Pipeline([
            ('feature_imputing', imputer),
            ('feature_engineering', feature_engineer)
        ]),
        split_date=split_date
    ).train_split_pipeline_diff_features()
    if selected_features_list is not None:
        selected_features_list_y = selected_features_list.copy()
        selected_features_list_y.extend([target_column, response_column, 'date'])
        data_train = data_train[selected_features_list_y]
        data_test = data_test[selected_features_list_y]

    model_pipeline = GeneratePipeline(scaler, selector, model, pipeline_mode, param_grid, n_iter, verbose,
                                      scoring_criterion="neg_mean_squared_error").get_pipeline()

    convert_to_numeric_pipeline = Pipeline([('convert_to_numeric', DataFrameToMatrix([response_column, 'date']))])

    X_train = convert_to_numeric_pipeline.fit_transform(data_train)
    y_train = data_train.as_matrix([response_column])
    y_train = np.ravel(y_train)

    X_test = convert_to_numeric_pipeline.transform(data_test)
    y_test = data_test.as_matrix([target_column])
    y_test = np.ravel(y_test)

    # fit
    model_pipeline.fit(X_train, y_train)

    # predict
    y_predict = model_pipeline.predict(X_test)

    # calculate fluctuation
    data_test_save = data_test[['date', target_column]]
    data_test_save['look_forward_days'] = look_forward_days
    data_test_save['predict'] = y_predict
    data_test_save.reset_index(drop=True, inplace=True)

    return data_test_save


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--impute_way", help="choose the way to impute", required=False, default="directly", type=str)
    parser.add_argument("--look_forward_days", help="look_forward_days", required=False, default=1, type=int)
    args = parser.parse_args()    

    data = pd.read_csv('./../data/data_live/data_live_1222_imputated_' + args.impute_way + '.csv', encoding='utf-8')

    # delete column x_59 and warning
    del data['x59'], data['warning']

    # rename response column
    data.rename(columns={'R007': 'y'}, inplace=True)

    # processing
    # imputer
    imputer_dataframe = ImputeNumericDataFrame(
        list(data.columns),
        convert_type="numeric",
        numeric_impute_method="mean",
        immune_columns=['date']
    )

    # scaler
    scaler_minmax = MinMaxScaler()
    scaler_standard = StandardScaler()

    # feature engineering
    # t + 1, for hard thresholding
    generate_features_t_1_0_order_diff = GenerateNDiffFeatures(
        target_column='y',
        look_forward_days=args.look_forward_days,
        look_backward_days=60,
        diff_order=0,
        addition_time_features=True,
        date_column="date"
    )

    # hard thresholding
    data_train, data_test = TrainTestForTimeSeries(
        data.copy(),
        2018,
        Pipeline([
            ('feature_imputing', imputer_dataframe),
            ('feature_engineering', generate_features_t_1_0_order_diff)
        ])
    ).train_split_pipeline_diff_features_by_year()

    # generate columns
    hard_thres_test_columns = []
    hard_thres_ctrl_columns = []

    for column_i in list(data_train.columns):
        if column_i in ['y', 'date'] or 'forward' in column_i:
            continue
        if 'y_back_0_order_diff' in column_i:
            hard_thres_ctrl_columns.append(column_i)
        else:
            hard_thres_test_columns.append(column_i)
    print("There are in total {} columns, while we only need {} columns".format(
        len(hard_thres_test_columns),
        len(hard_thres_ctrl_columns))
    )

    # run regressions and extract the t-statistics value
    hard_thres_test_t_stats = {}
    for column_i in hard_thres_test_columns:
        t_i = regression_t_statistic(data_train, 'y', hard_thres_ctrl_columns, column_i)
        hard_thres_test_t_stats[column_i] = abs(float(t_i))

    sorted_hard_thres_test_t_stats = sorted(hard_thres_test_t_stats.items(), key=operator.itemgetter(1), reverse=True)


    def selected_features_k_hard(feature_k, sorted_hard_thres_test_t_stats=sorted_hard_thres_test_t_stats):
        selected_features_k_hard_inside = []
        for i in sorted_hard_thres_test_t_stats[:feature_k]:
            selected_features_k_hard_inside.append(i[0])
        print(selected_features_k_hard_inside)
        print(len(selected_features_k_hard_inside))

        return selected_features_k_hard_inside

    # select top-k features using hard thresholding
    selected_features_10_hard = selected_features_k_hard(10)
    selected_features_20_hard = selected_features_k_hard(20)
    selected_features_30_hard = selected_features_k_hard(30)
    selected_features_40_hard = selected_features_k_hard(40)
    selected_features_50_hard = selected_features_k_hard(50)

    # soft thresholding
    selector_lasso = SelectFromModel(Lasso(alpha=0.1), prefit=False)

    # train and predict
    print("data shape: {}".format(data.shape))
    data_predict = train_test_model_pipeline(
        data=data.copy(),
        test_year=2017,
        split_date=datetime.date(2017, 12, 22) - datetime.timedelta(days=args.look_forward_days - 1),
        target_column="y",
        response_column="y_forward_" + str(args.look_forward_days),
        imputer=imputer_dataframe,
        feature_engineer=GenerateNDiffFeatures(
            target_column="y",
            look_forward_days=args.look_forward_days,
            look_backward_days=20,
            diff_order=0,
            addition_time_features=True,
            date_column="date"
        ),
        scaler=scaler_minmax,
        selector=SelectFromModel(Lasso(alpha=0.0001), prefit=False),  # SelectKBest(k="all"),
        model=RandomForestRegressor(n_estimators=1000, n_jobs=-1, random_state=1234),
        pipeline_mode="single",
        param_grid=None,
        selected_features_list=None,
        look_forward_days=args.look_forward_days
    )
    print(data_predict)

    # save results
    data_predict.to_csv("./../results/data_predict_" + str(args.look_forward_days) + "_days_" + datetime.date.today().strftime("%Y%m%d") + ".csv",
                        encoding="utf-8",
                        index=None,
                        header=True)

    print("done")

