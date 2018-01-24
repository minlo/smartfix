from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from xgboost import XGBClassifier


imputer_param_grid = {
    "imputer__method": ["directly"]
}
engineer_param_grid = {
    "engineer__lag": [10, 30, 50]
}

selector_param_grid = {
    "selector__k": [10, 30, 50, -1],
    "selector__select_method": ["hard"],  # ["hard", "soft"]
}
# temporarily not used
reducer_param_grid = {
    "reducer__n_components": [10]
}
model_dict = {
    "random_forest": RandomForestClassifier(n_estimators=1000, n_jobs=-1, random_state=1234),
    "xgboost": XGBClassifier(),
    "logistic_regression": LogisticRegression(penalty="l1"),
    "svm": SVC(probability=True)
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
    "svm": {
        "model__C": [0.01, 0.1, 1, 10, 100, 1000],
        "model__kernel": ["linear", "poly", "rbf", "sigmoid"],
        "model__epsilon": [x / 100 for x in range(10, 50, 5)],
        "model_gamma": ["auto"]
    }
}
model_pipeline_mode_dict = {
    "random_forest": "grid",
    "xgboost": "random",
    "logistic_regression": "single",
    "svm": "random"
}

