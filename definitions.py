"""
To define various variables for later use.
"""
import os


ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_SAVE_DIR = os.path.join(ROOT_DIR, "results/models/")
MODEL_PARAMS_JSON_DIR = os.path.join(ROOT_DIR, "results/model_history/regression_model_params/")
MODEL_TRAINING_EVAL_RESULTS_DIR = os.path.join(ROOT_DIR, "results/model_history/")
MODEL_TRAINING_FAILED_MODELS_DIR = os.path.join(ROOT_DIR, "results/model_history/")
BEST_MODEL_HISTORY_DIR = os.path.join(ROOT_DIR, "results/model_history/")
PREDICTION_RESULTS_ALL_MODELS_DIR = os.path.join(ROOT_DIR, "results/predict/")
PREDICTION_RESULTS_BEST_MODEL_DIR = os.path.join(ROOT_DIR, "results/predict/")
DATA_DIR = os.path.join(ROOT_DIR, "data/data_live/")


