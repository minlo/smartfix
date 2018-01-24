"""
Predict repurchasing interest rate in production
================================================

This package aims to predict repurchasing interest rate in production environment. It is expected that this
implementation is both accurate and robust.

See subdirectory docs for complete documentation.
"""
import sys
import os
import logging


# setting logging configuration
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


__all__ = ["data_processing", "evaluation", "feature_engineering", "feature_selecting",
           "models", "pipeline_lib", "train_test", "config"]

