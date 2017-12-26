#!/bin/bash


# run grid search, ahead_step=1
/opt/anaconda3/bin/python train_test_utils.py --split_date=20171218 --look_forward_days=1 --validation_period_length=30 --save_k_best=2 --is_training=true


echo "done"
