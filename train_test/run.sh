#!/bin/bash


# run grid search, ahead_step=1
/opt/anaconda3/bin/python train_test_utils.py --split_date=20171222 --look_forward_days=1 --validation_period_length=60 --save_k_best=1 --is_training=true --data_path=raw_data_20171222.xls &
/opt/anaconda3/bin/python train_test_utils.py --split_date=20171221 --look_forward_days=2 --validation_period_length=60 --save_k_best=1 --is_training=true --data_path=raw_data_20171222.xls &
/opt/anaconda3/bin/python train_test_utils.py --split_date=20171220 --look_forward_days=3 --validation_period_length=60 --save_k_best=1 --is_training=true --data_path=raw_data_20171222.xls &
/opt/anaconda3/bin/python train_test_utils.py --split_date=20171219 --look_forward_days=4 --validation_period_length=60 --save_k_best=1 --is_training=true --data_path=raw_data_20171222.xls &
/opt/anaconda3/bin/python train_test_utils.py --split_date=20171218 --look_forward_days=5 --validation_period_length=60 --save_k_best=1 --is_training=true --data_path=raw_data_20171222.xls &
/opt/anaconda3/bin/python train_test_utils.py --split_date=20171215 --look_forward_days=6 --validation_period_length=60 --save_k_best=1 --is_training=true --data_path=raw_data_20171222.xls &
/opt/anaconda3/bin/python train_test_utils.py --split_date=20171214 --look_forward_days=7 --validation_period_length=60 --save_k_best=1 --is_training=true --data_path=raw_data_20171222.xls &
/opt/anaconda3/bin/python train_test_utils.py --split_date=20171213 --look_forward_days=8 --validation_period_length=60 --save_k_best=1 --is_training=true --data_path=raw_data_20171222.xls &

wait

echo "done"
