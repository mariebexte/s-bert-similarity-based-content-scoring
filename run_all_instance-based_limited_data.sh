#!/bin/sh

echo "Running instance-based: Limited data setting; Prompt 1"
bash run_instance_based.sh instance-based/limited_data_60 data/limited_data_60/1_60_train_val_raw.csv data/gold_train_test_split/1_GOLD_test.csv
echo "Running instance-based: Limited data setting; Prompt 2"
bash run_instance_based.sh instance-based/limited_data_60 data/limited_data_60/2_60_train_val_raw.csv data/gold_train_test_split/2_GOLD_test.csv
echo "Running instance-based: Limited data setting; Prompt 3"
bash run_instance_based.sh instance-based/limited_data_60 data/limited_data_60/3_60_train_val_raw.csv data/gold_train_test_split/3_GOLD_test.csv
echo "Running instance-based: Limited data setting; Prompt 4"
bash run_instance_based.sh instance-based/limited_data_60 data/limited_data_60/4_60_train_val_raw.csv data/gold_train_test_split/4_GOLD_test.csv
echo "Running instance-based: Limited data setting; Prompt 5"
bash run_instance_based.sh instance-based/limited_data_60 data/limited_data_60/5_60_train_val_raw.csv data/gold_train_test_split/5_GOLD_test.csv
echo "Running instance-based: Limited data setting; Prompt 6"
bash run_instance_based.sh instance-based/limited_data_60 data/limited_data_60/6_60_train_val_raw.csv data/gold_train_test_split/6_GOLD_test.csv
echo "Running instance-based: Limited data setting; Prompt 7"
bash run_instance_based.sh instance-based/limited_data_60 data/limited_data_60/7_60_train_val_raw.csv data/gold_train_test_split/7_GOLD_test.csv
echo "Running instance-based: Limited data setting; Prompt 8"
bash run_instance_based.sh instance-based/limited_data_60 data/limited_data_60/8_60_train_val_raw.csv data/gold_train_test_split/8_GOLD_test.csv
echo "Running instance-based: Limited data setting; Prompt 9"
bash run_instance_based.sh instance-based/limited_data_60 data/limited_data_60/9_60_train_val_raw.csv data/gold_train_test_split/9_GOLD_test.csv
echo "Running instance-based: Limited data setting; Prompt 10"
bash run_instance_based.sh instance-based/limited_data_60 data/limited_data_60/10_60_train_val_raw.csv data/gold_train_test_split/10_GOLD_test.csv