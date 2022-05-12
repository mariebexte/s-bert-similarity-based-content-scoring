#!/bin/sh

venv_dir=venv

if [ ! -d $venv_dir ]; then
  echo "Virtual environment not found. Setting up virtual environment under $venv_dir"
  python3 -m venv $venv_dir
  source $venv_dir/bin/activate
  pip install -r requirements.txt
  pip install -U sentence-transformers
fi

source $venv_dir/bin/activate

echo "Preparing data: Gold ASAP splits"
python3 data/split_asap.py --target_path data/gold_train_test_split --asap_path data/asap.tsv

echo "Preparing data: Limited data setting"
python3 data/sample_reference_answers_overall.py --target_path data/limited_data_60 --data_path data/gold_train_test_split --data_name _GOLD_train.csv --num_ref_answers 60 --val_proportion 0.2

echo "Preparing data: Full data setting"
python3 data/sample_reference_answers_fixed_counts.py --target_path data/full_data --data_path data/gold_train_test_split --data_name _GOLD_train.csv --num_ref_answers_per_score 10 --num_val 100