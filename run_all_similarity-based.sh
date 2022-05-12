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

echo "Running similarity-based: Pretrained, limited data"
python3 similarity-based/sbert_similarity.py --use_pretrained --train_condition pretrained --test_condition limited_test --results_folder similarity-based/results --num_epochs 0 --eval_steps 0  --train_path data/limited_data_60 --train_name _60_train.csv --raw_train_name _60_train_raw.csv --val_name _60_val.csv --raw_val_name _60_val_raw.csv --test_path data/gold_train_test_split --raw_test_name _GOLD_test.csv

echo "Running similarity-based: Pretrained, full data"
python3 similarity-based/sbert_similarity.py --use_pretrained --train_condition pretrained --test_condition full_test --results_folder similarity-based/results --num_epochs 0 --eval_steps 0  --train_path data/full_data --train_name _FULL_train.csv --raw_train_name _FULL_train_raw.csv --val_name _FULL_val.csv --raw_val_name _FULL_val_raw.csv --test_path data/gold_train_test_split --raw_test_name _GOLD_test.csv

echo "Running similarity-based: Finetune using limited data, test with limited data"
python3 similarity-based/sbert_similarity.py --train_condition finetuned_train-limited --test_condition limited_test --results_folder similarity-based/results --num_epochs 15 --eval_steps 5  --train_path data/limited_data_60 --train_name _60_train.csv --raw_train_name _60_train_raw.csv --val_name _60_val.csv --raw_val_name _60_val_raw.csv --test_path data/gold_train_test_split --raw_test_name _GOLD_test.csv

echo "Running similarity-based: Evaluate finetuned using limited data, test with full data"
python3 similarity-based/sbert_similarity.py --eval_finetuned --train_condition finetuned_train-limited --test_condition full_test --results_folder similarity-based/results --num_epochs 0 --eval_steps 0  --train_path data/full_data --train_name _FULL_train.csv --raw_train_name _FULL_train_raw.csv --val_name _FULL_val.csv --raw_val_name _FULL_val_raw.csv --test_path data/gold_train_test_split --raw_test_name _GOLD_test.csv

echo "Running similarity-based: Finetune using full data, test with full data"
python3 similarity-based/sbert_similarity.py --train_condition finetuned_train-full --test_condition full_test --results_folder similarity-based/results --num_epochs 30 --eval_steps 500  --train_path data/full_data --train_name _FULL_train.csv --raw_train_name _FULL_train_raw.csv --val_name _FULL_val.csv --raw_val_name _FULL_val_raw.csv --test_path data/gold_train_test_split --raw_test_name _GOLD_test.csv

echo "Running similarity-based: Evalute finetuned using full data, test with limited data"
python3 similarity-based/sbert_similarity.py --eval_finetuned --train_condition finetuned_train-full --test_condition limited_test --results_folder similarity-based/results --num_epochs 0 --eval_steps 0  --train_path data/limited_data_60 --train_name _60_train.csv --raw_train_name _60_train_raw.csv --val_name _60_val.csv --raw_val_name _60_val_raw.csv --test_path data/gold_train_test_split --raw_test_name _GOLD_test.csv
