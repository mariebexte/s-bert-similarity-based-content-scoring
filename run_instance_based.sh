#!/bin/bash

venv_dir=venv

if [ ! -d $venv_dir ]; then
  echo "Virtual environment not found. Setting up virtual environment under $venv_dir"
  python3 -m venv $venv_dir
  source $venv_dir/bin/activate
  pip install -r requirements.txt
  pip install -U sentence-transformers
fi

if [ -z "$1" ]; then
  echo "
  usage: bash run.sh train_csv_path [test_csv_path]

  parameters:
    train_csv_path: Path to a csv file to be used for training. The csv file at must start with an id column, followed
                    by a column containing the input sentences. Each following column in interpreted as a different
                    target label.
    test_csv_path:  Path to a csv file to be used for testing. The csv file at must have the same format as the train
                    csv. If omitted, a cross-validation is run on the targets."

else
  source $venv_dir/bin/activate
  export PYTHONPATH=$PYTHONPATH:./
  if [ -z "$3" ]; then
    echo "This code was only optimized for running it with training and test set."
    #echo "running cross validation with data from $2."
    #python instance-based/train_classical.py $1 $2
    #python instance-based/train_bert.py $1 $2
  else
    echo "Running train test with data from $2 as training set and $3 as test set."
    python instance-based/train_classical.py $1 $2 $3
    python instance-based/train_bert.py $1 $2 $3
  fi
fi

#echo "calculating means."
#python instance-based/calculate_means.py $(dirname $1)/results
