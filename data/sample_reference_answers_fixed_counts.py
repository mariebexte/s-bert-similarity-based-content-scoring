import pandas as pd
import os
from sbert_util import get_answer_pairs
import argparse


def main(argv):

    asap_path = argv.data_path
    target_path = argv.target_path
    num_ans_per_score = int(argv.num_ref_answers_per_score)
    num_val = int(argv.num_val)

    random_state = 468

    if not os.path.exists(target_path):
        os.mkdir(target_path)

    # For every prompt in the ASAP data
    for i in range(1, 11):

        df = pd.read_csv(os.path.join(asap_path, str(i) + argv.data_name))

        # Split away validation data
        df_val_raw = df.sample(num_val, random_state=random_state)
        df_train_raw = df.drop(df_val_raw.index)

        # Save raw answer files
        df_val_raw.to_csv(os.path.join(target_path, str(i) + "_FULL_val_raw.csv"), index=None)
        df_train_raw.to_csv(os.path.join(target_path, str(i) + "_FULL_train_raw.csv"), index=None)

        # Write training data: pair all answers within
        train_df = get_answer_pairs(df_train_raw, df_train_raw, num_ans_per_score, random_state)
        train_df.to_csv(os.path.join(target_path, str(i) + "_FULL_train.csv"), index=None)

        # Write val data: pair all answers within
        val_df = get_answer_pairs(df_val_raw, df_val_raw, num_ans_per_score, random_state)
        val_df.to_csv(os.path.join(target_path, str(i) + "_FULL_val.csv"), index=None)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--target_path", help="name of folder for resulting data")
    parser.add_argument("--data_path", help="location of gold training and testing files")
    parser.add_argument("--data_name", help="filenames of gold training data")
    parser.add_argument("--num_ref_answers_per_score", help="how many answers of each score to pair an answer with")
    parser.add_argument("--num_val", help="how many answers to reserve for validation")
    args = parser.parse_args()
    main(args)
