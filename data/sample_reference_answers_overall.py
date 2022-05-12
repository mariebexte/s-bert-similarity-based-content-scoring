import pandas as pd
import os
from sbert_util import get_all_possible_answer_pairs
import argparse

# Only samples answers where score1 == score2


def main(argv):

    target_path = argv.target_path
    num_ref_answers = int(argv.num_ref_answers)
    val_percentage = float(argv.val_proportion)

    random_state = 468

    if not os.path.exists(target_path):
        os.mkdir(target_path)

    # For every prompt in the ASAP data
    for i in range(1, 11):

        df = pd.read_csv(os.path.join(argv.data_path, str(i) + argv.data_name))

        # Only take from answers where the two annotators agreed upon a score
        candidates = df[df["Score1"] == df["Score2"]]
        df = df.drop(candidates.index)

        # See how many answers remain to choose from
        #for i in set(df["Score1"]):
            #print(i, len(df[df["Score1"] == i]))

        # Determine how many answers to sample per class
        labels = set(candidates["Score1"])
        num_answers_per_label_train = int(((1-val_percentage)*num_ref_answers)/len(labels))
        num_answers_per_label_val = int((val_percentage * num_ref_answers)/len(labels))

        df_reference_answers_train = pd.DataFrame(columns=df.columns)
        df_reference_answers_val = pd.DataFrame(columns=df.columns)

        # For each label: sample required amount
        for label in labels:

            # Sample Train
            df_reference_answers_train_temp = candidates[candidates["Score1"] == label].sample(num_answers_per_label_train, random_state=random_state)
            # Append Train
            df_reference_answers_train = df_reference_answers_train.append(df_reference_answers_train_temp)
            # Remove from data pool
            candidates = candidates.drop(df_reference_answers_train_temp.index)

            # Sample Val
            df_reference_answers_val_temp = candidates[candidates["Score1"] == label].sample(num_answers_per_label_val, random_state=random_state)
            # Append Val
            df_reference_answers_val = df_reference_answers_val.append(df_reference_answers_val_temp)
            # Remove from data pool
            candidates = candidates.drop(df_reference_answers_val_temp.index)

        # Return remaining candidate data into overall data pool
        df = df.append(candidates)

        # Save raw answer files
        df_reference_answers_train.to_csv(os.path.join(target_path, str(i) + "_"+str(num_ref_answers)+"_train_raw.csv"), index=None)
        df_reference_answers_val.to_csv(os.path.join(target_path, str(i) + "_"+str(num_ref_answers)+"_val_raw.csv"), index=None)
        df.to_csv(os.path.join(target_path, str(i) + "_"+str(num_ref_answers)+"_rest_raw.csv"), index=None)

        df_reference_answers_total = df_reference_answers_train.copy()
        df_reference_answers_total = df_reference_answers_total.append(df_reference_answers_val)
        df_reference_answers_total.to_csv(os.path.join(target_path, str(i) + "_"+str(num_ref_answers)+"_train_val_raw.csv"), index=None)

        # Write training data: pair all answers within
        train_df = get_all_possible_answer_pairs(df_reference_answers_train, df_reference_answers_train)
        train_df.to_csv(os.path.join(target_path, str(i) + "_"+str(num_ref_answers)+"_train.csv"), index=None)

        # Write val data: pair all answers within
        val_df = get_all_possible_answer_pairs(df_reference_answers_val, df_reference_answers_val)
        val_df.to_csv(os.path.join(target_path, str(i) + "_"+str(num_ref_answers)+"_val.csv"), index=None)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--target_path", help="name of folder for resulting data")
    parser.add_argument("--data_path", help="location of gold training and testing files")
    parser.add_argument("--data_name", help="filenames of gold training data")
    parser.add_argument('--num_ref_answers', help="how many reference answers to sample")
    parser.add_argument('--val_proportion', help="which proportion of reference answers to use for validation")
    args = parser.parse_args()
    main(args)
