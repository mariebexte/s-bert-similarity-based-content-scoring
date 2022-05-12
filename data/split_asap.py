import pandas as pd
import os
import argparse


def main(argv):

    target_folder = argv.target_path
    file = argv.asap_path
    test_percentage = 0.1

    if not os.path.exists(target_folder):
        os.mkdir(target_folder)

    asap_df = pd.read_csv(file, sep="\t")

    # For every prompt in the asap data
    for i in range(1, 11):

        # Take only the answers to the current prompt, select desired column order
        prompt_df = asap_df[asap_df["EssaySet"] == i]
        prompt_df = prompt_df[["Id", "EssayText", "Score1", "Score2"]]

        # Split away testing data
        gold_test = prompt_df.sample(int(test_percentage*len(prompt_df)), random_state=468)
        gold_train = prompt_df.drop(gold_test.index)

        gold_test.to_csv(os.path.join(target_folder, str(i)+"_GOLD_test.csv"), index=None)
        gold_train.to_csv(os.path.join(target_folder, str(i)+"_GOLD_train.csv"), index=None)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--target_path", "-c", help="name of folder for resulting data split")
    parser.add_argument("--asap_path", "-p", help="location of asap csv file")
    args = parser.parse_args()
    main(args)