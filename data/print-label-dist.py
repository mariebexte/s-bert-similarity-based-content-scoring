import pandas as pd
import os

asap_path = "gold_train_test_split"

for i in range(1, 11):

    train_df = pd.read_csv(os.path.join(asap_path, str(i) + "_GOLD_train.csv"))
    test_df = pd.read_csv(os.path.join(asap_path, str(i) + "_GOLD_test.csv"))

    print("_______",)
    print(i)

    for label in set(train_df["Score1"]):
        print("train", label, len(train_df[train_df["Score1"] == label]))

    print()
    for label in set(test_df["Score1"]):
        print("test", label, len(test_df[test_df["Score1"] == label]))
