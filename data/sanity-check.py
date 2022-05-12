import pandas as pd
import os

full_data_path = "full_data"
full_train_name = "_FULL_train.csv"
full_train_raw_name = "_FULL_train_raw.csv"
full_val_name = "_FULL_val.csv"
full_val_raw_name = "_FULL_val_raw.csv"

limited_data_path = "limited_data_60"
limited_train_name = "_60_train.csv"
limited_train_raw_name = "_60_train_raw.csv"
limited_val_name = "_60_val.csv"
limited_val_raw_name = "_60_val_raw.csv"
limited_train_val_raw_name = "_60_train_val_raw.csv"

gold_split_path = "gold_train_test_split"
gold_train_name = "_GOLD_train.csv"
gold_test_name = "_GOLD_test.csv"


for prompt in range(1, 11):

    prompt = str(prompt)

    #### Regarding TRAIN
    # For each of the files: read into dataframe, collect set of answer ids that occur within
    full_train = pd.read_csv(os.path.join(full_data_path, prompt + full_train_name))
    full_train_raw = pd.read_csv(os.path.join(full_data_path, prompt + full_train_raw_name))
    full_val = pd.read_csv(os.path.join(full_data_path, prompt + full_val_name))
    full_val_raw = pd.read_csv(os.path.join(full_data_path, prompt + full_val_raw_name))

    limited_train = pd.read_csv(os.path.join(limited_data_path, prompt + limited_train_name))
    limited_train_raw = pd.read_csv(os.path.join(limited_data_path, prompt + limited_train_raw_name))
    limited_val = pd.read_csv(os.path.join(limited_data_path, prompt + limited_val_name))
    limited_val_raw = pd.read_csv(os.path.join(limited_data_path, prompt + limited_val_raw_name))
    limited_train_val_raw = pd.read_csv(os.path.join(limited_data_path, prompt + limited_train_val_raw_name))

    gold_train = pd.read_csv(os.path.join(gold_split_path, prompt + gold_train_name))
    gold_test = pd.read_csv(os.path.join(gold_split_path, prompt + gold_test_name))

    # Transform into sets of IDs
    full_train_ids = set(full_train["id_1"])
    full_train_ids = full_train_ids.union(set(full_train["id_2"]))
    full_val_ids = set(full_val["id_1"])
    full_val_ids = full_val_ids.union(set(full_val["id_2"]))
    full_train_raw_ids = set(full_train_raw["Id"])
    full_val_raw_ids = set(full_val_raw["Id"])

    limited_train_ids = set(limited_train["id_1"])
    limited_train_ids = limited_train_ids.union(set(limited_train["id_2"]))
    limited_val_ids = set(limited_val["id_1"])
    limited_val_ids = limited_val_ids.union(set(limited_val["id_2"]))
    limited_train_raw_ids = set(limited_train_raw["Id"])
    limited_val_raw_ids = set(limited_val_raw["Id"])
    limited_train_val_raw_ids = set(limited_train_val_raw["Id"])

    gold_train_ids = set(gold_train["Id"])
    gold_test_ids = set(gold_test["Id"])

    # Same IDs in train and train raw (limited, full)
    limited_train_same = limited_train_ids == limited_train_raw_ids
    print(prompt, "TRAIN", "LIMITED", "Compare pairs and raw, should be the same: ", limited_train_same)
    full_train_same = full_train_ids == full_train_raw_ids
    print(prompt, "TRAIN", "FUll", "Compare pairs and raw, should be the same", full_train_same)

    # Same IDs in val and val raw (limited, full)
    limited_val_same = limited_val_ids == limited_val_raw_ids
    print(prompt, "VAL", "LIMITED", "Compare between pairs and raw, should be same:", limited_val_same)
    full_val_same = full_val_ids == full_val_raw_ids
    print(prompt, "VAL", "FULL", "Compare between pairs and raw, should be same:", full_val_same)

    # limited train val raw is exactly limited train raw + limited val raw
    train_val_raw_same = limited_train_raw_ids.union(limited_val_raw_ids) == limited_train_val_raw_ids
    print(prompt, "TRAIN_VAL", "LIMITED", "Check if train_val_raw consists of train_raw and val_raw:", train_val_raw_same)

    # GOLD train is exactly full train raw + full val raw
    gold_train_val_same = full_train_raw_ids.union(full_val_raw_ids) == gold_train_ids
    print(prompt, "TRAIN_VAL", "FULL", "Check if gold_train consists of train_raw and val_raw:", gold_train_val_same)

    #### Regarding TEST

    # No overlap with any: GOLD train, any similarity-based
    gold_test_train_intersect = len(gold_train_ids.intersection(gold_test_ids))
    gold_test_limited_train_intersect = len(gold_test_ids.intersection(limited_train_raw_ids))
    gold_test_limited_val_intersect = len(gold_test_ids.intersection(limited_val_raw_ids))
    gold_test_full_train_intersect = len(gold_test_ids.intersection(full_train_raw_ids))
    gold_test_full_val_intersect = len(gold_test_ids.intersection(full_val_raw_ids))
    limited_train_val_intersect = len(limited_train_raw_ids.intersection(limited_val_raw_ids))
    full_train_val_intersect = len(full_train_raw_ids.intersection(full_val_raw_ids))
    print(prompt, "Check: All overlaps between splits are zero:",
          gold_test_train_intersect, gold_test_limited_train_intersect, gold_test_limited_val_intersect,
          gold_test_full_train_intersect, gold_test_full_val_intersect, limited_train_val_intersect, full_train_val_intersect)

    #### Print count stats: Absolute and type-level
    # For each of the files: num rows, num distinct ids
    print()
    print(prompt, "LIMITED", "train", len(limited_train), len(limited_train_ids))
    print(prompt, "LIMITED", "val", len(limited_val), len(limited_val_ids))
    print(prompt, "LIMITED", "train raw", len(limited_train_raw), len(limited_train_raw_ids))
    print(prompt, "LIMITED", "val raw", len(limited_val_raw), len(limited_val_raw_ids))
    print()
    print(prompt, "FULL", "train", len(full_train), len(full_train_ids))
    print(prompt, "FULL", "val", len(full_val), len(full_val_ids))
    print(prompt, "FULL", "train raw", len(full_train_raw), len(full_train_raw_ids))
    print(prompt, "FULL", "val raw", len(full_val_raw), len(full_val_raw_ids))
    print()
    print(prompt, "GOLD", "train", len(gold_train), len(gold_train_ids))
    print(prompt, "GOLD", "test", len(gold_test), len(gold_test_ids))
    print("_____")