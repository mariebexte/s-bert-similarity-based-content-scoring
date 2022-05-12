import os
from sentence_transformers import SentenceTransformer, InputExample, losses, evaluation
import pandas as pd
from torch.utils.data import DataLoader
import torch
from eval_sbert import evaluate
import argparse
import shutil

base_model = "sentence-transformers/all-MiniLM-L6-v2"


def main(argv):

    device = "cpu"
    if torch.cuda.is_available():
        device = "cuda"

    if not os.path.exists(argv.results_folder):
        os.mkdir(argv.results_folder)

    target_folder = os.path.join(argv.results_folder, argv.train_condition)
    if not os.path.exists(target_folder):
        os.mkdir(target_folder)

    range_end = 11
    # When evaluating a pretrained model: Skip to evaluation
    if not argv.finetune:
        range_end = 2

    # For every prompt in the ASAP data
    for train_prompt_num in range(1, range_end):

        train_prompt_num = str(train_prompt_num)

        # When evaluating a pretrained model: Create generic RESULTS folder
        if not argv.finetune:
            train_prompt_num = "RESULTS"

        target_folder_prompt = os.path.join(target_folder, train_prompt_num)
        if not os.path.exists(target_folder_prompt):
            os.mkdir(os.path.join(target_folder_prompt))

        model = SentenceTransformer(base_model, device=device)
        train_path = argv.train_path

        if argv.finetune:

            # Where to store model
            model_path = os.path.join(target_folder_prompt, "finetuned_model")

            # If it is just about evaluating a finetuned model: Skip this
            if not argv.eval_finetuned:

                train_data = os.path.join(train_path, train_prompt_num + argv.train_name)
                val_data = os.path.join(train_path, train_prompt_num + argv.val_name)

                # Copy training and validation files into results folder
                shutil.copyfile(train_data, os.path.join(target_folder_prompt, train_prompt_num + argv.train_name))
                shutil.copyfile(val_data, os.path.join(target_folder_prompt, train_prompt_num + argv.val_name))

                # Read train examples
                train_df = pd.read_csv(train_data)
                train_examples = []
                for idx, row in train_df.iterrows():
                    train_examples.append(InputExample(texts=[row[1], row[4]], label=row[6]*1.0))

                # Read val examples
                val_df = pd.read_csv(val_data)
                val_examples = val_df[["answer_1", "answer_2", "sim_label"]]

                # Define train dataset, dataloader, train loss
                train_dataloader = DataLoader(train_examples, shuffle=True, batch_size=16)
                train_loss = losses.CosineSimilarityLoss(model)

                # Define validation
                evaluator = evaluation.BinaryClassificationEvaluator(val_examples["answer_1"].tolist(), val_examples["answer_2"].tolist(), val_examples["sim_label"].tolist(), show_progress_bar=False)

                # Tune the model
                model.fit(train_objectives=[(train_dataloader, train_loss)], use_amp=True, epochs=int(argv.num_epochs), evaluator=evaluator, evaluation_steps=int(argv.eval_steps), output_path=model_path, save_best_model=True, show_progress_bar=True)

            # Evaluate best model
            model = SentenceTransformer(model_path)

        # Create folder for current testing setup
        target_folder_prompt_results = os.path.join(target_folder_prompt, argv.test_condition)
        if not os.path.exists(target_folder_prompt_results):
            os.mkdir(target_folder_prompt_results)

        # Evaluate model on every prompt in the ASAP data
        for test_prompt_num in range(1, 11):

            test_prompt_num = str(test_prompt_num)

            test_result_folder = os.path.join(target_folder_prompt_results, str(test_prompt_num))
            if not os.path.exists(test_result_folder):
                os.mkdir(test_result_folder)

            train_data_raw = os.path.join(train_path, test_prompt_num + argv.raw_train_name)
            val_data_raw = os.path.join(train_path, test_prompt_num + argv.raw_val_name)
            test_data_raw = os.path.join(argv.test_path, test_prompt_num + argv.raw_test_name)

            # Copy files used for testing into testing results folder
            shutil.copyfile(train_data_raw, os.path.join(test_result_folder, test_prompt_num + argv.raw_train_name))
            shutil.copyfile(val_data_raw, os.path.join(test_result_folder, test_prompt_num + argv.raw_val_name))
            shutil.copyfile(test_data_raw, os.path.join(test_result_folder, test_prompt_num + argv.raw_test_name))

            # Eval testing data: Get sentence embeddings for all testing and reference answers
            test_df_raw = pd.read_csv(test_data_raw)
            test_df_raw['embedding'] = test_df_raw['EssayText'].apply(model.encode)

            val_df_raw = pd.read_csv(val_data_raw)
            train_df_raw = pd.read_csv(train_data_raw)
            ref_df_raw = val_df_raw.append(train_df_raw)
            ref_df_raw['embedding'] = ref_df_raw["EssayText"].apply(model.encode)

            evaluate(test_result_folder, test_df_raw, ref_df_raw)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_condition", help="name of folder for results to this condition")
    parser.add_argument("--test_condition", help="name of folder for test results")
    parser.add_argument('--do_not_use_pretrained', dest='finetune', action='store_true')
    parser.add_argument('--use_pretrained', dest='finetune', action='store_false')
    parser.set_defaults(finetune=True)
    parser.add_argument('--do_not_eval_finetuned', dest='eval_finetuned', action='store_false')
    parser.add_argument('--eval_finetuned', dest='eval_finetuned', action='store_true')
    parser.set_defaults(eval_finetuned=False)
    parser.add_argument("--num_epochs")
    parser.add_argument("--eval_steps")
    parser.add_argument("--results_folder", help="path to results folder")
    parser.add_argument("--train_path", help="location of training files")
    parser.add_argument("--train_name", help="filenames of training data")
    parser.add_argument("--raw_train_name", help="filenames of raw training data")
    parser.add_argument("--val_name", help="filenames of validation data")
    parser.add_argument("--raw_val_name", help="filenames of raw validation data")
    parser.add_argument("--test_path", help="location of testing files")
    parser.add_argument("--raw_test_name", help="filenames of raw testing data")
    args = parser.parse_args()
    main(args)
