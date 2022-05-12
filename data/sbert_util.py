import pandas as pd


def get_answer_pairs(df1, df2, num_ans_per_score, random_state):

    result_dict = {}
    result_dict_index = 0

    # For every answer in first dataframe
    for idx1, row1 in df1.iterrows():

        # For every score
        for score in set(df2["Score1"]):

            # Set similarity label depending on whether currently considered score is that of the current answer
            sim_label = 0
            if score == row1["Score1"]:
                sim_label = 1

            # Pick desired number of answers to compare to: If there aren't enough, take as many as possible (=all)
            score_df = df2[df2["Score1"] == score]
            try:
                chosen = score_df.sample(num_ans_per_score, random_state=random_state)
                success = True
            except ValueError:
                chosen = score_df

            # Create example pairs from all chosen answers to compare to
            for id2, row2 in chosen.iterrows():

                if not row1["Id"] == row2["Id"]:
                    result_dict[result_dict_index] = {"id_1": row1["Id"], "answer_1": row1["EssayText"],
                                                      "score_1": row1["Score1"],
                                                      "id_2": row2["Id"], "answer_2": row2["EssayText"],
                                                      "score_2": row2["Score1"],
                                                      "sim_label": sim_label}
                    result_dict_index += 1

    result_df = pd.DataFrame.from_dict(result_dict, "index")
    return result_df


# Method to pair all answers in two datasets
def get_all_possible_answer_pairs(df1, df2):

    result_dict = {}
    result_dict_index = 0

    # For every answer in the first dataframe
    for idx1, row1 in df1.iterrows():

        # Compare with every answer in the second dataframe
        for idx2, row2 in df2.iterrows():

            if not row1["Id"] == row2["Id"]:
                # Pair these two

                # Set similarity label depending on whether they have the same score
                sim_label = 0
                if row1["Score1"] == row2["Score1"]:
                    sim_label = 1

                result_dict[result_dict_index] = {"id_1": row1["Id"], "answer_1": row1["EssayText"],
                                                  "score_1": row1["Score1"], "id_2": row2["Id"],
                                                  "answer_2": row2["EssayText"], "score_2": row2["Score1"],
                                                  "sim_label": sim_label}
                result_dict_index += 1

    result_df = pd.DataFrame.from_dict(result_dict, "index")
    return result_df
