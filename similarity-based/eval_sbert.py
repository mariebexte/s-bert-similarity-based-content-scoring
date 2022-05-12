import pandas as pd
import os
from scipy import spatial
from sklearn.metrics import cohen_kappa_score, accuracy_score, precision_recall_fscore_support


def evaluate(data_path, test_df_raw, ref_df_raw):

    # To save the actual similarity scores
    #sim_df = pd.DataFrame(columns=["id1", "text1", "embedding1", "score1", "id2", "text2", "embedding2", "score2", "cos_sim"])
    # To save predictions
    pred_df = pd.DataFrame(columns=["id", "text", "gold(Score1)", "sim(avg)", "pred(avg)", "sim(max)", "pred(max)"])

    # Cross every test embedding with every train embedding
    for idx, line in test_df_raw.iterrows():

        # Determine similarities
        copy_eval = ref_df_raw[["Id", "EssayText", "Score1", "embedding"]].copy()
        # Put reference answers as 'answers 2'
        copy_eval.columns = ["id2", "text2", "score2", "embedding2"]
        # Put current answer to score as 'answer 1' (copy num_ref_answers times) to compare to all of them
        copy_eval["id1"] = [line["Id"]]*len(copy_eval)
        copy_eval["text1"] = [line["EssayText"]]*len(copy_eval)
        copy_eval["score1"] = [line["Score1"]]*len(copy_eval)
        copy_eval["embedding1"] = [line["embedding"]]*len(copy_eval)
        emb1 = list(copy_eval["embedding1"])
        emb2 = list(copy_eval["embedding2"])
        copy_eval["cos_sim"] = [1 - spatial.distance.cosine(emb1[i], emb2[i]) for i in range(len(copy_eval))]
        #sim_df = sim_df.append(copy_eval)

        # Determine prediction: MAX
        max_row = copy_eval.iloc[[copy_eval["cos_sim"].argmax()]]
        max_sim = max_row.iloc[0]["cos_sim"]
        max_pred = max_row.iloc[0]["score2"]

        # Determine prediction: AVG
        cluster_avgs = {}
        for cluster in set(copy_eval["score2"]):
            cluster_subset = copy_eval[copy_eval["score2"] == cluster]
            cluster_avgs[cluster] = cluster_subset["cos_sim"].mean()
        avg_pred = max(cluster_avgs, key=cluster_avgs.get)
        a_series = pd.Series([line["Id"], line["EssayText"], line["Score1"], cluster_avgs[avg_pred], avg_pred, max_sim, max_pred], index=pred_df.columns)
        # Change to .from_dict for larger amounts of testing data
        pred_df = pred_df.append(a_series, ignore_index=True)

    #sim_df.to_csv(os.path.join(data_path, "scores.csv"))
    pred_df.to_csv(os.path.join(data_path, "predictions.csv"))

    # Calculate statistics
    with open(os.path.join(data_path, "stats.txt"), 'w') as out:

        # Confusion matrix
        confusion_matrix_max = pd.crosstab(pred_df['gold(Score1)'], pred_df['pred(max)'])
        confusion_matrix_avg = pd.crosstab(pred_df['gold(Score1)'], pred_df['pred(avg)'])
        out.write(str(confusion_matrix_max)+"\n\n")
        out.write(str(confusion_matrix_avg)+"\n\n")

        # Accuracy
        acc_max = accuracy_score(pred_df["gold(Score1)"].tolist(), pred_df["pred(max)"].tolist())
        acc_avg = accuracy_score(pred_df["gold(Score1)"].tolist(), pred_df["pred(avg)"].tolist())
        out.write("acc(max)\t"+str(acc_max)+"\n")
        out.write("acc(avg)\t"+str(acc_avg)+"\n\n")

        # Per cluster: precision/recall/fscore
        p_max, r_max, f_max, s_max = precision_recall_fscore_support(pred_df["gold(Score1)"].tolist(), pred_df["pred(max)"].tolist())
        p_avg, r_avg, f_avg, s_avg = precision_recall_fscore_support(pred_df["gold(Score1)"].tolist(), pred_df["pred(avg)"].tolist())

        out.write("per-class(max)\n")
        for i in range(len(p_max)):
            out.write("Score1\t"+str(i)+"\tprecision\t"+str(p_max[i])+"\trecall\t"+str(r_max[i])+"\tfscore\t"+str(f_max[i])+"\tsupport\t"+str(s_max[i])+"\n")

        out.write("\nper-class(avg)\n")
        for i in range(len(p_avg)):
            out.write("Score1\t"+str(i)+"\tprecision\t"+str(p_avg[i])+"\trecall\t"+str(r_avg[i])+"\tfscore\t"+str(f_avg[i])+"\tsupport\t"+str(s_avg[i])+"\n")

        out.write("\nQWK(max)\t"+str(cohen_kappa_score(pred_df["gold(Score1)"].tolist(), pred_df["pred(max)"].tolist(), weights='quadratic'))+"\n")
        out.write("QWK(avg)\t"+str(cohen_kappa_score(pred_df["gold(Score1)"].tolist(), pred_df["pred(avg)"].tolist(), weights='quadratic')))
