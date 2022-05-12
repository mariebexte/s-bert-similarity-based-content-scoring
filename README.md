# Similarity-based Content Scoring -</br>How to Make S-BERT Keep up with BERT

This is the code to our 2022 BEA paper for similarity-based scoring of the ASAP short answer scoring data. See below for a graphical overview of the approach. We first learn a model based on the similarities between S-BERT embeddings of a set of reference answers (left part of the image). During inference (right part of the image), the answers to score are compared to these reference answers, assigning the score for which the set of reference answers gives the highest average similiarity.

<p align="center">
<img src="https://github.com/mariebexte/s-bert-similarity-based-content-scoring/blob/main/overview.png" width="700">
</p>

To run all experiments (including the instance-based models), simply execute `bash run_everything.sh`.
To run only the similarity-based experiments, refer to `run_all_similarity_based.sh`.
For convenience, we also include scripts to generate alternative training data configurations (see `prepare_data.sh` for examples).

Make sure to unzip `data/full_data.zip` before running the full data setting.


# Cite

```
@InProceedings{bexte2022,
  title     = {Similarity-based Content Scoring - How to Make S-BERT Keep up with BERT},
  author    = {Bexte, Marie and Horbach, Andrea and Zesch, Torsten},
  booktitle = {Proceedings of The 17th Workshop on Innovative Use of NLP for Building Educational Applications},
  year      = {2022}
}
```
