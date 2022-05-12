import sys
import numpy as np
import pandas as pd
from imblearn.under_sampling import RandomUnderSampler
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.metrics import cohen_kappa_score
from sklearn.naive_bayes import ComplementNB, MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.model_selection import StratifiedKFold
from preprocessing import tokenize
from utils import compose, scores_data_frame, predictions_data_frame, save_data_frames, determine_avg_type, prepare_directories, read_dataset
from nltk.corpus import stopwords
import time

# For undersampling of training data
#samples_per_label = 100


def train(target_path, train_fp, test_fp=None):
    train_fp, test_fp, data_dir, out_dir = prepare_directories(target_path, train_fp, test_fp)
    df_train, input_col, target_cols = read_dataset(train_fp)

    run_train_test = test_fp is not None
    if run_train_test:
        df_test, _, _ = read_dataset(test_fp)

    df_test_full = df_test

    lang = "english"
    #stop_en = stopwords.words(lang)
    #stop_en = stop_en + [x.capitalize() for x in stop_en]

    # feature_extractors = {
    #     'bow_1-2-ngram': CountVectorizer(
    #         ngram_range=(1,2),
    #         stop_words=stop_en,
    #         lowercase=False,
    #         tokenizer=tokenize_fn),
    #     'bow_1-ngram': CountVectorizer(
    #         ngram_range=(1,1),
    #         stop_words=stop_en,
    #         lowercase=False,
    #         tokenizer=tokenize_fn),
    #     'bow_1-ngram-w/olemma': CountVectorizer(
    #         ngram_range=(1,1),
    #         stop_words=stop_en,
    #         lowercase=False,
    #         tokenizer=tokenize),
    #     'bow_1-ngram-w/ostop-w/olemma-lower': CountVectorizer(
    #         ngram_range=(1,1),
    #         lowercase=True,
    #         tokenizer=tokenize),
    #     'tfidf_1-ngram-w/ostop-w/olemma': TfidfVectorizer(
    #         ngram_range=(1,1),
    #         lowercase=False,
    #         tokenizer=tokenize),
    #     'tfidf_1-2-ngram': TfidfVectorizer(
    #         ngram_range=(1,2),
    #         stop_words=stop_en,
    #         lowercase=False,
    #         tokenizer=tokenize_fn),
    #     'tfidf_1-ngram': TfidfVectorizer(
    #         ngram_range=(1,1),
    #         stop_words=stop_en,
    #         lowercase=False,
    #         tokenizer=tokenize_fn),
    # }

    feature_extractors = {
        'bow_1-3-gram': CountVectorizer(
            ngram_range=(1, 3),
            lowercase=True,
            tokenizer=tokenize),
    }

    estimators = {
        'random_forest': RandomForestClassifier(n_estimators=250, criterion='entropy'),
        #'random_forest_extreme': ExtraTreesClassifier(n_estimators=100, criterion='entropy'),
        'svm_C': SVC(class_weight='balanced'),
        #'svm_C=0.001': SVC(class_weight='balanced', C=0.001),
        #'naive-bayes': MultinomialNB(),
        #'naive-bayes_complement': ComplementNB(),
        'logistic-regression': LogisticRegression(class_weight='balanced', max_iter=1000),
        #'logistic-regression_C=0.001': LogisticRegression(class_weight='balanced', C=0.001)
    }

    cv = StratifiedKFold(n_splits=10, shuffle=True, random_state=4669)

    for target_col in target_cols:
        if run_train_test:

            df_test.columns = df_train.columns

            n_train = len(df_train)
            n_test = len(df_test)
            splits = [(np.arange(n_train), np.arange(n_train, n_train + n_test))]

            start_fold = -2
            df = pd.concat([df_train, df_test], keys=['train', 'test']).reset_index()
            dataset_name = f'{train_fp.stem}-{test_fp.stem}'
        else:
            df = df_train
            splits = cv.split(df[input_col], df[target_col])
            start_fold = 1
            dataset_name = f'{train_fp.stem}-cv'

        for i, idxs in enumerate(splits, start=start_fold):
            train_idx, test_idx = idxs

            for feature_name, extractor in feature_extractors.items():

                train_start_1 = time.time()

                # Cleaner, during dev
                #min_label = int(df[target_col].min())
                #max_label = int(df[target_col].max())
                # For testing to ensure nonsense labels in testing data do not affect range
                min_label = int(df.loc[df.index[train_idx], target_col].min())
                max_label = int(df.loc[df.index[train_idx], target_col].max())

                labels = [x for x in range(min_label, max_label + 1)]
                le = LabelEncoder().fit(labels)
                y = le.transform(df[target_col])
                avg = determine_avg_type(le.classes_)

                X_train = extractor.fit_transform(df.loc[df.index[train_idx], input_col])
                X_test = extractor.transform(df.loc[df.index[test_idx], input_col])

                y_train, y_test = y[train_idx], y[test_idx]

                # For undersampling of training data
                #label_dict = {}
                #for i in labels:
                #    label_dict[i] = samples_per_label
                #rus = RandomUnderSampler(sampling_strategy=label_dict, random_state=42)
                #X_train, y_train = rus.fit_resample(X_train, y_train)

                for estimator_name, estimator in estimators.items():

                    model = estimator.fit(X_train, y_train)
                    y_pred = model.predict(X_test)

                    #if not (df_test_full[target_col] == le.inverse_transform(y_test)).all():
                        #print("Gold labels are not in correct order!")
                        #sys.exit()

                    df_test_full[target_col] = df_test_full[target_col].astype(int)
                    df_test_full["predicted_score"] = le.inverse_transform(y_pred)
                    df_test_full["model"] = [estimator_name+"_"+feature_name] * len(df_test_full)

                    predictions = df_test_full
                    #predictions = predictions_data_frame(y_test, y_pred, dataset_name, estimator_name, feature_name, i)

                    scores = scores_data_frame(y_test, y_pred, dataset_name, estimator_name, feature_name, i, avg)
                    save_data_frames(out_dir/target_col, [predictions, scores], ['predictions.csv', 'scores.csv'])


if __name__ == '__main__':
    args = sys.argv[1:]
    train(*args)
