#!/usr/bin/env python
# coding: utf-8

# In[1]:


import warnings
warnings.filterwarnings("ignore")

import sys
import os
import time
import json
import numpy as np
import pandas as pd
import dill

from functools import partial
from os.path import join, dirname, abspath, pardir, basename
from sklearn.pipeline import FeatureUnion, Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import KFold
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix


# In[2]:


def select(elements, num):
    if len(elements) >= num:
        elements[:num] = True
    return elements

def select_df(df, num_samples):
    df2 = df.copy()
    df2['selected'] = False

    groups = df2.groupby('class_label')
    p_select = partial(select, num = num_samples)
    df2['selected'] = groups.selected.transform(p_select)

    df_selected = df[df2.selected]

    return df_selected

def clean_df(df, num_samples, num_classes):
    df = select_df(df, num_samples)
    df = df[df.class_label.isin(num_classes)]

    df = df.sort_values('class_label')
    df.index = range(len(df.index))

    return df

def parse_file(fpath):
    with open(fpath) as f:
        data_dict = json.loads(f.read())
        try:
            for keys, values in data_dict.items():
                site_id = keys
                site_lengths = np.array(values['lengths'])
                yield site_id, site_lengths
        except Exception as e:
            print ("ERROR:", fpath, e)

def load_data(path):
    selected_files = []

    if os.path.isfile(path):
        fpath = path
        selected_files.append(fpath)
    else:
        dpath = path
        for root, _, files in os.walk(dpath):
            for fname in files:
                if not fname.endswith('.json'):
                    continue
                fpath = os.path.join(root, fname)
                selected_files.append(fpath)

    df = pd.DataFrame()
    for fpath in selected_files:
        row = {}
        for i, (site_id, site_lengths) in enumerate(parse_file(fpath)):
            row['fname'] = os.path.basename(fpath)
            row['class_label'] = site_id
            row['lengths'] = site_lengths
            df = df.append(row, ignore_index=True)
        print (i + 1, fpath)

    return df

def join_str(lengths):
    return ' '.join(map(str, lengths))

def get_bursts(len_seq):
    directions = len_seq / abs(len_seq)
    index_dir_change = np.where(directions[1:] - directions[:-1] != 0)[0] + 1
    bursts = np.split(len_seq, index_dir_change)
    return bursts

def get_url_list(url_list):
    urls = []
    with open(url_list) as f:
        lines = f.readlines()
        urls = [x.strip() for x in lines]
    return urls


# In[3]:


class NgramsExtractor:
    def __init__(self, max_ngram_len = 2):
        self.packet_counter = CountVectorizer(analyzer='word',
                                              tokenizer=lambda x: x.split(),
                                              stop_words=None,
                                              ngram_range=(1, max_ngram_len),)
        self.burst_counter = CountVectorizer(analyzer='word',
                                             tokenizer=lambda x: x.split(),
                                             stop_words=None,
                                             ngram_range=(1, max_ngram_len),)

    def fit(self, x, y = None):
        bursts = x.lengths.apply(get_bursts)
        self.packet_counter.fit(x.lengths.apply(join_str))
        self.burst_counter.fit(bursts.apply(join_str))
        return self

    def transform(self, data_list):
        bursts = data_list.lengths.apply(get_bursts)
        data_str = data_list.lengths.apply(join_str)
        bursts_str = bursts.apply(join_str)

        packet_ngrams = self.packet_counter.transform(data_str)
        burst_ngrams = self.burst_counter.transform(bursts_str)

        return np.concatenate((packet_ngrams.todense(), burst_ngrams.todense()), axis=1)


# In[4]:


def classify(train, test):
    # Ngrams feature extractor
    combinedFeatures = FeatureUnion([('ngrams', NgramsExtractor(max_ngram_len = 2))])

    # Create the pipeline for classification
    pipeline = Pipeline([
      ('features', combinedFeatures),
      ('classifier', RandomForestClassifier(n_estimators=100))
    ])

    # Train the pipeline
    pipeline.fit(train, train.class_label)

    # Evaluate the pipeline
    y_pred = pipeline.predict(test)
    acc = accuracy_score(test.class_label, y_pred)
    print("Accuracy Score:", acc)

    ''' Save the pipeline
    with open('doh_data_classify.pickle', 'wb') as model_file:
        dill.dump(pipeline, model_file)
    '''
    return list(test.class_label), list(y_pred)


# In[5]:


def classifier_train():
    # Locate dataset
    data_dir = join(abspath(join(dirname("__file__"), pardir)), 'dataset')
    print(data_dir)

    # Load dataset
    df = load_data(data_dir)
    print("initial data", df.shape)

    # Clean dataset
    num_classes = 500 # 1~500: websites.txt; 0: reserved
    num_samples = 10  # Number of samples for one class
    df_cleaned = clean_df(df, num_samples, map(str, range(num_classes + 1)))
    print("cleaned data", df_cleaned.shape)

    # Perform k-fold cross classification
    results = []
    df_cv = df_cleaned
    kf = StratifiedKFold(n_splits = 5)
    for k, (train_k, test_k) in enumerate(kf.split(df_cv, df_cv.class_label)):
        print("k-fold", k)
        result = classify(df_cv.iloc[train_k], df_cv.iloc[test_k])
        results.append(result)

    # Classification report
    reports = pd.DataFrame(columns=['k-fold', 'label', 'precision', 'recall', 'f1-score', 'support'])
    true_vectors, pred_vectors = [r[0] for r in results], [r[1] for r in results]
    for i, (y_true, y_pred) in enumerate(zip(true_vectors, pred_vectors)):
        # The precision, recall, F1 score for each class and averages in one k-fold
        output = classification_report(y_true, y_pred, output_dict=True)
        
        report = pd.DataFrame(output).transpose()
        report = report.reset_index()
        report = report.rename(columns={'index': 'label'})
        report['k-fold'] = i
        reports = reports.append(report)

    # Statistics report
    statistics = reports.groupby('label').describe().loc['macro avg']
    print("Mean")
    print(statistics.xs('mean', level=1))
    print("Standard deviation")
    print(statistics.xs('std', level=1))


# In[6]:


def classifier_serve():
    # Load pipeline
    loaded_model = dill.load(open('doh_data_classify.pickle', 'rb'))
    print("Model Loaded")

    # Load websites
    urls = get_url_list("../collection/websites.txt")

    for line in sys.stdin:
        # Locate file
        data_file = join(abspath(dirname("__file__")), line)[:-1]

        # Load file
        df_new = load_data(data_file)

        # Predict with pipeline
        pred_new = loaded_model.predict(df_new)
        pred_pro = loaded_model.predict_proba(df_new)
        pred_url = [ urls[int(index) - 1] for index in pred_new ]
        print("Prediction:", pred_url, np.max(pred_pro, axis=1))


# In[7]:


if __name__ == '__main__':
    if (len(sys.argv) == 2):
        if (sys.argv[1] == 'train'):
            print("Training...")
            classifier_train()
            print("Training done!!!")
            exit(0)
        elif (sys.argv[1] == 'serve'):
            print("Serving...")
            classifier_serve()
            print("Serving done!!!")
            exit(0)
    print("usage: doh_data_classify.py { train | serve }")
    exit(1)

