# -*- coding: utf-8 -*-
"""
Created on Fri Nov 30 17:03:50 2018

@author: nataz
"""


import joblib
import joblib
import pandas as pd
import time
import json
import random
from gensim.models import Word2Vec
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import nltk
from itertools import chain
from tqdm import tqdm
import collections
from operator import itemgetter
from nltk.corpus import stopwords
import operator

def read_data(path):
    pickling_on = open(path, 'rb')
    jobs_df = joblib.load(pickling_on)
    pickling_on.close()
    return jobs_df


def save_data(jobs_df, path):
    pickling_on = open(path, 'wb')
    joblib.dump(jobs_df, pickling_on)
    pickling_on.close()
    
def connect_elasticsearch():
    _es = None
    _es = Elasticsearch([{'host': 'localhost', 'port': 9200}])
    if _es.ping():
        return _es
    else:
        print('Unable to connect!')

def get_bow(job_id,imp_fields, jobs_df):

    bow = []
    stop_words = set(stopwords.words('english') + stopwords.words('french') +stopwords.words('greek'))
    job = jobs_df[jobs_df["id"] == job_id]
    for col in imp_fields:
        if col == "keywords" :
            bow.extend(job[col].tolist()[0])
        elif col == "title" or col == "function":
            bow.extend(job[col].tolist()[0].split())
        elif col == "requirement_summary":
            bow.extend(job[col].tolist()[0].split())
    bow_stop = [w for w in bow if w not in stop_words]

    return bow_stop

if __name__ == "__main__":
    stop_words = stopwords.words('english')
    
    test_path = "dataset/test_sets/test_MAP"
    test_df = read_data(test_path)
    jobs_path = "dataset/jobs_df_v1"
    jobs_df = read_data(jobs_path)
    results_path = "dataset/results/baseline_dict"
    w2v_model = Word2Vec.load("dataset/w2v_model.w2v")

    results_dict = {"wmv_top_10": [], "BM25_top_10":[]}
    imp_fields = ["keywords", "title", "function", "requirement_summary"]
    for i, r in test_df.iterrows():
        q_tokens = test_df.loc[i, "query_tokens"]
        jobs_to_rerank = test_df.loc[i,"positive_doc_ids"].tolist() + test_df.loc[i,"negative_doc_ids"]
        wmv_results = []
        for job_id in jobs_to_rerank:
            doc_tokens = get_bow(job_id, imp_fields,jobs_df)
            dist_score = w2v_model.wv.wmdistance(q_tokens, doc_tokens)
            wmv_results.append((r, dist_score))

        wmv_sorted = sorted(wmv_results, key=itemgetter(1))
        wmv_top = list(zip(*wmv_sorted))[0]
        BM25_res = jobs_to_rerank
        results_dict["wmv_top_10"].append(wmv_top[:10])
        results_dict["BM25_top_10"].append(BM25_res[:10])
    save_data(results_dict, path)

