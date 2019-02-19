import joblib
from elasticsearch import Elasticsearch
import joblib
import itertools
import string
import pandas as pd
import time
import json
import random
from gensim.models import Word2Vec
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import nltk
from itertools import chain
import operator
from tqdm import tqdm
import collections
from operator import itemgetter
from nltk.corpus import stopwords

#nltk.download('stopwords')

def save_dataframe(df, path):
    pickling_on = open(path, 'wb')
    joblib.dump(df, pickling_on)
    pickling_on.close()

def connect_elasticsearch():
    _es = None
    _es = Elasticsearch([{'host': 'localhost', 'port': 9200}])
    if _es.ping():
        return _es
    else:
        print('Unable to connect!')

def read_data(path):
    pickling_on = open(path, 'rb')
    df = joblib.load(pickling_on)
    pickling_on.close()

    return df

def get_idf(tokens):
    small_num = 0.1
    pickling_on = open("C:/Users/nataz/Downloads/job_prop_data/dataset/final_data/idf", "rb")
    idf = joblib.load(pickling_on)
    pickling_on.close()
    idf_dict = {}
    idf_list = []
    for tok in tokens:
        try:
            idf_dict[tok] = idf[tok]
        except KeyError as e:
            idf_dict[tok] = small_num
      #  idf = {your_key: idf_list[your_key] for your_key in tokens}

    for tok in tokens:
        idf_list.append(idf_dict[tok])

    return idf_dict, idf_list

def get_bow(jobs, imp_fields, type = "query"):
    bow = []
    imp_bow = []
    stop_words_eng = stopwords.words('english')
    stop_words_fr = stopwords.words('french')
    stop_words_greek = stopwords.words('greek')

    if type == "query":
        for job in jobs:
            for col in imp_fields:
                if col == "keywords":
                    bow.extend(job[col].tolist()[0])
                    imp_bow.extend(job[col].tolist()[0])
                elif col == "title" or col == "function":
                    bow.extend(job[col].tolist()[0].split())
                    imp_bow.extend(job[col].tolist()[0].split())
                elif col == "requirement_summary":
                    bow.extend(job[col].tolist()[0].split())
    elif type == "pos" or type == "neg":
        for col in imp_fields:
            if col == "keywords":
                bow.extend(jobs[col].tolist()[0])
                imp_bow.extend(jobs[col].tolist()[0])
            elif col == "title" or col == "function":
                bow.extend(jobs[col].tolist()[0].split())
                imp_bow.extend(jobs[col].tolist()[0].split())
            elif col == "requirement_summary":
                bow.extend(jobs[col].tolist()[0].split())
    stop_words = set(stop_words_eng + stop_words_fr + stop_words_greek)
    bow_stop = [w for w in bow if w not in stop_words]
    imp_bow_stop = [w for w in imp_bow if w not in stop_words]
    return bow_stop, imp_bow_stop

def query_preprocessing(q_tokens, q_imp_tokens, excluding_set, max_query_length = 300):
    q_tokens = set(q_tokens)
    q_tokens = set(q_tokens).difference(excluding_set)
    q_idf_dict, q_idf_list = get_idf(q_tokens)
    initial_qtokens = q_tokens
    if q_tokens.__len__() > max_query_length:
        z = collections.Counter(q_tokens)
        q_tokens = sorted(z, key=lambda k: (z[k], q_idf_dict[k]), reverse=True)
        q_idf_dict, q_idf_list = get_idf(q_tokens)
        i = 1
        flag = False
        q_len = len(q_tokens)
        while q_len > max_query_length:
            if q_tokens[-i] not in set(q_imp_tokens):
                del q_tokens[-i]
                del q_idf_list[-i]
                q_len = len(q_tokens)
                continue
            q_len = q_tokens.__len__()
            i += 1
    neg_index = np.random.randint(0, len(initial_qtokens), 100)
    query_neg = itemgetter(*neg_index)(list(initial_qtokens))
    return list(q_tokens), q_idf_list, list(query_neg)

def make_query(q_tokens, num_of_res = 100):
    """
    Takes a preprocessed tokenized query and returns results from elastic search
    :param q_tokens: list of tokens
    :return: list of dictionaries containing the result
    """
    es = connect_elasticsearch()
    query = " ".join(q_tokens)
    q = {
        "query": {
            "bool": {
                "should": [
                    {
                        "multi_match": {
                            "query": query,
                            "type": "most_fields",
                            "fields": ["function^1.5", "title^1.5", "requirement_summary^1.2", "keywords^1.5",
                                       "description^1"],
                            "auto_generate_synonyms_phrase_query": "false"

                        }, }]}}, "sort": [{"_score": {"order": "desc"}}]
    }

    ranked_results = []
    from elasticsearch import TransportError
    try:
        res = es.search(index='jobs', body=q, size=num_of_res, from_=0)  # how you index your document here
    except TransportError as e:
        print(e.info)

    res_BM25 = []
    for i in range(len(res['hits']['hits'])):
        res_dict = {}
        for k in res['hits']['hits'][i]['_source'].keys():
            res_dict[k] = res['hits']['hits'][i]['_source'][k]
        ranked_results.append(res_dict)

        res_BM25.append(res['hits']['hits'][i]['_score'])
    res_id_list = []


    for i in range(len(ranked_results)):
        res_id_list.append(ranked_results[i]["id"])

    return ranked_results, res_BM25, res_id_list

def exclude_overlaping_jobs(rel_docs, res_id_list):
    num_of_overlap = 0
    overlap_jobs = []
    for res in res_id_list:
        if res in rel_docs:
            overlap_jobs.append(res)
            num_of_overlap += 1
    return overlap_jobs



def create_data(cand_df, jobs_df, save_path, pairs):

    excluding_set = read_data("C:/Users/nataz/Downloads/job_prop_data/dataset/final_data/set_tokens_less_10")

    test_dict = {"cand_id": [], "query_tokens": [], "query_idf": [], "positive_doc_ids": [], "positive_doc_BM25": [],
                   "negative_doc_ids": [], "negative_doc_BM25": []}
    fields = ["function", "title", "keywords", "requirement_summary"]
    start = time.time()
    for index, row in tqdm(pairs.iterrows()):
        cand_id = pairs.loc[index,"cand_id"]
        if cand_id not in set(
        pairs["cand_id"].tolist()):
            continue

        #create bow using all jobs in job list
        jobs_bow = []
        jobs_list_query = cand_df["job_list"][cand_df["cand_id"] == cand_id].tolist()[0]
        for j in jobs_list_query:
            jobs_bow.append(jobs_df[jobs_df["id"] == j])
        cand_bow, cand_imp_bow = get_bow(jobs_bow, fields, type="query")


        #get positive labelled documents
        q_tokens, q_idf, query_neg = query_preprocessing(cand_bow, cand_imp_bow, excluding_set, max_query_length=300)

        num_of_res = 100
        ranked_results, res_BM25, res_id_list = make_query(q_tokens, num_of_res)

        # exclude jobs used for querying
        jobs_list_query = cand_df["job_list"][cand_df["cand_id"] == cand_id].tolist()[0][1:]

        overlap_jobs = exclude_overlaping_jobs(jobs_list_query, res_id_list)

        ranked_res_index = [i for i in range(len(res_id_list)) if res_id_list[i] not in overlap_jobs]
        ranked_res_ind = ranked_res_index[:50]

        positive_results = np.asarray(res_id_list)[ranked_res_ind]
        positive_results_BM25 = np.asarray(res_BM25)[ranked_res_ind]


        #get negativelly labeled documents
        num_of_res = 110
        neg_results, neg_res_BM25, neg_res_id = make_query(query_neg, num_of_res)

        # exclude jobs used for querying
        neg_ranked_res_index = [i for i in range(len(neg_res_id)) if neg_res_id[i] not in overlap_jobs]
        neg_res_id_list = np.asarray(neg_res_id)[neg_ranked_res_index]

        negative_results = []
        negative_results_BM25 = []
        for i in range(len(neg_res_id_list)):
            if len(negative_results)==50:
                break
            else:
                if neg_res_id_list[i] not in set(positive_results):
                    negative_results.append(neg_res_id_list[i])
                    negative_results_BM25.append(neg_res_BM25[i])


        end = time.time()
        test_dict["cand_id"].append(cand_id)
        test_dict["query_tokens"].append(q_tokens)
        test_dict["query_idf"].append(q_idf)
        test_dict["positive_doc_ids"].append(positive_results)
        test_dict["positive_doc_BM25"].append(positive_results_BM25)
        test_dict["negative_doc_ids"].append(negative_results)
        test_dict["negative_doc_BM25"].append(negative_results_BM25)

    test_df = pd.DataFrame.from_dict(test_dict)

    save_dataframe(test_df, save_path)


cand_df = read_data("C:/Users/nataz/Downloads/job_prop_data/dataset/temp/train_df")
jobs_df = read_data("C:/Users/nataz/Downloads/job_prop_data/dataset/final_data/jobs_df_v1")

save_path = "C:/Users/nataz/Downloads/job_prop_data/dataset2/test_split_100_toks/test_MAP"
test_pairs = read_data("C:/Users/nataz/Downloads/job_prop_data/dataset2/test_split/test_pairs")
create_data(cand_df, jobs_df, save_path, test_pairs)

save_path = "C:/Users/nataz/Downloads/job_prop_data/dataset2/test_split_100_toks/dev_MAP"
dev_pairs = read_data("C:/Users/nataz/Downloads/job_prop_data/dataset2/test_split/dev_pairs")
create_data(cand_df, jobs_df, save_path, dev_pairs)

save_path = "C:/Users/nataz/Downloads/job_prop_data/dataset2/test_split_100_toks/tune_MAP"
dev_pairs = read_data("C:/Users/nataz/Downloads/job_prop_data/dataset2/test_split/tuning_pairs")
create_data(cand_df, jobs_df, save_path, dev_pairs)
