import joblib
from sklearn.preprocessing import StandardScaler
from elasticsearch import Elasticsearch
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


# nltk.download('stopwords')

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

def get_bow(jobs, imp_fields, type="query"):
    """
    This function takes a job application or a list of job applications  and returns its bag of words.
    Originally job datus is a dictionary containing field names as keys.
    :param jobs:  either a dictionary or a list of dictionaries
    :param imp_fields: list of fields. Fields of the document that will be used for the formation of the query. These fieldes should be considered less noisy.
    :param type: string "query" or "pos" or "neg".
    :return:  list of tokens
    """
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


def query_preprocessing(q_tokens, q_imp_tokens, excluding_set, max_query_length=300):
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
        # print(set(bef).difference(set(q_tokens)))
    neg_index = np.random.randint(0, len(initial_qtokens), 50)
    query_neg = itemgetter(*neg_index)(list(initial_qtokens))
    return list(q_tokens), q_idf_list, list(query_neg)


def doc_preprocessing(d_tokens, excluding_set):
    # takes doc from elasticsearch and return tokens and tokens with unk
    doc_tokens = [d for d in d_tokens if d not in excluding_set]
    return doc_tokens


# query into elasticsearch - > get results excluding jobs used for querying

def get_bigrams(tokens):
    bigrams = []
    for i in range(len(tokens) - 1):
        bigrams.append(" ".join([tokens[i], tokens[i + 1]]))
    return bigrams


def make_query(q_tokens, num_of_res=100):
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


def generate_log_countbased_histogram(query, doc, w2v_model):
    q_vector = []
    for q_term in query:
        try:
            q_vector.append(w2v_model.wv[q_term])
        except KeyError:
            q_vector.append(list(np.random.randn(100, )))

    doc_vector = []
    for d_term in doc:
        try:
            doc_vector.append(w2v_model.wv[d_term])
        except:
            doc_vector.append(list(np.random.randn(100, )))

    cos_res = cosine_similarity(q_vector, doc_vector)

    histograms = []
    for i in range(cos_res.shape[0]):
        curr_row = cos_res[i, :]
        hist = np.histogram(curr_row, bins=29, range=(-1, 0.99))[0]
        exact_match_bin = len(curr_row[curr_row > 0.99])
        hist = np.append(hist, exact_match_bin)
        histograms.append(np.log([h + 1 for h in hist]))
    return histograms


def get_overlapping_ngrams(query_ngrams, doc_ngrams):
    query_bigrams_set = set(query_ngrams)
    doc_bigrams_set = set(doc_ngrams)
    overlap = query_bigrams_set.intersection(doc_bigrams_set)

    return len(overlap)


def save_dataframe(jobs_df, path):
    """
    Save preprocessed dataframe
    """
    pickling_on = open(path, 'wb')
    joblib.dump(jobs_df, pickling_on)
    pickling_on.close()


def normalize_BM25(res_BM25, scaler):
    data = scaler.transform(res_BM25)
    data = list(chain.from_iterable(data))
    return data


if __name__ == "__main__":
    # read cand_data
    np.random.seed(seed=2018)
    w2v_model = Word2Vec.load("C:/Users/nataz/Downloads/job_prop_data/dataset/word2vec/final_w2v/w2v_model.w2v")
    cand_df = read_data("C:/Users/nataz/Downloads/job_prop_data/dataset/train_df")

    path_to_jobs = "C:/Users/nataz/Downloads/job_prop_data/dataset/final_data/jobs_df_v1"
    jobs_df = read_data(path_to_jobs)
    drmm_train_data = read_data("C:/Users/nataz/Downloads/job_prop_data/dataset2/DRMM_train_data")
    pos_neg_dict = {"cand_id": [], "pos_BM25": [], "pos_normBM25": [], "neg_BM25": [], "neg_normBM25": [],
                    "query_idf": [],
                    "pos_histogram": [], "neg_histogram": [], "query_bow": [], "pos_bow": [], "neg_bow": [],
                    "overlapping_unigrams_pos": [], "overlapping_bigrams_pos": [],
                    "overlapping_unigrams_neg": [], "overlapping_bigrams_neg": []}
    count_pos_missing = 0
    num_of_cands = 0
    start = time.time()
    random.seed(a=2018)
    fields = ["function", "title", "keywords", "requirement_summary"]
    for cand_id in cand_df["cand_id"]:
        #if cand_id in drmm_train_data["cand_id"].tolist():
         #   continue

        flag = False
        # get pos document bow
        pos_doc_id = cand_df["job_list"][cand_df["cand_id"] == cand_id].tolist()[0][0]
        jobs_for_bow = jobs_df[jobs_df["id"] == pos_doc_id]
        pos_bow, pos_imp_bow = get_bow(jobs_for_bow, fields, type="pos")

        # create query bow set excluding pos doc!
        jobs_bow = []
        jobs_list_query = cand_df["job_list"][cand_df["cand_id"] == cand_id].tolist()[0][1:]
        for j in jobs_list_query:
            jobs_bow.append(jobs_df[jobs_df["id"] == j])
        cand_bow, cand_imp_bow = get_bow(jobs_bow, fields, type="query")

        # query into elasticsearch - > get results
        pickling_on = open("C:/Users/nataz/Downloads/job_prop_data/dataset/final_data/set_tokens_less_10", "rb")
        excluding_set = joblib.load(pickling_on)
        pickling_on.close()

        q_tokens, q_idf, query_neg = query_preprocessing(cand_bow, cand_imp_bow, excluding_set)
        num_of_res = 100
        ranked_results, res_BM25, res_id_list = make_query(q_tokens, num_of_res)

        # get positive BM25
        for i in range(len(ranked_results)):
            if ranked_results[i]["id"] == pos_doc_id:
                pos_neg_dict["cand_id"].append(cand_id)
                pos_neg_dict["pos_BM25"].append(res_BM25[i])
                pos_neg_dict["query_idf"].append(q_idf)
                pos_ind = i
                flag = True
                break
        if not flag:
            count_pos_missing += 1
            continue

        num_of_cands += 1
        pos_neg_dict["query_bow"].append(cand_bow)

        # exclude jobs used for querying
        overlap_jobs = exclude_overlaping_jobs(jobs_list_query, res_id_list)
        ranked_res = [r for r in ranked_results if r["id"] not in overlap_jobs]

        # get negative doc, BM25
        num_of_res = 100
        neg_results, neg_res_BM25, neg_res_id_list = make_query(query_neg, num_of_res)

        neg_ind = random.sample(range(50, 100), 1)[0]
        neg_not_found = True
        while neg_not_found:
            if neg_res_id_list[neg_ind] not in jobs_list_query:
                neg_not_found = False
                neg_doc_id = neg_res_id_list[neg_ind]
                neg_bow, neg_imp_bow = get_bow(jobs_df[jobs_df["id"] == neg_doc_id], fields, type="neg")
                if len(neg_bow) <= 1:
                    neg_ind = random.sample(range(50, 100), 1)[0]
                    neg_not_found = True
                    continue
            else:
                neg_ind = random.sample(range(50, 100), 1)[0]

        # Create normBM25 for the negative document
        # Normalize BM25
        scaler = StandardScaler()
        data = np.array(res_BM25 + neg_res_BM25).reshape(-1, 1)
        scaler.fit(data)
        pos_norm_BM25 = scaler.transform(np.array(res_BM25[pos_ind]).reshape(1,-1))
     
        neg_normBM25 = scaler.transform(np.array(neg_res_BM25[neg_ind]).reshape(1, -1))

        pos_neg_dict["pos_normBM25"].append(pos_norm_BM25)

        # Overlapping unigrams and bigrams
        query_bigrams = get_bigrams(cand_bow)
        pos_bigrams = get_bigrams(pos_bow)
        neg_bigrams = get_bigrams(neg_bow)

        overlapping_bigrams_pos = get_overlapping_ngrams(query_bigrams, pos_bigrams)
        overlapping_unigrams_pos = get_overlapping_ngrams(cand_bow, pos_bow)

        overlapping_bigrams_neg = get_overlapping_ngrams(query_bigrams, neg_bigrams)
        overlapping_unigrams_neg = get_overlapping_ngrams(cand_bow, neg_bow)

        # store number of overlapping unigrams and bigrams
        pos_neg_dict["overlapping_unigrams_pos"].append(overlapping_unigrams_pos)
        pos_neg_dict["overlapping_bigrams_pos"].append(overlapping_bigrams_pos)

        pos_neg_dict["overlapping_unigrams_neg"].append(overlapping_unigrams_neg)
        pos_neg_dict["overlapping_bigrams_neg"].append(overlapping_bigrams_neg)

        # CALCULATE NEGATIVE HISTOGRAM
        neg_tokens = doc_preprocessing(neg_bow, excluding_set)
        pos_tokens = doc_preprocessing(pos_bow, excluding_set)

        pos_neg_dict["neg_bow"].append(neg_bow)
        pos_neg_dict["pos_bow"].append(pos_bow)

        if len(pos_tokens) <= 1:
            pos_neg_dict["pos_histogram"].append([])
        else:
            pos_neg_dict["pos_histogram"].append(generate_log_countbased_histogram(q_tokens, pos_tokens, w2v_model))

        pos_neg_dict["neg_histogram"].append(generate_log_countbased_histogram(q_tokens, neg_tokens, w2v_model))

        # STORE INTO pos_neg_dict
        # store query
        pos_neg_dict["neg_BM25"].append(neg_res_BM25[neg_ind])
        pos_neg_dict["neg_normBM25"].append(neg_normBM25)

        if num_of_cands % 1000 == 0:
            print(num_of_cands)
            end = time.time()
            print(end - start)
            pos_neg_df = pd.DataFrame.from_dict(pos_neg_dict)
            save_dataframe(pos_neg_df, path="C:/Users/nataz/Downloads/job_prop_data/dataset2/DRMM_train_data_50")

        if num_of_cands >= 10000:
            end = time.time()
            print(len(pos_neg_dict["pos_histogram"]))
            print(len(pos_neg_dict["query_idf"]))
            print("time for 15000 data", end - start)
            break

    pos_neg_df = pd.DataFrame.from_dict(pos_neg_dict)
    print(len(pos_neg_df.loc[0, "pos_histogram"]))
    print(len(pos_neg_df.loc[0, "query_idf"]))

    save_dataframe(pos_neg_df, path="C:/Users/nataz/Downloads/job_prop_data/dataset2/DRMM_train_data_50")
    print("done")