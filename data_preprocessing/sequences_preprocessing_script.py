# -*- coding: utf-8 -*-
"""
Created on Mon Nov 12 12:57:00 2018

@author: natassa

This script implements the first cleaning phase where we keep only the candidates that have applied in the past into relavant to each other jobs
"""


import pandas as pd
import numpy as np
import joblib
import json
import os
import time
from gensim.models import Word2Vec
import numpy as np
from nltk import download
from nltk.corpus import stopwords
from sklearn.metrics.pairwise import cosine_similarity
from gensim.models import Word2Vec
from gensim.similarities import WmdSimilarity

def get_json_filenames(path):
    path_to_json = path
    json_files = [pos_json for pos_json in os.listdir(path_to_json) if pos_json.endswith('.json')]

    json_list = []
    for index, js in enumerate(json_files):
        with open(os.path.join(path_to_json, js)) as json_file:
            json_list.append(json_file.read())
    return json_list

def read_df(path):
    '''
    :param path: path to data stored as pickle
    :return: data
    '''
    pickling_on = open(path, 'rb')
    df = joblib.load(pickling_on)
    pickling_on.close()

    return df

def read_cand_df(path):
    '''
    :param path: path to candidate data stored in json format
    :return: candidates data stored in pandas format
    '''
    files_list = get_json_filenames(path)
    cand_json = []
    for i in range(len(files_list)):
        cand_json.extend(json.loads(files_list[i]))
    cand_df = pd.DataFrame(cand_json)

    return cand_df


def get_bow(jobs, imp_fields, type="query"):
    """
    :param jobs: pandas Series corresponding to the job ids used to create bag of words
    :param imp_fields: fields of documents with less noisy content
    :param type: choose between query , pos, neg if we are about to construct query bag of words(bow) or bow for positive or negative document
    :return: bag of words containing all document, bag of words containing the words of the more informative fields,
             dictionary of fields  storing as keys the fields of the bow and as values the words contained on each field
    """
    bow = []
    imp_bow = []
    word2field = {"function": [], "keywords": [], "title": []}
    if type == "query":
        for job in jobs:
            for col in imp_fields:
                if col == "keywords":
                    bow.extend(job[col].tolist()[0])
                    imp_bow.extend(job[col].tolist()[0])
                    word2field[col].extend(job[col].tolist()[0])
                elif col == "title" or col == "function":
                    bow.extend(job[col].tolist()[0].split())
                    imp_bow.extend(job[col].tolist()[0].split())
                    word2field[col].extend(job[col].tolist()[0].split())
                elif col == "requirement_summary":
                    bow.extend(job[col].tolist()[0].split())
                    word2field[col].extend(job[col].tolist()[0].split())
    elif type == "pos" or type == "neg":
        for col in imp_fields:
            if col == "keywords":
                bow.extend(jobs[col].tolist()[0])
                imp_bow.extend(jobs[col].tolist()[0])
                word2field[col].extend(jobs[col].tolist()[0])
            elif col == "title" or col == "function":
                bow.extend(jobs[col].tolist()[0].split())
                imp_bow.extend(jobs[col].tolist()[0].split())
                word2field[col].extend(jobs[col].tolist()[0].split())
            elif col == "requirement_summary":
                bow.extend(jobs[col].tolist()[0].split())
                word2field[col].extend(jobs[col].tolist()[0].split())
    return bow, imp_bow, word2field



def discard_non_relevant(jobs_list, stop_words, w2v_model):
    '''
    :param jobs_list: list of jobs where candidate applied in the past. Each item in the list is a dictionary of field names and strings as values
    :stop_words: words that will be discarded from each job to decrease noise
    :w2v_model: gensim library word2vec model used to calculate Word Movers Similarity Metric to find the irrelevant jobs in the list 
    '''
    dist_list = []
    jobs_corpus = []
    imp_fields = ["function", "title", "keywords"]
    for i in range(len(jobs_list)):
        bow1, _, _ = get_bow(jobs_df[jobs_df["id"] == jobs_list[i]], imp_fields, type="pos")
        bow1_stop = [w for w in bow1 if w not in stop_words]
        jobs_corpus.append(bow1_stop) 
    instance = WmdSimilarity(jobs_corpus, w2v_model, num_best = None)
    for i in range(len(jobs_list)):
        similarity = instance[jobs_corpus[i]]
        dist_list.extend([ (sum(similarity) - 1) / (len(similarity)-1)])
    new_jobs_list = [jobs_list[i] for i in range(len(jobs_list)) if dist_list[i] > 0.45]
    
    #print(jobs_list)
    #print(dist_list)
    #print(new_jobs_list)
    #print(len(jobs_list),len(new_jobs_list))
    return new_jobs_list

def process_candidates(cand_df, jobs_df):
    '''
    :param cand_df: Candidates data in pandas dataframe format
    :param jobs_df: Jobs data in pandas format
    :return: Candidates after the first cleaning phase. We keep only the candidates for which  querying into Elasticsearch engine using their bag of words brings as the positive document in the first 100 results.
             Also we keep only the documents for which top score is greater than 4.
    '''
    stop_words = stopwords.words('english')
    w2v_model = Word2Vec.load("C:/Users/nataz/Downloads/job_prop_data/dataset/word2vec/final_w2v/w2v_model.w2v")
    w2v_model.init_sims(replace=True)
    time_list = 0
    count_non_empty = 0
    num_of_loops = 0
    train_dict = {"cand_id": [], "job_list": []}
    test_dict = {"cand_id": [], "job_list": []}
    for index, row in cand_df.iterrows():
        end = time.time()
        if index != 0:
            time_list += end - start
            num_of_loops += 1
        start = time.time()
        
        new_jobs_list = []
        cand_id = row["identifier"]
        
        
        cand_jobs = pd.DataFrame(row["sequence"])
        cand_jobs.sort_values("score", axis=0, inplace=True, ascending=False)
        if cand_jobs["score"].tolist()[0] < 4:  # if top score is less tha 5 discard candidate
            continue
        else:
            #discard empty jobs
            for i, v in cand_jobs.iterrows():
                if not jobs_df[jobs_df["id"] == v["job_id"]].empty:
                    count_non_empty += 1
                    new_jobs_list.append([v["job_id"], v["score"]])

            # if number of non empty jobs < 2 or top non empty score < 3 discard candidate
            if len(new_jobs_list) < 3 or new_jobs_list[0][1] < 3 or count_non_empty < 3:
                continue
            else:
                jobs_list = list(zip(*new_jobs_list))[0]
                jobs_list = discard_non_relevant(jobs_list, stop_words, w2v_model)
                if len(jobs_list) < 3:
                    continue
        
                if len(test_dict["cand_id"]) <= 12000:
                    test_dict["cand_id"].append(cand_id)
                    test_dict["job_list"].append(jobs_list)
                    pass
                else:
                    train_dict["cand_id"].append(cand_id)
                    train_dict["job_list"].append(jobs_list)

                
            if index%10000 == 0:
                print("index:",index)
                print("train",len(train_dict["cand_id"]))
                print("test", len(test_dict["cand_id"]))
                print(time_list/num_of_loops)

        if len(train_dict["cand_id"]) >= 300000 and len(test_dict["cand_id"]) > 12000:
            break


    print("num_of_loops:",num_of_loops)
    print(time_list/num_of_loops)
    train_df = pd.DataFrame.from_dict(train_dict)
    test_df = pd.DataFrame.from_dict(test_dict)
    return train_df, test_df ,train_dict, test_dict



#read candidate files
path_to_cand_file = 'C:/Users/nataz/Downloads/job_prop_data/data/candidate/sequences/sequences~/sequences'
cand_df = read_cand_df(path_to_cand_file)

#read jobs dataframe
path_to_jobs_file = "C:/Users/nataz/Downloads/job_prop_data/dataset/final_data/jobs_df_v1"
jobs_df = read_df(path_to_jobs_file)

#First data processing phase
train_df, test_df, train_dict, test_dict = process_candidates(cand_df, jobs_df)

#Save results
pickling_on = open("C:/Users/nataz/Downloads/job_prop_data/dataset/final_data/train_df", 'wb')
joblib.dump(train_df, pickling_on)
pickling_on.close()

pickling_on = open("C:/Users/nataz/Downloads/job_prop_data/dataset/final_data/test_df", 'wb')
joblib.dump(test_df, pickling_on)
pickling_on.close()
