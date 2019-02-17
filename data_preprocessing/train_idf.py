#script used to calculate the inverse document frequency of every word in the corpus. IDF is necessary for our deep learning model(DRMM)

import joblib
import string
import time
import nltk
import math
import operator
import itertools
import collections


def read_jobs(path):
    """
    :param path: path to jobs pandas dataframe
    :return: pandas DataFrame
    """
    with open(path, 'rb') as data_file:
        jobs_df = joblib.load(data_file)

    return jobs_df




def calculate_idf(tokens, jobs_df, common_words):
    '''
    This function calculates the idf of each word in the corpus using sets.
    :tokens: tokens of all words of the contained in our database of job announcements
    :jobs_df: job announcements database in pandas dataframe format
    :common_words: common words of the english dictionary to discard
    '''
    
    fdist = nltk.FreqDist(tokens)
    tokens_less10 = [k for k, v in fdist.items() if v <= 5]
    cm = set(common_words[:40])
    vocabulary = set(tokens) - set(tokens_less10) - cm


    docs_with_word = {}
    idf = {}
    
    all_docs = jobs_df.shape[0]


    for i, row in jobs_df.iterrows():
        req_sum = jobs_df.loc[i, 'requirement_summary'].split()
        title = jobs_df.loc[i, 'title'].split()
        function = jobs_df.loc[i, 'function'].split()
        description = jobs_df.loc[i, 'description'].split()
        tokens_doc = set(req_sum + title + function)


        if tokens_doc == []:
            continue

        words_intersection = vocabulary.intersection(tokens_doc)


        for w in words_intersection:
            try:
                docs_with_word[w] += 1
            except:
                docs_with_word[w] = 1


    for v in vocabulary:
        try:
            idf[v] = math.log(all_docs/docs_with_word[v])
        except:
            idf[v] = 0.01
    
    return idf


    
with open ("C:/Users/nataz/PycharmProjects/capstone_final/common_words", 'r') as file:
    common_words = file.read().splitlines()

jobs_df = read_jobs("C:/Users/nataz/Downloads/job_prop_data/dataset/final_data/jobs_df_v1")
tokens = list(itertools.chain(*jobs_df["keywords"].tolist())) + (" ".join(jobs_df["requirement_summary"].tolist()) + " ".join(jobs_df["function"].tolist()) + " ".join(jobs_df["title"].tolist())).split()
fdist = nltk.FreqDist(tokens)
tokens_less10 = [k for k, v in fdist.items() if v <= 5]
cm = set(common_words[:40])   
idf = calculate_idf(tokens, jobs_df, common_words)    
  
pickling_on = open("C:/Users/nataz/Downloads/job_prop_data/dataset/final_data/idf","wb")
joblib.dump(idf ,pickling_on)
pickling_on.close()

pickling_on = open("C:/Users/nataz/Downloads/job_prop_data/dataset/final_data/set_tokens_less_10","wb")
joblib.dump(set(tokens_less10) ,pickling_on)
pickling_on.close()
