# This script processes the job announcements that will be fed into Elasticsearch engine. We strip html tags and turn letters to lowercase and remove punctuation

import pandas as pd
import numpy as np
import re
import joblib
import glob
import os, json
import string
from string import digits

def clean_data(sent_list:list)->list:
    """
    turn string to lower case and remove punctuation
    :param sent_str: text that we want to preprocess
    :return: preprocessed text
    """
    sentences = []
    for sent in sent_list:
        if sent is None:
            sentences.append("")
            continue

        sentence = str(sent).lower()  # turn to lowercase
        x = ' '.join(''.join([ch if ch.isspace() or ch.isalpha() else ' ' for ch in sentence]).split())
        sentences.append(x)
    return sentences

#preprocessing of jobs file (datasets)
def strip_html_tags(file:str)->str:
    """
    Strip HTML tags
    """
    file = re.sub("<.*?>"," ", file)
    #file = re.sub('{\"id\"', '\n{\"id\"', file)
    return file

def save_dataframe(jobs_df, path="C:/Users/nataz/Downloads/job_prop_data/dataset/final_data/jobs_df_v1"):
    """
    Save preprocessed dataframe
    """
    pickling_on = open(path, 'wb')
    joblib.dump(jobs_df, pickling_on)
    pickling_on.close()



def process_jobs(jobs_df):
    """
    Preprocess jobs data and get keywords field as list of tokens instead of json
    """
    keywords_bow = []
    for col in ["keywords", 'title', 'function', 'requirement_summary', "description"]:
        if col == "keywords":
            for index, row in jobs_df.iterrows():
                if row["keywords"] == [] or row["keywords"] is None or row["keywords"] == {}:
                    keywords_bow.append([])
                else:
                    name_pd = pd.DataFrame(jobs_df["keywords"].tolist()[index])
                    keywords_bow.append(" ".join(name_pd["name"].tolist()).split())
            jobs_df["keywords"] = pd.Series(keywords_bow)
        else:
            jobs_df[col] = pd.Series(clean_data(jobs_df[col].tolist()))

    return jobs_df


def get_json_filenames():
    path_to_json = 'C:/Users/nataz/Downloads/job_prop_data/data/jobs/datasets~/datasets'
    json_files = [pos_json for pos_json in os.listdir(path_to_json) if pos_json.endswith('.json')]

    json_list = []
    for index, js in enumerate(json_files):
        with open(os.path.join(path_to_json, js)) as json_file:
            json_list.append(json_file.read())
    return json_list


if __name__ == "__main__":

    json_list = get_json_filenames()
    jobs_json = []

    for i in range(len(json_list)):
        json_list[i] = strip_html_tags(json_list[i])
        jobs_json.extend(json.loads(json_list[i]))

    jobs_df = pd.DataFrame(jobs_json)
    jobs_df = process_jobs(jobs_df)

    save_dataframe(jobs_df)


