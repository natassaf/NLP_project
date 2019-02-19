

import numpy as np
from nltk.corpus import stopwords
import nltk
# nltk.download('stopwords')
from sklearn.metrics.pairwise import cosine_similarity

class DRMM_DATA:    
    """
    This class is used in the drmm_main/def reranking to calculate the features needed to calculate DRMM score(histogram,overlapping unigrams,bigrams etc).
    Given the query tokens and the top 100 relevant docs for it, it returns the DRMM features between each query - document pair.
    """
    def __init__(self, data_df, jobs_df, excluding_set, w2v_model):
        self.data_df = data_df
        self.jobs_df = jobs_df
        self.excluding_set = excluding_set
        self.w2v_model = w2v_model
        self.imp_fields = ["keywords", "title", "function", "requirement_summary"]

    def get_bow(self, job_id):

        bow = []
        stop_words = set(stopwords.words('english') + stopwords.words('french') +stopwords.words('greek'))
        job = self.jobs_df[self.jobs_df["id"] == job_id]
        for col in self.imp_fields:
            if col == "keywords":
                bow.extend(job[col].tolist()[0])
            elif col == "title" or col == "function":
                bow.extend(job[col].tolist()[0].split())
            elif col == "requirement_summary":
                bow.extend(job[col].tolist()[0].split())
        bow_stop = [w for w in bow if w not in stop_words]

        return bow_stop


    def doc_preprocessing(self, d_tokens):

        doc_tokens = [d for d in d_tokens if d not in self.excluding_set]

        return doc_tokens

    def generate_log_countbased_histogram(self, query, doc):
        q_vector = []
        for q_term in query:
            try:
                q_vector.append(self.w2v_model.wv[q_term])
            except KeyError:
                q_vector.append(list(np.random.randn(100, )))

        doc_vector = []
        if len(doc) == 0:
          doc_vector.append(list(np.random.randn(100, )))
        else:
          for d_term in doc:
              try:
                  doc_vector.append(self.w2v_model.wv[d_term])
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

    def get_overlapping_ngrams(self,query_ngrams, doc_ngrams):
        query_bigrams_set = set(query_ngrams)
        doc_bigrams_set = set(doc_ngrams)
        overlap = query_bigrams_set.intersection(doc_bigrams_set)

        return len(overlap)

    def get_bigrams(self,tokens):
        bigrams = []
        for i in range(len(tokens ) -1):
            bigrams.append(" ".join([tokens[i] ,tokens[ i +1]]))
        return bigrams

  
    def get_drmm_data(self, index, doc_id, query, query_bigrams):
        overlapping_features = []
        histograms = []
        doc_bow = self.get_bow(doc_id)
        doc_bigrams = self.get_bigrams(doc_bow)
        uni_overlap = self.get_overlapping_ngrams(query, doc_bow)
        bi_overlap = self.get_overlapping_ngrams(query_bigrams, doc_bigrams)
        doc_tokens = self.doc_preprocessing(doc_bow)
        histogram = self.generate_log_countbased_histogram(query ,doc_tokens)
        overlapping_features = [uni_overlap, bi_overlap]
        return histogram, overlapping_features
