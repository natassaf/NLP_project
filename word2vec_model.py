#This script is used to train gensims word2vec model with the words of our database.
import gensim
import joblib
import string
from gensim.models import Word2Vec
from nltk.corpus import stopwords
from nltk import download
import nltk
download('stopwords')

def create_file_for_word2vec(csv_path):
    """
    Create csv_file that will be used to train word to vec
    :param csv_path: path where preprocessed csv file will stored
    :return: the words from the keywords field that have a special format.
    """
    pickling_on = open("C:/Users/nataz/Downloads/job_prop_data/dataset/jobs_df", 'rb')
    jobs_df = joblib.load(pickling_on)
    pickling_on.close()


    keywords_sent = jobs_df["keywords"].tolist()
  
    #create csv using these particular columns to extract info. Columns id, description and created_at are not used
    jobs_df[['function', 'title', 'requirement_summary']] .to_csv(csv_path, sep='\n', encoding='utf-8-sig', header=False, index=False)
    
    return keywords_sent

def tokenize_jobs(csv_path):
    sentences_tokenized=[]
    with open(csv_path, encoding="utf8") as file:
        sentences = file.read()

    sentences = sentences.split("\n")
    for line in sentences:
        tokens = line.split()
        if len(tokens) > 1:
            stop_words = stopwords.words('english')
            tokens_stop = [w for w in tokens if w not in stop_words]
            sentences_tokenized.append(tokens_stop)
    return sentences_tokenized

def create_word2Vec(tok_corp , save_path, seed=2018, num_workers=4, num_features=150, min_word_count=1, context_size=5):
    """
    :param tok_corp: list of tokenized sentences
    :param save_path: path where word2vec model will be saved
    :param seed: random variable
    :param num_workers: processors to be used
    :param num_features: number of features to be extracted
    :return: word2vec model
    """
    word2vec_model = gensim.models.word2vec.Word2Vec(sentences=tok_corp,sg=0,seed=seed,workers=num_workers,size=num_features,min_count=min_word_count,window=context_size)
    #word2vec_model.train(sentences = tok_corp,total_examples=corpus_count, epochs=5)
    word2vec_model.save(save_path)

    return word2vec_model

def tune_word2vec(sentences):
    """
    Tune the hyperparameters of word2vec model by creating a model for each version.
    """
    num_features_list = [100,150,180,200,250]
    cont_size_list = [5,8,10]

    for features in num_features_list:
        for window in cont_size_list:
            path_for_w2v = "C:/Users/nataz/Downloads/job_prop_data/dataset/word2vec/w2v_model_" + str(features) + str(window) + ".w2v"
            word2vec_model = create_word2Vec(sentences, path_for_w2v, num_features=features, context_size=window)


store_csv_path = "C:/Users/nataz/Downloads/job_prop_data/dataset/word2vec/job_data_for_word2vec.csv"
keywords_sent = create_file_for_word2vec(store_csv_path)
sentences = tokenize_jobs(store_csv_path)
sentences = sentences + keywords_sent
tune_word2vec(sentences)


window = 5
features = 100
path_for_w2v = "C:/Users/nataz/Downloads/job_prop_data/dataset/word2vec/final_w2v/w2v_model.w2v"
word2vec_model = create_word2Vec(sentences, path_for_w2v, num_features=features, context_size=window)
