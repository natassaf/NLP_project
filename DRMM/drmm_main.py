from collections import Counter
#import linecache2
import os
import json
import numpy as np
import pandas as pd
import joblib
import datus_drmm
from gensim.models import Word2Vec
from gensim.models import KeyedVectors
from operator import itemgetter
import tracemalloc
from tqdm import tqdm
import time
import matplotlib.pyplot as plt

import drmm_model_drop_out as drmm_model

#configuration for GPU
import dynet_config
dynet_config.set_gpu()
dynet_config.set(mem=6000,random_seed = 2)
import dynet as dy

#Configuration for CPU to expand memory  
'''
import _dynet as dy
dyparams = dy.DynetParams()
dyparams.from_args()
dyparams.set_mem(6000)
dyparams.set_random_seed(666)
dyparams.init()
'''

  


def read_dataframe(path):
    pickling_on = open(path, 'rb')
    jobs_df = joblib.load(pickling_on)
    pickling_on.close()
    return jobs_df

def chunks(l, n, s=0):
    chunks_list = []
    for i in range(s, len(l), n):
        chunks_list.append(l[i:i + n])
    return chunks_list

def calculate_precision_per_query(predicted, rel_docs):
    score = 0.0
    num_hits = 0.0
    num_of_rel_docs = 0
    for i ,p in enumerate(predicted):
        if p in rel_docs:
            num_hits += 1.0
            score += num_hits / ( i +1.0)
            num_of_rel_docs += 1
    if num_of_rel_docs == 0 :
        return 0.0
    return score / num_of_rel_docs


def save_dataframe(jobs_df, path):
    """
    Save preprocessed dataframe
    """
    pickling_on = open(path, 'wb')
    joblib.dump(jobs_df, pickling_on)
    pickling_on.close()


def rerank(dev_data, drmm_model, jobs_df, excluding_set, w2v_model,query_doc_df, flag):  # takes all dev data
    average_precision_at_10 = 0
    query_doc_data = {"cand_id":[],"doc_id":[],"histogram":[],"overlapping_features":[]}
    drmm_data_obj = datus_drmm.DRMM_DATA(dev_data, jobs_df, excluding_set, w2v_model)
    check_manually = {"cand_id":[],"drmm_top_10":[],"bm25_top_10":[]}
    for index, row in tqdm(dev_data.iterrows()):
        pos_set_list = dev_data.loc[index, "positive_doc_ids"]
        cand_id = dev_data.loc[index, "cand_id"]

        query = dev_data.loc[index, "query_tokens"]
        query_bigrams = drmm_data_obj.get_bigrams(query)
        docs_to_rerank = dev_data.loc[index, "positive_doc_ids"].tolist() # + dev_data.loc[index, "negative_doc_ids"]
        #print("docs_to_rerank_len",len(docs_to_rerank))
        docs_BM25 = dev_data.loc[index, "positive_doc_BM25"].tolist() #+ dev_data.loc[index, "negative_doc_BM25"]
        docs_scores = []
        for i in range(len(docs_to_rerank)):
            dy.renew_cg()
            #drmm_data_obj = datus_drmm.DRMM_DATA(dev_data, jobs_df, excluding_set, w2v_model)
            doc = docs_to_rerank[i]
            d_BM25 = docs_BM25[i]
            if flag:
                histogram = query_doc_df["histogram"][(query_doc_df["cand_id"] == cand_id) & (query_doc_df["doc_id"] ==  doc)].tolist()[0]
                overlapping_features = query_doc_df["overlapping_features"][(query_doc_df["cand_id"] == cand_id) & (query_doc_df["doc_id"] == doc)].tolist()[0]
            else:
                histogram, overlapping_features = drmm_data_obj.get_drmm_data_new(index, doc, query, query_bigrams)
                query_doc_data["cand_id"].append(cand_id)
                query_doc_data["doc_id"].append(doc)
                query_doc_data["histogram"].append(histogram)
                query_doc_data["overlapping_features"].append(overlapping_features)
                              
           
            docs_scores.append(drmm_model.predict_doc_score(histogram, dev_data.loc[index, "query_idf"], d_BM25, overlapping_features).value())

        query_results = list(zip(docs_to_rerank, docs_scores))
        query_results.sort(key=itemgetter(1), reverse=True)
        top_10 = list(zip(*query_results))[0][:10]
        BM25_top_10 =  dev_data.loc[index, "positive_doc_ids"].tolist()[:10] 
        check_manually["cand_id"].append(cand_id)
        check_manually["drmm_top_10"].append(top_10)
        check_manually["bm25_top_10"].append(BM25_top_10)
        average_precision_at_10 += calculate_precision_per_query(top_10, pos_set_list)

   
    if flag == False:
      query_doc_df = pd.DataFrame.from_dict(query_doc_data)
      #save_dataframe(query_doc_df, path = "dataset/results/query_doc_df")
      
    flag = True
    return average_precision_at_10/len(dev_data), query_doc_df, flag, check_manually


def tune_DRMM(train_pairs, tuning_data_pairs, tuning_data, jobs_df, excluding_set, w2v_model, train_batch_size = 128, n_epochs = 10, mlp_layers=5, hidden_size=10):
    """
    This function is used to tune the hyperparameters hidden size and number of layers using a "tuning dataset"
    :return best number of mlp layers, best number of units per layer
    """

    mlp_layers_list = [3, 5, 8]
    nodes_per_layer = [10,20]
    metrics_dict = {"hidden_size":[], "mlp_layers":[], "train_accuracy":[],"dev_pairs_accuracy":[], "best_map":[]}

    for layer in tqdm(mlp_layers_list,position=1):
        for hidden_size in nodes_per_layer:
            drmm_mod = drmm_model.DRMM(mlp_layers, hidden_size)
            print("mlp_layers:", layer,"\n","hidden_size:",hidden_size)
            metrics_dict["mlp_layers"].append(layer)
            metrics_dict["hidden_size"].append(hidden_size)
            dev_accuracy_prev = 0.0
            train_shuffled = train_pairs.copy()
            best_map = -1
            for epoch in range(1, n_epochs+1):
                print('\nEpoch: {0}/{1}'.format(epoch, n_epochs))
                sum_of_losses = 0
                train_shuffled = train_shuffled.sample(frac=1)
                train_batches = chunks(range(len(train_shuffled['cand_id'])), train_batch_size)
                hits = 0
                for batch in train_batches:
                    dy.renew_cg()  # new computation graph
                    batch_losses = []
                    batch_preds = []
                    for i in batch:
                        q_dpos_hist = train_shuffled.loc[i, 'pos_histogram']
                        q_dneg_hist = train_shuffled.loc[i, 'neg_histogram']
                        query_idf = train_shuffled.loc[i,'query_idf']
                        pos_bm25 = train_shuffled.loc[i, 'pos_normBM25'][0]
                        neg_bm25 = train_shuffled.loc[i, 'neg_normBM25'][0]
                        pos_uni_overlap = train_shuffled.loc[i, 'overlapping_unigrams_pos']
                        pos_bi_overlap = train_shuffled.loc[i, 'overlapping_bigrams_pos']
                        pos_overlap_features = [pos_uni_overlap,pos_bi_overlap]
                        neg_uni_overlap = train_shuffled.loc[i, 'overlapping_unigrams_neg']
                        neg_bi_overlap = train_shuffled.loc[i, 'overlapping_bigrams_neg']
                        neg_overlap_features = [neg_uni_overlap, neg_bi_overlap]
                        preds = drmm_mod.predict_pos_neg_scores(q_dpos_hist, q_dneg_hist, query_idf, pos_bm25, neg_bm25, pos_overlap_features,neg_overlap_features)
                        batch_preds.append(preds)
                        loss = dy.hinge(preds, 0)
                        batch_losses.append(loss)
                    batch_loss = dy.esum(batch_losses)/len(batch)
                    #print(float(batch_loss.npvalue())/len(batch))
                    sum_of_losses += float(batch_loss.npvalue()[0])
                    for p in batch_preds:
                        p_v = p.value()
                        if p_v[0] > p_v[1]:
                            hits += 1
                    batch_loss.backward()
                    drmm_mod.trainer.update() # this calls forward on the batch

                train_acc = hits / train_shuffled.shape[0]

                val_preds = []
                val_losses = []
                hits = 0
                dy.renew_cg()
                for i,row in tuning_data_pairs.iterrows():
                    q_dpos_hist = tuning_data_pairs.loc[i, 'pos_histogram']
                    q_dneg_hist = tuning_data_pairs.loc[i, 'neg_histogram']
                    query_idf = tuning_data_pairs.loc[i, 'query_idf']
                    pos_bm25 = tuning_data_pairs.loc[i, 'pos_normBM25'][0]
                    neg_bm25 = tuning_data_pairs.loc[i, 'neg_normBM25'][0]
                    pos_uni_overlap = tuning_data_pairs.loc[i, 'overlapping_unigrams_pos']
                    pos_bi_overlap = tuning_data_pairs.loc[i, 'overlapping_bigrams_pos']
                    pos_overlap_features = [pos_uni_overlap, pos_bi_overlap]
                    neg_uni_overlap = tuning_data_pairs.loc[i, 'overlapping_unigrams_neg']
                    neg_bi_overlap = tuning_data_pairs.loc[i, 'overlapping_bigrams_neg']
                    neg_overlap_features = [neg_uni_overlap, neg_bi_overlap]
                    preds_dev = drmm_mod.predict_pos_neg_scores(q_dpos_hist, q_dneg_hist, query_idf, pos_bm25, neg_bm25, pos_overlap_features, neg_overlap_features)
                    val_preds.append(preds_dev)
                    loss = dy.hinge(preds_dev,0)
                    val_losses.append(loss)
                val_loss = dy.esum(val_losses)
                sum_of_losses += val_loss.npvalue()[0] # this calls forward on the batch
                for p in val_preds:
                    p_v = p.value()
                    if p_v[0] > p_v[1]:
                        hits += 1

                dev_accuracy = hits / tuning_data_pairs.shape[0]



                print('Training acc: {0}'.format(train_acc))
                print('Dev acc: {0}'.format(dev_accuracy))

                if train_acc < 0.6:
                    continue

                map_dev, query_doc_data, flag = rerank(tuning_data, drmm_mod,jobs_df, excluding_set, w2v_model, query_doc_data, flag)
                print("map_dev",map_dev)

                if map_dev > best_map:
                    print('===== Best epoch so far =====')
                    best_map = map_dev
                    best_epoch = epoch
                    best_train_accuracy = train_acc
                    best_dev_accuracy = dev_accuracy
                    #drmm_mod.dump_weights("C:/Users/nataz/Downloads/job_prop_data/dataset2")

            if dev_accuracy - dev_accuracy_prev <= 0.005:
                break

            dev_accuracy_prev = dev_accuracy

            metrics_dict["train_accuracy"].append(train_acc)
            metrics_dict["dev_pairs_accuracy"].append(dev_accuracy)
            metrics_dict["best_map"].append(best_map)

    metrics_df = pd.DataFrame.from_dict(metrics_dict)
    max_map = metrics_df["best_map"].max()

    best_mlp_layer = metrics_df["mlp_layers"][metrics_df["best_map"] == max_map].tolist()[0]
    best_hidden_size = metrics_df["hidden_size"][metrics_df["best_map"] == max_map].tolist()[0]
    save_dataframe(metrics_df, path="dataset/results/metrics_tuning")

    return best_mlp_layer, best_hidden_size
                  
def train_DRMM(train_pairs, dev_data_pairs, dev_data, jobs_df, excluding_set, w2v_model, train_batch_size, n_epochs, mlp_layers=5, hidden_size=10, p=1):
    """
    This function trains the mlp of DRMM by
    :param train_pairs: triplets of (query, positive doc, negative doc) used for training the model
    :param dev_data_pairs:  triplets of (query, positive doc, negative doc) used to evaluate models performance
    :param dev_data: dict containing query_tokens, query idfs, 50 positive document ids, 50 negative document ids, 50 positive documents BM25 scores, 50 negative docs BM25 scores
    :param jobs_df: pandas dataframe containing jobs dataset
    :param excluding_set: tokens with small idf to exclude
    :param w2v_model:  gensim word2vec
    :param train_batch_size: batch size of data to backpropagate
    :param p: probability used to apply dropout
    :return: dictionary containing data for learning curve
    """

    #create an object of classs DRMM
    drmm_mod = drmm_model.DRMM(mlp_layers, hidden_size)

    #load pretrained weights of the MLP layer
    drmm_mod.load_weights("dataset/results/res_no_bm25/no_unigrams")

    train_size_list = np.arange(0, len(train_pairs) + 1, 9000)

    #dictionary containing data needed to construct a learning curve
    learning_curve_data = {"num_of_data":train_size_list[1:],"train_accuracy":[],
                           "dev_pairs_accuracy":[], "map_on_test_set":[]}

    query_doc_df = {}
    flag = False

    for t in range(len(train_size_list)-1):
        print(train_size_list[t],train_size_list[t+1])

        train_subset = train_pairs.iloc[train_size_list[t]:train_size_list[t+1]]
        print(len(train_subset))
        best_map = -1
        dev_accuracy_prev = 0.0
        for epoch in range(1, n_epochs+1):
            print('\nEpoch: {0}/{1}'.format(epoch, n_epochs))
            sum_of_losses = 0
            train_subset = train_subset.sample(frac=1)
            train_batches = chunks(range(train_size_list[t],train_size_list[t+1]), train_batch_size)
            hits = 0
            for batch in train_batches:
                dy.renew_cg()  # new computation graph
                batch_losses = []
                batch_preds = []
                for i in batch:
                    q_dpos_hist = train_subset.loc[i, 'pos_histogram']
                    q_dneg_hist = train_subset.loc[i, 'neg_histogram']
                    query_idf = train_subset.loc[i,'query_idf']
                    pos_bm25 = train_subset.loc[i, 'pos_normBM25'][0]
                    neg_bm25 = train_subset.loc[i, 'neg_normBM25'][0]
                    pos_uni_overlap = train_subset.loc[i, 'overlapping_unigrams_pos']
                    pos_bi_overlap = train_subset.loc[i, 'overlapping_bigrams_pos']
                    pos_overlap_features = [pos_uni_overlap,pos_bi_overlap]
                    neg_uni_overlap = train_subset.loc[i, 'overlapping_unigrams_neg']
                    neg_bi_overlap = train_subset.loc[i, 'overlapping_bigrams_neg']
                    neg_overlap_features = [neg_uni_overlap, neg_bi_overlap]
                    preds = drmm_mod.predict_pos_neg_scores(q_dpos_hist, q_dneg_hist, query_idf, pos_bm25, neg_bm25, pos_overlap_features,neg_overlap_features, p)
                    batch_preds.append(preds)
                    loss = dy.hinge(preds, 0)
                    batch_losses.append(loss)
                batch_loss = dy.esum(batch_losses)/len(batch)
                #print(float(batch_loss.npvalue())/len(batch))
                sum_of_losses += float(batch_loss.npvalue()[0])
                for p in batch_preds:
                    p_v = p.value()
                    if p_v[0] > p_v[1]:
                        hits += 1
                batch_loss.backward()
                drmm_mod.trainer.update() # this calls forward on the batch

            train_acc = hits / train_subset.shape[0]

            val_preds = []
            val_losses = []
            hits = 0
            dy.renew_cg()
            for i,row in dev_data_pairs.iterrows():
                q_dpos_hist = dev_data_pairs.loc[i, 'pos_histogram']
                q_dneg_hist = dev_data_pairs.loc[i, 'neg_histogram']
                query_idf = dev_data_pairs.loc[i, 'query_idf']
                pos_bm25 = dev_data_pairs.loc[i, 'pos_normBM25'][0]
                neg_bm25 = dev_data_pairs.loc[i, 'neg_normBM25'][0]
                pos_uni_overlap = dev_data_pairs.loc[i, 'overlapping_unigrams_pos']
                pos_bi_overlap = dev_data_pairs.loc[i, 'overlapping_bigrams_pos']
                pos_overlap_features = [pos_uni_overlap, pos_bi_overlap]
                neg_uni_overlap = dev_data_pairs.loc[i, 'overlapping_unigrams_neg']
                neg_bi_overlap = dev_data_pairs.loc[i, 'overlapping_bigrams_neg']
                neg_overlap_features = [neg_uni_overlap, neg_bi_overlap]
                preds_dev = drmm_mod.predict_pos_neg_scores(q_dpos_hist, q_dneg_hist, query_idf, pos_bm25, neg_bm25, pos_overlap_features, neg_overlap_features, p = 1)
                val_preds.append(preds_dev)
                loss = dy.hinge(preds_dev,0)
                val_losses.append(loss)
            val_loss = dy.esum(val_losses)
            sum_of_losses += val_loss.npvalue()[0] # this calls forward on the batch
            for p in val_preds:
                p_v = p.value()
                if p_v[0] > p_v[1]:
                    hits += 1

            dev_accuracy = hits / dev_data_pairs.shape[0]

            print('\nTraining acc: {0}'.format(train_acc))
            print('Dev acc: {0}'.format(dev_accuracy))
            
            if dev_accuracy - dev_accuracy_prev <=0.008 or epoch==50 or dev_accuracy>0.9:
                print(dev_accuracy - dev_accuracy_prev)
                map_dev, query_doc_df, flag, check_manually  = rerank(dev_data, drmm_mod, jobs_df, excluding_set, w2v_model, query_doc_df, flag)
                print("map_dev",map_dev)
                best_map = map_dev
                best_train_accuracy = train_acc
                best_dev_accuracy = dev_accuracy
                drmm_mod.dump_weights("dataset/results/res_dropout")
                break
            
            #if dev_accuracy >= 0.85: #early stop
            #    break
                
            dev_accuracy_prev = dev_accuracy
        metrics_results = [best_map, best_train_accuracy, best_dev_accuracy]
             
        learning_curve_data["train_accuracy"].append(best_train_accuracy)
        learning_curve_data["dev_pairs_accuracy"].append(best_dev_accuracy)
        learning_curve_data["map_on_test_set"].append(best_map)
        
        save_dataframe(check_manually, path = "dataset/results/res_dropout/check_manually_dev" + str(train_size_list[t+1]))     
    #save_dataframe(metrics_results, path = "dataset/results/res_dropout_mlp/metrics_results_dev")    
    save_dataframe(learning_curve_data, path = "dataset/results/res_dropout/learning_curve_data_dict")

    return learning_curve_data

def test_DRMM(test_pairs, test_data, jobs_df, excluding_set, w2v_model, mlp_layers=10, hidden_size=10):
    """
    Calculates Mean average precision of the trained model using a test set
    :return: Mean average precision on test set
    """
    drmm_mod = drmm_model.DRMM(mlp_layers, hidden_size)
    val_preds = []
    val_losses = []
    hits = 0
    dy.renew_cg()
    sum_of_losses = 0
    query_doc_df = {}
    flag = False
    drmm_mod.load_weights("dataset/results/res_mlp")
    
    learning_curve_data = { "map_on_test_set":[]}

    map_test, query_doc_df, flag, check_manually  = rerank(test_data, drmm_mod, jobs_df, excluding_set, w2v_model, query_doc_df, flag)
    
    print("map_dev",map_testt)
    
    #learning_curve_data["dev_pairs_accuracy"].append(dev_accuracy)
    learning_curve_data["map_on_test_set"].append(map_test)
        
    save_dataframe(check_manually, path = "dataset/results/res_mlp/check_manually_test")    
    save_dataframe(learning_curve_data, path = "dataset/results/res_mlp/metrics_test")

    return map_test

def learning_curves(results, method="DRMM"):
    """
    :param results: dictionary  contaning a list of accuracies on training set, a list of accuracies on dev set and MAP on dev set per training subset
    :return: Print learning curve
    """
    fontP = FontProperties()
    fontP.set_size('small')
    fig = plt.figure()
    fig.suptitle('Learning Curves - ' + method, fontsize=17)
    ax = fig.add_subplot(111)
    ax.axis([300, 3800, 0, 1])


    line_up, = ax.plot(results['num_of_data'][1:], results["train_accuracy"],
                       'o-', label='Accuracy on Train')
    line_down, = ax.plot(results['num_of_data'][1:], results['map_on_test_set'],
                         'o-', label='Accuracy on Test')
    plt.ylabel('Accuracy', fontsize=13)

    plt.legend([line_up, line_down], ['Accuracy on Train', 'Accuracy on Test'],
               prop=fontP)


    plt.xlabel('Number of training instances', fontsize=13)
    plt.grid(True)
    plt.show()



#read data
train_pairs = read_dataframe("dataset/train_df_all")
train_pairs.reset_index(inplace = True)
jobs_df = read_dataframe("dataset/jobs_df_v1")
w2v_model = KeyedVectors.load("dataset/w2v_model.w2v", mmap='r')
excluding_set = read_dataframe("dataset/set_tokens_less_10")
p = 0.2

#read tuning pairs and data
tuning_pairs = read_dataframe("dataset/tuning_pairs")
tuning_data =  read_dataframe("dataset/test_sets/tune_MAP")

#tune DRMM
best_mlp_layer, best_hidden_size = tune_DRMM(train_pairs, tuning_pairs, tuning_data, jobs_df, excluding_set, w2v_model, 256, 5)


dev_pairs = read_dataframe("dataset/dev_pairs")
dev_data =  read_dataframe("dataset/test_sets/dev_MAP")

#train DRMM
learning_curve_data = train_DRMM(train_pairs, dev_pairs, dev_data, jobs_df, excluding_set, w2v_model, 256, 50, mlp_layers=best_mlp_layer, hidden_size=best_hidden_size, p)



test_pairs = read_dataframe("dataset/test_pairs")
test_data =  read_dataframe("dataset/test_sets//test_manual_precision/test_MAP")

test_DRMM(test_pairs, test_data, jobs_df, excluding_set, w2v_model , mlp_layers=10, hidden_size=10)

learning_curves(learning_curve_data, method="DRMM")

