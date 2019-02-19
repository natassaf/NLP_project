#Deep Relevance Matching Model Implementation on dynet tool

import dynet as dy
import numpy as np
from gensim.models import KeyedVectors
#import _dynet as dy


class DRMM:

    def __init__(self, mlp_layers=5, hidden_size = 10):

        # Input hyperparameters
        self.hist_size = 30

        # MLP hyperparameters
        self.mlp_layers = mlp_layers
        self.hidden_size = hidden_size

        # Model initialization
        self.model = dy.ParameterCollection()
        self.trainer = dy.AdamTrainer(self.model)

        self.w_g = self.model.add_parameters((1,))
        #self.w_g.set_updated(False)
        
        self.W_1 = self.model.add_parameters((self.hist_size, self.hidden_size))
        #self.W_1.set_updated(False)
        self.b_1 = self.model.add_parameters((1, self.hidden_size))
        #self.b_1.set_updated(False)
        
        if self.mlp_layers > 1:
            self.W_n = []
            self.b_n = []
            for i in range(self.mlp_layers):
                self.W_n.append(self.model.add_parameters((self.hidden_size, self.hidden_size)))
                self.b_n.append(self.model.add_parameters((1, self.hidden_size)))
            '''   
            for i in range(self.mlp_layers):
                self.W_n[i].set_updated(False)
                self.b_n[i].set_updated(False)
            '''
            
    
        self.W_last = self.model.add_parameters((self.hidden_size, 1))
        #self.W_last.set_updated(False)
        self.b_last = self.model.add_parameters((1))
        #self.b_last.set_updated(False)
        
        self.W_bm25 = self.model.add_parameters((1, 1))
        #self.W_bm25.set_updated(False)
        self.b_bm25 = self.model.add_parameters((1))
        #self.b_bm25.set_updated(False)

        self.W_scores = self.model.add_parameters((3, 1))
        #self.W_scores.set_updated(False)
        self.b_scores = self.model.add_parameters((1))
        #self.b_scores.set_updated(False)
        
    def dump_weights(self, w_dir):
        self.model.save(w_dir + '/weights.bin')

    def load_weights(self, w_dir):
        self.model.populate(w_dir + '/weights.bin')

    def predict_pos_neg_scores(self, q_dpos_hists, q_dneg_hists, q_idf, pos_bm25_score, neg_bm25_score, pos_overlap_features, neg_overlap_features, p):
        """
        This function is used to calculate the DRMM relavant score for a positive and a negative doc of a query and thus train the model.
        :param q_dpos_hists: vector of integer numbers representing the histograms extracted using the query and the positive document
        :param q_dneg_hists: vector of integer numbers representing the histograms extracted using the query and the negative document
        :param q_idf: dictionary with query terms as keys and inverse document frequency as values
        :param pos_bm25_score:  bm25 score(float)of the positive document extracted by Elasticsearch platform results
        :param neg_bm25_score:  bm25 score(float) of the negative document extracted by Elasticsearch platform results
        :param pos_overlap_features: integer number of common unigrams and bigrams between the query and the positive documents
        :param neg_overlap_features: integer number of common unigrams and bigrams between the query and the negative documents
        :param p: probability of keeping a unit active used for the dropout.
        :return: list of DRMM score for the positive document and for the negative document
        """

        pos_score = self.scorer(q_dpos_hists, q_idf, pos_bm25_score, pos_overlap_features, p)
        neg_score = self.scorer(q_dneg_hists, q_idf, neg_bm25_score, neg_overlap_features, p)

        # return probability of first (relevant) document.
        return dy.concatenate([pos_score, neg_score])


    def predict_doc_score(self, q_d_hists, q_idf, bm25_score, overlap_features):
        """
        Calculates DRMM relevance score given a query and a document
        """
        doc_score = self.scorer(q_d_hists, q_idf, bm25_score,  overlap_features,p=1)
        return doc_score

    def scorer(self, q_d_hists, q_idf, bm25_score, overlap_features, p):
        """
        Makes all the calculations and returns a relevance score
        """
        idf_vec = dy.inputVector(q_idf)
        bm25_score = dy.scalarInput(bm25_score)
        overlap_features = dy.inputVector(overlap_features)
        # Pass each query term representation through the MLP
        term_scores = []
        for hist in q_d_hists:
            q_d_hist = dy.reshape(dy.inputVector(hist), (1, len(hist)))
            hidd_out = dy.rectify(q_d_hist * self.W_1 + self.b_1)
            for i in range(0, self.mlp_layers):
                hidd_out = dy.rectify(hidd_out * self.W_n[i] + self.b_n[i])
            term_scores.append(hidd_out * self.W_last + self.b_last)

        # Term Gating
        gating_weights = idf_vec * self.w_g
        
        bm25_feature = bm25_score * self.W_bm25 + self.b_bm25 
        drop_out =  dy.scalarInput(1)
        drop_num = (np.random.rand(1) < p)/p #p= probability of keeping a unit active
        drop_out.set(drop_num)
        
        bm25_feature *= drop_out
        drmm_score = dy.transpose(dy.concatenate(term_scores)) * dy.reshape(gating_weights, (len(q_idf), 1)) #basic MLPs output
        doc_score = dy.transpose(dy.concatenate([drmm_score, overlap_features])) * self.W_scores + self.b_scores #extra features layer
        
        
        return doc_score