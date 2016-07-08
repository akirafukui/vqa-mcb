import caffe
import numpy as np

import random
import os
import sys
import re
import json
import spacy
from operator import mul

GLOVE_EMBEDDING_SIZE = 300

CURRENT_DATA_SHAPE = None

class LoadVQADataProvider:

    def __init__(self, vdict_path, adict_path, \
        batchsize=128, max_length=15, n_ans_vocabulary=1000, mode='train', data_shape=(2048)):

        self.batchsize = batchsize
        self.d_vocabulary = None
        self.batch_index = None
        self.batch_len = None
        self.rev_adict = None
        self.max_length = max_length
        self.n_ans_vocabulary = n_ans_vocabulary
        self.mode = mode
        self.data_shape = data_shape

        assert self.mode == 'test'

        # load vocabulary
        with open(vdict_path,'r') as f:
            vdict = json.load(f)
        with open(adict_path,'r') as f:
            adict = json.load(f)
        self.n_vocabulary, self.vdict = len(vdict), vdict
        self.n_ans_vocabulary, self.adict = len(adict), adict

        self.nlp = spacy.load('en', vectors='en_glove_cc_300_1m_vectors')
        self.glove_dict = {} # word -> glove vector

    def getQuesIds(self):
        return self.qdic.keys()

    def getImgId(self,qid):
        return self.qdic[qid]['iid']

    def getQuesStr(self,qid):
        return self.qdic[qid]['qstr']

    def getAnsObj(self,qid):
        if self.mode == 'test-dev' or self.mode == 'test':
            return -1
        return self.adic[qid]

    def seq_to_list(self, s):
        t_str = s.lower()
        for i in [r'\?',r'\!',r'\'',r'\"',r'\$',r'\:',r'\@',r'\(',r'\)',r'\,',r'\.',r'\;']:
            t_str = re.sub( i, '', t_str)
        for i in [r'\-',r'\/']:
            t_str = re.sub( i, ' ', t_str)
        q_list = re.sub(r'\?','',t_str.lower()).split(' ')
        q_list = filter(lambda x: len(x) > 0, q_list)
        return q_list

    def extract_answer(self,answer_obj):
        """ Return the most popular answer in string."""
        if self.mode == 'test-dev' or self.mode == 'test':
            return -1
        answer_list = [ answer_obj[i]['answer'] for i in xrange(10)]
        dic = {}
        for ans in answer_list:
            if dic.has_key(ans):
                dic[ans] +=1
            else:
                dic[ans] = 1
        max_key = max((v,k) for (k,v) in dic.items())[1]
        return max_key

    def extract_answer_prob(self,answer_obj):
        """ Return the most popular answer in string."""
        if self.mode == 'test-dev' or self.mode == 'test':
            return -1

        answer_list = [ ans['answer'] for ans in answer_obj]
        prob_answer_list = []
        for ans in answer_list:
            if self.adict.has_key(ans):
                prob_answer_list.append(ans)

        if len(prob_answer_list) == 0:
            if self.mode == 'val' or self.mode == 'test-dev' or self.mode == 'test':
                return 'hoge'
            else:
                raise Exception("This should not happen.")
        else:
            return random.choice(prob_answer_list)
 
    def qlist_to_vec(self, max_length, q_list):
        """
        Converts a list of words into a format suitable for the embedding layer.

        Arguments:
        max_length -- the maximum length of a question sequence
        q_list -- a list of words which are the tokens in the question

        Returns:
        qvec -- A max_length length vector containing one-hot indices for each word
        cvec -- A max_length length sequence continuation indicator vector
        glove_matrix -- A max_length x GLOVE_EMBEDDING_SIZE matrix containing the glove embedding for
            each word
        """
        qvec = np.zeros(max_length)
        cvec = np.zeros(max_length)
        glove_matrix = np.zeros(max_length * GLOVE_EMBEDDING_SIZE).reshape(max_length, GLOVE_EMBEDDING_SIZE)
        for i in xrange(max_length):
            if i < max_length - len(q_list):
                cvec[i] = 0
            else:
                w = q_list[i-(max_length-len(q_list))]
                if w not in self.glove_dict:
                    self.glove_dict[w] = self.nlp(u'%s' % w).vector
                glove_matrix[i] = self.glove_dict[w]
                # is the word in the vocabulary?
                if self.vdict.has_key(w) is False:
                    w = ''
                qvec[i] = self.vdict[w]
                cvec[i] = 0 if i == max_length - len(q_list) else 1

        return qvec, cvec, glove_matrix
 
    def answer_to_vec(self, ans_str):
        """ Return answer id if the answer is included in vocaburary otherwise '' """
        if self.mode =='test-dev' or self.mode == 'test':
            return -1

        if self.adict.has_key(ans_str):
            ans = self.adict[ans_str]
        else:
            ans = self.adict['']
        return ans
 
    def vec_to_answer(self, ans_symbol):
        """ Return answer id if the answer is included in vocaburary otherwise '' """
        if self.rev_adict is None:
            rev_adict = {}
            for k,v in self.adict.items():
                rev_adict[v] = k
            self.rev_adict = rev_adict

        return self.rev_adict[ans_symbol]
 
    def create_batch(self, question):

        qvec = (np.zeros(self.batchsize*self.max_length)).reshape(self.batchsize,self.max_length)
        cvec = (np.zeros(self.batchsize*self.max_length)).reshape(self.batchsize,self.max_length)
        avec = (np.zeros(self.batchsize)).reshape(self.batchsize)
        glove_matrix = np.zeros(self.batchsize * self.max_length * GLOVE_EMBEDDING_SIZE).reshape(\
            self.batchsize, self.max_length, GLOVE_EMBEDDING_SIZE)

        q_list = self.seq_to_list(question)
        t_qvec, t_cvec, t_glove_matrix = self.qlist_to_vec(self.max_length, q_list)

        i = 0
        qvec[i,...] = t_qvec
        cvec[i,...] = t_cvec
        glove_matrix[i,...] = t_glove_matrix

        return qvec, cvec, avec, glove_matrix

 


class VQADataProviderLayer(caffe.Layer):
    """
    Provide input data for VQA.
    """

    def setup(self, bottom, top):
        self.batchsize = 1
        names = ['data','cont','feature','label', 'glove']
        self.top_names = names
        top[0].reshape(15,self.batchsize)
        top[1].reshape(15,self.batchsize)
        top[2].reshape(self.batchsize, *CURRENT_DATA_SHAPE)
        top[3].reshape(self.batchsize)
        top[4].reshape(15,self.batchsize,GLOVE_EMBEDDING_SIZE)

        self.mode = json.loads(self.param_str)['mode']
        if self.mode == 'val' or self.mode == 'test-dev' or self.mode == 'test':
            pass
        else:
            raise NotImplementedError

    def reshape(self, bottom, top):
        pass

    def forward(self, bottom, top):
        if self.mode == 'val' or self.mode == 'test-dev' or self.mode == 'test':
            pass
        else:
            raise NotImplementedError

    def backward(self, top, propagate_down, bottom):
        pass

