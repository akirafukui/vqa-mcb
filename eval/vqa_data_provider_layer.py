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
SPATIAL_COORD = None
GLOVE = None

class LoadVQADataProvider:

    def __init__(self, ques_file_path, img_file_pre, vdict_path, adict_path, \
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

        # spatial coordinates
        normalized_coords = np.linspace(0, 2, num=14, endpoint=True, dtype=np.float32) / 200
        self.x_coords = np.tile(normalized_coords, (14, 1)).reshape(1, 14, 14)
        normalized_coords = normalized_coords.reshape((14, 1))
        self.y_coords = np.tile(normalized_coords, (1, 14)).reshape(1, 14, 14)
        self.coords = np.concatenate([self.x_coords, self.y_coords])

        self.quesFile = ques_file_path
        self.img_file_pre = img_file_pre
        # load ques file
        with open(self.quesFile,'r') as f:
            print 'reading : ', self.quesFile
            qdata = json.load(f)
            qdic = {}
            for q in qdata['questions']:
                qdic[q['question_id']] = { 'qstr':q['question'], 'iid':q['image_id']}
            self.qdic = qdic
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
 
    def create_answer_vocabulary_dict(self, genome=False):
        n_ans_vocabulary=self.n_ans_vocabulary
        qid_list = self.getQuesIds()
        adict = {'':0}
        nadict = {'':1000000}
        vid = 1
        for qid in qid_list:
            if genome and qid[0] == 'g':
                continue
            answer_obj = self.getAnsObj(qid)
            answer_list = [ans['answer'] for ans in answer_obj]
            
            for q_ans in answer_list:
                # create dict
                if adict.has_key(q_ans):
                    nadict[q_ans] += 1
                else:
                    nadict[q_ans] = 1
                    adict[q_ans] = vid
                    vid +=1

        # debug
        klist = []
        for k,v in sorted(nadict.items()):
            klist.append((k,v))
        nalist = []
        for k,v in sorted(nadict.items(), key=lambda x:x[1]):
            nalist.append((k,v))
        alist = []
        for k,v in sorted(adict.items(), key=lambda x:x[1]):
            alist.append((k,v))

        # remove words that appear less than once 
        n_del_ans = 0
        n_valid_ans = 0
        adict_nid = {}
        for i, w in enumerate(nalist[:-n_ans_vocabulary]):
            del adict[w[0]]
            n_del_ans += w[1]
        for i, w in enumerate(nalist[-n_ans_vocabulary:]):
            n_valid_ans += w[1]
            adict_nid[w[0]] = i
        print 'Valid answers are : ', n_valid_ans
        print 'Invalid answers are : ', n_del_ans

        return n_ans_vocabulary, adict_nid


    def create_vocabulary_dict(self):
        #qid_list = self.vqa.getQuesIds()
        qid_list = self.getQuesIds()
        vdict = {'':0}
        ndict = {'':0}
        vid = 1
        for qid in qid_list:
            # sequence to list
            q_str = self.getQuesStr(qid)
            q_list = self.seq_to_list(q_str)

            # create dict
            for w in q_list:
                if vdict.has_key(w):
                    ndict[w] += 1
                else:
                    ndict[w] = 1
                    vdict[w] = vid
                    vid +=1

        # debug
        klist = []
        for k,v in sorted(ndict.items()):
            klist.append((k,v))
        nlist = []
        for k,v in sorted(ndict.items(), key=lambda x:x[1]):
            nlist.append((k,v))
        vlist = []
        for k,v in sorted(vdict.items(), key=lambda x:x[1]):
            vlist.append((k,v))

        n_vocabulary = len(vlist)

        #from IPython import embed; embed(); sys.exit()
        return n_vocabulary, vdict

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
 
    def create_batch(self,qid_list):

        qvec = (np.zeros(self.batchsize*self.max_length)).reshape(self.batchsize,self.max_length)
        cvec = (np.zeros(self.batchsize*self.max_length)).reshape(self.batchsize,self.max_length)
        ivec = (np.zeros(self.batchsize*reduce(mul, self.data_shape))).reshape(self.batchsize,*self.data_shape)
        avec = (np.zeros(self.batchsize)).reshape(self.batchsize)
        glove_matrix = np.zeros(self.batchsize * self.max_length * GLOVE_EMBEDDING_SIZE).reshape(\
            self.batchsize, self.max_length, GLOVE_EMBEDDING_SIZE)

        for i,qid in enumerate(qid_list):

            # load raw question information
            q_str = self.getQuesStr(qid)
            q_ans = self.getAnsObj(qid)
            q_iid = self.getImgId(qid)

            # convert question to vec
            q_list = self.seq_to_list(q_str)
            t_qvec, t_cvec, t_glove_matrix = self.qlist_to_vec(self.max_length, q_list)

            # convert answer to vec
            try:
                if type(qid) == int:
                    t_ivec = np.load(self.img_file_pre + str(q_iid).zfill(12) + '.jpg.npz')['x']
                    t_ivec = ( t_ivec / np.sqrt((t_ivec**2).sum()) )
                elif qid[0] == 't':
                    t_ivec = np.load(self.img_file_pre_t + str(q_iid).zfill(12) + '.jpg.npz')['x']
                    t_ivec = ( t_ivec / np.sqrt((t_ivec**2).sum()) )
                elif qid[0] =='v':
                    t_ivec = np.load(self.img_file_pre_v + str(q_iid).zfill(12) + '.jpg.npz')['x']
                    t_ivec = ( t_ivec / np.sqrt((t_ivec**2).sum()) )
                elif qid[0] == 'g':
                    t_ivec = np.load(self.img_file_pre_g + str(q_iid) + '.jpg.npz')['x']
                    t_ivec = ( t_ivec / np.sqrt((t_ivec**2).sum()) )
                else:
                    raise Exception('Error occured here')
                    t_ivec = np.load(self.img_file_pre + str(q_iid).zfill(12) + '.jpg.npz')['x']
                    t_ivec = ( t_ivec / np.sqrt((t_ivec**2).sum()) )
                if SPATIAL_COORD:
                    t_ivec = np.concatenate([t_ivec, self.coords.copy()])
            except:
                t_ivec = 0.
                print 'data not found for qid : ', q_iid,  self.mode
             
            # convert answer to vec
            if self.mode == 'val' or self.mode == 'test-dev' or self.mode == 'test':
                q_ans_str = self.extract_answer(q_ans)
            else:
                q_ans_str = self.extract_answer_prob(q_ans)
            t_avec = self.answer_to_vec(q_ans_str)

            qvec[i,...] = t_qvec
            cvec[i,...] = t_cvec
            ivec[i,...] = t_ivec
            avec[i,...] = t_avec
            glove_matrix[i,...] = t_glove_matrix

        return qvec, cvec, ivec, avec, glove_matrix

 
    def get_batch_vec(self):
        if self.batch_len is None:
            #qid_list = self.vqa.getQuesIds()
            self.n_skipped = 0
            qid_list = self.getQuesIds()
            # random.shuffle(qid_list)
            self.qid_list = qid_list
            self.batch_len = len(qid_list)
            self.batch_index = 0
            self.epoch_counter = 0

        def has_at_least_one_valid_answer(t_qid):
            #answer_obj = self.vqa.qa[t_qid]['answers']
            answer_obj = self.getAnsObj(t_qid)
            answer_list = [ans['answer'] for ans in answer_obj]
            for ans in answer_list:
                if self.adict.has_key(ans):
                    return True

        counter = 0
        t_qid_list = []
        t_iid_list = []
        while counter < self.batchsize:
            # get qid
            t_qid = self.qid_list[self.batch_index]
            # get answer
            #t_ans = self.extract_answer(self.vqa.qa[t_qid]['answers'])
            # get image id
            #t_ann = self.vqa.loadQA([t_qid])[0]
            #t_iid = t_ann['image_id']
            t_iid = self.getImgId(t_qid)
            if self.mode == 'val' or self.mode == 'test-dev' or self.mode == 'test':
                t_qid_list.append(t_qid)
                t_iid_list.append(t_iid)
                counter += 1
            elif has_at_least_one_valid_answer(t_qid):
                t_qid_list.append(t_qid)
                t_iid_list.append(t_iid)
                counter += 1
            else:
                self.n_skipped += 1 

            if self.batch_index < self.batch_len-1:
                self.batch_index += 1
            else:
                self.epoch_counter += 1
                #qid_list = self.vqa.getQuesIds()
                qid_list = self.getQuesIds()
                # random.shuffle(qid_list)
                self.qid_list = qid_list
                self.batch_index = 0
                print("%d questions were skipped in a single epoch" % self.n_skipped)
                self.n_skipped = 0

        t_batch = self.create_batch(t_qid_list)
        return t_batch + (t_qid_list, t_iid_list, self.epoch_counter)


class VQADataProviderLayer(caffe.Layer):
    """
    Provide input data for VQA.
    """

    def setup(self, bottom, top):
        self.batchsize = json.loads(self.param_str)['batchsize']
        names = ['data','cont','feature','label']
        if GLOVE:
            names.append('glove')
        self.top_names = names
        top[0].reshape(15,self.batchsize)
        top[1].reshape(15,self.batchsize)
        top[2].reshape(self.batchsize, *CURRENT_DATA_SHAPE)
        top[3].reshape(self.batchsize)
        if GLOVE:
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

