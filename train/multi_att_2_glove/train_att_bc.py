import matplotlib
matplotlib.use('Agg')
import os
import sys
import numpy as np
import json
import matplotlib.pyplot as plt

import caffe
from caffe import layers as L
from caffe import params as P

from vqa_data_provider_layer import VQADataProvider
from visualize_tools import exec_validation, drawgraph
import config


def qlstm(mode, batchsize, T, question_vocab_size):
    n = caffe.NetSpec()
    mode_str = json.dumps({'mode':mode, 'batchsize':batchsize})
    n.data, n.cont, n.img_feature, n.label, n.glove = L.Python(\
        module='vqa_data_provider_layer', layer='VQADataProviderLayer', param_str=mode_str, ntop=5 )

    n.embed_ba = L.Embed(n.data, input_dim=question_vocab_size, num_output=300, \
        weight_filler=dict(type='uniform',min=-0.08,max=0.08))
    n.embed = L.TanH(n.embed_ba) 
    concat_word_embed = [n.embed, n.glove]
    n.concat_embed = L.Concat(*concat_word_embed, concat_param={'axis': 2}) # T x N x 600

    # LSTM1
    n.lstm1 = L.LSTM(\
                   n.concat_embed, n.cont,\
                   recurrent_param=dict(\
                       num_output=1024,\
                       weight_filler=dict(type='uniform',min=-0.08,max=0.08),\
                       bias_filler=dict(type='constant',value=0)))
    tops1 = L.Slice(n.lstm1, ntop=T, slice_param={'axis':0})
    for i in xrange(T-1):
        n.__setattr__('slice_first'+str(i), tops1[int(i)])
        n.__setattr__('silence_data_first'+str(i), L.Silence(tops1[int(i)],ntop=0))
    n.lstm1_out = tops1[T-1]
    n.lstm1_reshaped = L.Reshape(n.lstm1_out,\
                          reshape_param=dict(\
                              shape=dict(dim=[-1,1024])))
    n.lstm1_reshaped_droped = L.Dropout(n.lstm1_reshaped,dropout_param={'dropout_ratio':0.3})
    n.lstm1_droped = L.Dropout(n.lstm1,dropout_param={'dropout_ratio':0.3})
    # LSTM2
    n.lstm2 = L.LSTM(\
                   n.lstm1_droped, n.cont,\
                   recurrent_param=dict(\
                       num_output=1024,\
                       weight_filler=dict(type='uniform',min=-0.08,max=0.08),\
                       bias_filler=dict(type='constant',value=0)))
    tops2 = L.Slice(n.lstm2, ntop=T, slice_param={'axis':0})
    for i in xrange(T-1):
        n.__setattr__('slice_second'+str(i), tops2[int(i)])
        n.__setattr__('silence_data_second'+str(i), L.Silence(tops2[int(i)],ntop=0))
    n.lstm2_out = tops2[T-1]
    n.lstm2_reshaped = L.Reshape(n.lstm2_out,\
                          reshape_param=dict(\
                              shape=dict(dim=[-1,1024])))
    n.lstm2_reshaped_droped = L.Dropout(n.lstm2_reshaped,dropout_param={'dropout_ratio':0.3})
    concat_botom = [n.lstm1_reshaped_droped, n.lstm2_reshaped_droped]
    n.lstm_12 = L.Concat(*concat_botom)

    n.q_emb_tanh_droped_resh = L.Reshape(n.lstm_12,reshape_param=dict(shape=dict(dim=[-1,2048,1,1])))
    n.q_emb_tanh_droped_resh_tiled_1 = L.Tile(n.q_emb_tanh_droped_resh, axis=2, tiles=14)
    n.q_emb_tanh_droped_resh_tiled = L.Tile(n.q_emb_tanh_droped_resh_tiled_1, axis=3, tiles=14)
    n.i_emb_tanh_droped_resh = L.Reshape(n.img_feature,reshape_param=dict(shape=dict(dim=[-1,2048,14,14])))
    n.blcf = L.CompactBilinear(n.q_emb_tanh_droped_resh_tiled, n.i_emb_tanh_droped_resh, compact_bilinear_param=dict(num_output=16000,sum_pool=False))
    n.blcf_sign_sqrt = L.SignedSqrt(n.blcf)
    n.blcf_sign_sqrt_l2 = L.L2Normalize(n.blcf_sign_sqrt)
    n.blcf_droped = L.Dropout(n.blcf_sign_sqrt_l2,dropout_param={'dropout_ratio':0.1})

    # multi-channel attention
    n.att_conv1 = L.Convolution(n.blcf_droped, kernel_size=1, stride=1, num_output=512, pad=0, weight_filler=dict(type='xavier'))
    n.att_conv1_relu = L.ReLU(n.att_conv1)
    n.att_conv2 = L.Convolution(n.att_conv1_relu, kernel_size=1, stride=1, num_output=2, pad=0, weight_filler=dict(type='xavier'))
    n.att_reshaped = L.Reshape(n.att_conv2,reshape_param=dict(shape=dict(dim=[-1,2,14*14])))
    n.att_softmax = L.Softmax(n.att_reshaped, axis=2)
    n.att = L.Reshape(n.att_softmax,reshape_param=dict(shape=dict(dim=[-1,2,14,14])))
    att_maps = L.Slice(n.att, ntop=2, slice_param={'axis':1})
    n.att_map0 = att_maps[0]
    n.att_map1 = att_maps[1]
    dummy = L.DummyData(shape=dict(dim=[batchsize, 1]), data_filler=dict(type='constant', value=1), ntop=1)
    n.att_feature0  = L.SoftAttention(n.i_emb_tanh_droped_resh, n.att_map0, dummy)
    n.att_feature1  = L.SoftAttention(n.i_emb_tanh_droped_resh, n.att_map1, dummy)
    n.att_feature0_resh = L.Reshape(n.att_feature0, reshape_param=dict(shape=dict(dim=[-1,2048])))
    n.att_feature1_resh = L.Reshape(n.att_feature1, reshape_param=dict(shape=dict(dim=[-1,2048])))
    n.att_feature = L.Concat(n.att_feature0_resh, n.att_feature1_resh)

    # merge attention and lstm with compact bilinear pooling
    n.att_feature_resh = L.Reshape(n.att_feature, reshape_param=dict(shape=dict(dim=[-1,4096,1,1])))
    n.lstm_12_resh = L.Reshape(n.lstm_12, reshape_param=dict(shape=dict(dim=[-1,2048,1,1])))
    n.bc_att_lstm = L.CompactBilinear(n.att_feature_resh, n.lstm_12_resh, 
                                      compact_bilinear_param=dict(num_output=16000,sum_pool=False))
    n.bc_sign_sqrt = L.SignedSqrt(n.bc_att_lstm)
    n.bc_sign_sqrt_l2 = L.L2Normalize(n.bc_sign_sqrt)

    n.bc_dropped = L.Dropout(n.bc_sign_sqrt_l2, dropout_param={'dropout_ratio':0.1})
    n.bc_dropped_resh = L.Reshape(n.bc_dropped, reshape_param=dict(shape=dict(dim=[-1, 16000])))

    n.prediction = L.InnerProduct(n.bc_dropped_resh, num_output=3000, weight_filler=dict(type='xavier'))
    n.loss = L.SoftmaxWithLoss(n.prediction, n.label)
    return n.to_proto()

def make_answer_vocab(adic, vocab_size):
    """
    Returns a dictionary that maps words to indices.
    """
    adict = {'':0}
    nadict = {'':1000000}
    vid = 1
    for qid in adic.keys():
        answer_obj = adic[qid]
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
    nalist = []
    for k,v in sorted(nadict.items(), key=lambda x:x[1]):
        nalist.append((k,v))

    # remove words that appear less than once 
    n_del_ans = 0
    n_valid_ans = 0
    adict_nid = {}
    for i, w in enumerate(nalist[:-vocab_size]):
        del adict[w[0]]
        n_del_ans += w[1]
    for i, w in enumerate(nalist[-vocab_size:]):
        n_valid_ans += w[1]
        adict_nid[w[0]] = i
    
    return adict_nid

def make_question_vocab(qdic):
    """
    Returns a dictionary that maps words to indices.
    """
    vdict = {'':0}
    vid = 1
    for qid in qdic.keys():
        # sequence to list
        q_str = qdic[qid]['qstr']
        q_list = VQADataProvider.seq_to_list(q_str)

        # create dict
        for w in q_list:
            if not vdict.has_key(w):
                vdict[w] = vid
                vid +=1

    return vdict

def make_vocab_files():
    """
    Produce the question and answer vocabulary files.
    """
    print 'making question vocab...', config.QUESTION_VOCAB_SPACE
    qdic, _ = VQADataProvider.load_data(config.QUESTION_VOCAB_SPACE)
    question_vocab = make_question_vocab(qdic)
    print 'making answer vocab...', config.ANSWER_VOCAB_SPACE
    _, adic = VQADataProvider.load_data(config.ANSWER_VOCAB_SPACE)
    answer_vocab = make_answer_vocab(adic, config.NUM_OUTPUT_UNITS)
    return question_vocab, answer_vocab

def main():
    if not os.path.exists('./result'):
        os.makedirs('./result')

    question_vocab, answer_vocab = {}, {}
    if os.path.exists('./result/vdict.json') and os.path.exists('./result/adict.json'):
        print 'restoring vocab'
        with open('./result/vdict.json','r') as f:
            question_vocab = json.load(f)
        with open('./result/adict.json','r') as f:
            answer_vocab = json.load(f)
    else:
        question_vocab, answer_vocab = make_vocab_files()
        with open('./result/vdict.json','w') as f:
            json.dump(question_vocab, f)
        with open('./result/adict.json','w') as f:
            json.dump(answer_vocab, f)

    print 'question vocab size:', len(question_vocab)
    print 'answer vocab size:', len(answer_vocab)

    with open('./result/proto_train.prototxt', 'w') as f:
        f.write(str(qlstm(config.TRAIN_DATA_SPLITS, config.BATCH_SIZE, \
            config.MAX_WORDS_IN_QUESTION, len(question_vocab))))
    
    with open('./result/proto_test.prototxt', 'w') as f:
        f.write(str(qlstm('val', config.VAL_BATCH_SIZE, \
            config.MAX_WORDS_IN_QUESTION, len(question_vocab))))

    caffe.set_device(config.GPU_ID)
    caffe.set_mode_gpu()
    solver = caffe.get_solver('./qlstm_solver.prototxt')

    train_loss = np.zeros(config.MAX_ITERATIONS)
    results = []

    for it in range(config.MAX_ITERATIONS):
        solver.step(1)
    
        # store the train loss
        train_loss[it] = solver.net.blobs['loss'].data
   
        if it % config.PRINT_INTERVAL == 0:
            print 'Iteration:', it
            c_mean_loss = train_loss[it-config.PRINT_INTERVAL:it].mean()
            print 'Train loss:', c_mean_loss
        if it != 0 and it % config.VALIDATE_INTERVAL == 0:
            solver.test_nets[0].save('./result/tmp.caffemodel')
            print 'Validating...'
            test_loss, acc_overall, acc_per_ques, acc_per_ans = exec_validation(config.GPU_ID, 'val', it=it)
            print 'Test loss:', test_loss
            print 'Accuracy:', acc_overall
            results.append([it, c_mean_loss, test_loss, acc_overall, acc_per_ques, acc_per_ans])
            best_result_idx = np.array([x[3] for x in results]).argmax()
            print 'Best accuracy of', results[best_result_idx][3], 'was at iteration', results[best_result_idx][0]
            drawgraph(results)

if __name__ == '__main__':
    main()
