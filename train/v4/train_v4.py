import matplotlib
matplotlib.use('Agg')
import os
import numpy as np
import json
import caffe
from caffe import layers as L
from caffe import params as P
from vqa_data_provider_layer import VQADataProvider
from visualize_tools import drawgraph, exec_validation
import config

def qlstm(mode, batchsize, max_words_in_question, question_vocab_size):
    n = caffe.NetSpec()
    mode_str = json.dumps({'mode':mode, 'batchsize':batchsize})
    n.data, n.cont, n.img_feature, n.label = L.Python( \
        module='vqa_data_provider_layer', layer='VQADataProviderLayer', \
        param_str=mode_str, ntop=4 )

    n.embed_ba = L.Embed(n.data, input_dim=question_vocab_size, num_output=300, \
        weight_filler=dict(type='uniform',min=-0.08,max=0.08))
    n.embed = L.TanH(n.embed_ba) 

    # LSTM1
    n.lstm1 = L.LSTM(\
                   n.embed, n.cont,\
                   recurrent_param=dict(\
                       num_output=1024,\
                       weight_filler=dict(type='uniform',min=-0.08,max=0.08),\
                       bias_filler=dict(type='constant',value=0)))
    tops1 = L.Slice(n.lstm1, ntop=max_words_in_question, slice_param={'axis':0})
    for i in xrange(max_words_in_question-1):
        n.__setattr__('slice_first'+str(i), tops1[int(i)])
        n.__setattr__('silence_data_first'+str(i), L.Silence(tops1[int(i)],ntop=0))
    n.lstm1_out = tops1[max_words_in_question-1]
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
    tops2 = L.Slice(n.lstm2, ntop=max_words_in_question, slice_param={'axis':0})
    for i in xrange(max_words_in_question-1):
        n.__setattr__('slice_second'+str(i), tops2[int(i)])
        n.__setattr__('silence_data_second'+str(i), L.Silence(tops2[int(i)],ntop=0))
    n.lstm2_out = tops2[max_words_in_question-1]
    n.lstm2_reshaped = L.Reshape(n.lstm2_out,\
                          reshape_param=dict(\
                              shape=dict(dim=[-1,1024])))
    n.lstm2_reshaped_droped = L.Dropout(n.lstm2_reshaped,dropout_param={'dropout_ratio':0.3})
    concat_botom = [n.lstm1_reshaped_droped, n.lstm2_reshaped_droped]
    n.lstm_12 = L.Concat(*concat_botom)

    n.q_emb_tanh_droped_resh = L.Reshape(n.lstm_12, \
        reshape_param=dict(shape=dict(dim=[-1,2048,1,1])))
    n.i_emb_tanh_droped_resh = L.Reshape(n.img_feature, \
        reshape_param=dict(shape=dict(dim=[-1,2048,1,1])))
    n.blcf = L.CompactBilinear(n.q_emb_tanh_droped_resh, n.i_emb_tanh_droped_resh, \
        compact_bilinear_param=dict(num_output=16000,sum_pool=False))
    n.blcf_sign_sqrt = L.SignedSqrt(n.blcf)
    n.blcf_sign_sqrt_l2 = L.L2Normalize(n.blcf_sign_sqrt)

    n.blcf_droped = L.Dropout(n.blcf_sign_sqrt_l2,dropout_param={'dropout_ratio':0.1})
    n.blcf_droped_resh = L.Reshape(n.blcf_droped,reshape_param=dict(shape=dict(dim=[-1,16000])))

    n.prediction = L.InnerProduct(n.blcf_droped_resh, num_output=config.NUM_OUTPUT_UNITS, \
        weight_filler=dict(type='xavier'))
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
        f.write(str(qlstm('val', config.BATCH_SIZE, \
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
            if it % config.VALIDATE_INTERVAL == 0:
                print 'Validating...'
                solver.test_nets[0].save('./result/tmp.caffemodel')
                test_loss, acc_overall, acc_per_ques, acc_per_ans = exec_validation(config.GPU_ID, 'val', it=it)
                print 'Test loss:', test_loss
                print 'Accuracy:', acc_overall
                results.append([it, c_mean_loss, test_loss, acc_overall, acc_per_ques, acc_per_ans])
                best_result_idx = np.array([x[3] for x in results]).argmax()
                print 'Best accuracy of', results[best_result_idx][3], 'was at iteration', results[best_result_idx][0]
            drawgraph(results)

if __name__ == '__main__':
    main()
