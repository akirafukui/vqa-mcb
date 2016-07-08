import numpy as np
import matplotlib.pyplot as plt
import os
import sys
import json
import re
import shutil
from PIL import Image
from PIL import ImageFont, ImageDraw

import caffe
from caffe import layers as L
from caffe import params as P

from vqa_data_provider_layer import VQADataProvider
from vqa_data_provider_layer import VQADataProviderLayer

import config
sys.path.append(config.VQA_TOOLS_PATH)
sys.path.append(config.VQA_EVAL_TOOLS_PATH)

from vqaTools.vqa import VQA
from vqaEvaluation.vqaEval import VQAEval

def visualize_failures(stat_list,mode):

    def save_qtype(qtype_list, save_filename, mode):

        if mode == 'val':
            savepath = os.path.join('./eval', save_filename)
            # TODO
            img_pre = '/home/dhpseth/vqa/02_tools/VQA/Images/val2014'
        elif mode == 'test-dev':
            savepath = os.path.join('./test-dev', save_filename)
            # TODO
            img_pre = '/home/dhpseth/vqa/02_tools/VQA/Images/test2015'
        elif mode == 'test':
            savepath = os.path.join('./test', save_filename)
            # TODO
            img_pre = '/home/dhpseth/vqa/02_tools/VQA/Images/test2015'
        else:
            raise Exception('Unsupported mode')
        if os.path.exists(savepath): shutil.rmtree(savepath)
        if not os.path.exists(savepath): os.makedirs(savepath)

        for qt in qtype_list:
            count = 0
            for t_question in stat_list:
                #print count, t_question
                if count < 40/len(qtype_list):
                    t_question_list = t_question['q_list']
                    saveflag = False
                    #print 'debug****************************'
                    #print qt
                    #print t_question_list
                    #print t_question_list[0] == qt[0]
                    #print t_question_list[1] == qt[1]
                    if t_question_list[0] == qt[0] and t_question_list[1] == qt[1]:
                        saveflag = True
                    else:
                        saveflag = False
                               
                    if saveflag == True:
                        t_iid = t_question['iid']
                        if mode == 'val':
                            t_img = Image.open(os.path.join(img_pre, \
                                'COCO_val2014_' + str(t_iid).zfill(12) + '.jpg'))
                        elif mode == 'test-dev' or 'test':
                            t_img = Image.open(os.path.join(img_pre, \
                                'COCO_test2015_' + str(t_iid).zfill(12) + '.jpg'))

                        # for caption
                        #print t_iid
                        #annIds = caps.getAnnIds(t_iid)
                        #anns = caps.loadAnns(annIds)
                        #cap_list = [ann['caption'] for ann in anns]
                        ans_list = t_question['ans_list']
                        draw = ImageDraw.Draw(t_img)
                        for i in range(len(ans_list)):
                            try:
                                draw.text((10,10*i), str(ans_list[i]))
                            except:
                                pass

                        ans = t_question['answer']
                        pred = t_question['pred']
                        if ans == -1:
                            pre = ''
                        elif ans == pred:
                            pre = 'correct  '
                        else:
                            pre = 'failure  '
                        #print ' aaa ', ans, pred
                        ans = re.sub( '/', ' ', str(ans))
                        pred = re.sub( '/', ' ', str(pred))
                        img_title = pre + str(' '.join(t_question_list)) + '.  a_' + \
                            str(ans) + ' p_' + str(pred) + '.png'
                        count += 1
                        print os.path.join(savepath,img_title)
                        t_img.save(os.path.join(savepath,img_title))

    print 'saving whatis'
    qt_color_list = [['what','color']]
    save_qtype(qt_color_list, 'colors', mode)

    print 'saving whatis'
    qt_whatis_list = [['what','is'],['what','kind'],['what','are']]
    save_qtype(qt_whatis_list, 'whatis', mode)

    print 'saving is'
    qt_is_list = [['is','the'], ['is','this'],['is','there']]
    save_qtype(qt_is_list, 'is', mode)

    print 'saving how many'
    qt_howmany_list =[['how','many']]
    save_qtype(qt_howmany_list, 'howmany', mode)

def exec_validation(device_id, mode, it='', visualize=False):

    caffe.set_device(device_id)
    caffe.set_mode_gpu()
    net = caffe.Net('./result/proto_test.prototxt',\
              './result/tmp.caffemodel',\
              caffe.TEST)

    dp = VQADataProvider(mode=mode,batchsize=config.VAL_BATCH_SIZE)
    total_questions = len(dp.getQuesIds())
    epoch = 0

    pred_list = []
    testloss_list = []
    stat_list = []

    while epoch == 0:
        t_word, t_cont, t_img_feature, t_answer, t_glove_matrix, t_qid_list, t_iid_list, epoch = dp.get_batch_vec()
        net.blobs['data'].data[...] = np.transpose(t_word,(1,0))
        net.blobs['cont'].data[...] = np.transpose(t_cont,(1,0))
        net.blobs['img_feature'].data[...] = t_img_feature
        net.blobs['label'].data[...] = t_answer
        net.blobs['glove'].data[...] = np.transpose(t_glove_matrix, (1,0,2))
        net.forward()
        t_pred_list = net.blobs['prediction'].data.argmax(axis=1)
        t_pred_str = [dp.vec_to_answer(pred_symbol) for pred_symbol in t_pred_list]
        testloss_list.append(net.blobs['loss'].data)
        for qid, iid, ans, pred in zip(t_qid_list, t_iid_list, t_answer.tolist(), t_pred_str):
            pred_list.append({u'answer':pred, u'question_id': int(dp.getStrippedQuesId(qid))})
            if visualize:
                q_list = dp.seq_to_list(dp.getQuesStr(qid))
                if mode == 'test-dev' or 'test':
                    ans_str = ''
                    ans_list = ['']*10
                else:
                    ans_str = dp.vec_to_answer(ans)
                    ans_list = [ dp.getAnsObj(qid)[i]['answer'] for i in xrange(10)]
                stat_list.append({\
                                    'qid'   : qid,
                                    'q_list' : q_list,
                                    'iid'   : iid,
                                    'answer': ans_str,
                                    'ans_list': ans_list,
                                    'pred'  : pred })
        percent = 100 * float(len(pred_list)) / total_questions
        sys.stdout.write('\r' + ('%.2f' % percent) + '%')
        sys.stdout.flush()



    mean_testloss = np.array(testloss_list).mean()

    if mode == 'val':
        valFile = './result/val2015_resfile'
        with open(valFile, 'w') as f:
            json.dump(pred_list, f)
        if visualize:
            visualize_failures(stat_list,mode)
        annFile = config.DATA_PATHS['val']['ans_file']
        quesFile = config.DATA_PATHS['val']['ques_file']
        vqa = VQA(annFile, quesFile)
        vqaRes = vqa.loadRes(valFile, quesFile)
        vqaEval = VQAEval(vqa, vqaRes, n=2)
        vqaEval.evaluate()
        acc_overall = vqaEval.accuracy['overall']
        acc_perQuestionType = vqaEval.accuracy['perQuestionType']
        acc_perAnswerType = vqaEval.accuracy['perAnswerType']
        return mean_testloss, acc_overall, acc_perQuestionType, acc_perAnswerType
    elif mode == 'test-dev':
        filename = './result/vqa_OpenEnded_mscoco_test-dev2015_v3t'+str(it).zfill(8)+'_results'
        with open(filename+'.json', 'w') as f:
            json.dump(pred_list, f)
        if visualize:
            visualize_failures(stat_list,mode)
    elif mode == 'test':
        filename = './result/vqa_OpenEnded_mscoco_test2015_v3c'+str(it).zfill(8)+'_results'
        with open(filename+'.json', 'w') as f:
            json.dump(pred_list, f)
        if visualize:
            visualize_failures(stat_list,mode)

def drawgraph(results, save_question_type_graphs=False):
    # 0:it
    # 1:trainloss
    # 2:testloss
    # 3:oa_acc
    # 4:qt_acc
    # 5:at_acc

    # training curve
    it = np.array([l[0] for l in results])
    loss = np.array([l[1] for l in results])
    valloss = np.array([l[2] for l in results])
    valacc = np.array([l[3] for l in results])

    fig = plt.figure()
    ax1 = fig.add_subplot(111)
    ax2 = ax1.twinx()

    ax1.plot(it,loss, color='blue', label='train loss')
    ax1.plot(it,valloss, '--', color='blue', label='test loss')
    ax2.plot(it,valacc, color='red', label='acc on val')
    plt.legend(loc='lower left')

    ax1.set_xlabel('Iterations')
    ax1.set_ylabel('Loss Value')
    ax2.set_ylabel('Accuracy on Val [%]')

    plt.savefig('./learning_curve max_%2.2f.png'%valacc.max())
    plt.clf()
    plt.close("all")

    # question type
    it = np.array([l[0] for l in results])
    oa_acc = np.array([l[3] for l in results])
    qt_dic_list = [l[4] for l in results]

    def draw_qt_acc(target_key_list, figname):
        fig = plt.figure()
        for k in target_key_list:
            print k,type(k)
            t_val = np.array([ qt_dic[k] for qt_dic in qt_dic_list])
            plt.plot(it,t_val,label=str(k))
        plt.legend(fontsize='small')
        plt.ylim(0,100.)
        #plt.legend(prop={'size':6})

        plt.xlabel('Iterations')
        plt.ylabel('Accuracy on Val [%]')

        plt.savefig(figname,dpi=200)
        plt.clf()
        plt.close("all")

    if save_question_type_graphs:
        s_keys = sorted(qt_dic_list[0].keys())
        draw_qt_acc(s_keys[ 0:13]+[s_keys[31],],  './ind_qt_are.png')
        draw_qt_acc(s_keys[13:17]+s_keys[49:], './ind_qt_how_where_who_why.png')
        draw_qt_acc(s_keys[17:31]+[s_keys[32],],  './ind_qt_is.png')
        draw_qt_acc(s_keys[33:49],             './ind_qt_what.png')
        draw_qt_acc(['what color is the','what color are the','what color is',\
            'what color','what is the color of the'],'./qt_color.png')
        draw_qt_acc(['how many','how','how many people are',\
            'how many people are in'],'./qt_number.png')
        draw_qt_acc(['who is','why','why is the','where is the','where are the',\
            'which'],'./qt_who_why_where_which.png')
        draw_qt_acc(['what is the man','is the man','are they','is he',\
            'is the woman','is this person','what is the woman','is the person',\
            'what is the person'],'./qt_human.png')


