"""
Generates predictions on test-dev or test using an ensemble of nets. The
ensemble is produced using the average of the pre-softmax output from each net.

Place each model in its own folder. The folder must contain:

- The .caffemodel file
- proto_test.prototxt
- adict.json
- vdict.json
- aux.json

aux.json should contain the following keys:

- batch_size (value should be integer)
- data_shape (value should be array of integer)
- img_feature_prefix (value should be string)
- spatial_coord (value should be boolean)
- glove (value should be boolean)

If the folder also contains "preds.pkl", evaluation is skipped for that network.

"""

import caffe
import numpy as np
import cPickle
import argparse, os, glob
import sys
import json
from collections import defaultdict
import vqa_data_provider_layer
from vqa_data_provider_layer import LoadVQADataProvider

def verify_all(folder_paths):
    """
    Calls verify_one on each folder path. Also checks to make sure all the
    answer vocabularies are the same.
    """
    adict_paths = []
    for folder_path in folder_paths:
        paths = verify_one(folder_path)
        adict_paths.append(paths[2])
    adicts = []
    for path in adict_paths:
        with open(path, 'r') as f:
            adict = json.load(f)
            adicts.append(adict)
    if len(adicts) > 1:
        for a2 in adicts[1:]:
            if set(adicts[0].keys()) != set(a2.keys()):
                print set(adicts[0].keys()) - set(a2.keys())
                print set(a2.keys()) - set(adicts[0].keys())
                raise Exception('Answer vocab mismatch')
    return adicts

def verify_one(folder_path):
    """
    Makes sure all the required files exist in the folder. If so, returns the
    paths to all the files.
    """
    model_path = glob.glob(folder_path + '/*.caffemodel')
    assert len(model_path) == 1, 'one .caffemodel per folder, please'
    model_path = model_path[0]
    proto_path = folder_path + '/proto_test.prototxt'
    adict_path = folder_path + '/adict.json'
    vdict_path = folder_path + '/vdict.json'
    aux_path = folder_path + '/aux.json'
    assert os.path.exists(proto_path), 'proto_test.prototxt missing'
    assert os.path.exists(adict_path), 'adict.json missing'
    assert os.path.exists(vdict_path), 'vdict.json missing'
    assert os.path.exists(aux_path), 'aux.json missing'
    with open(aux_path, 'r') as f:
        aux = json.load(f)
    batch_size = int(aux['batch_size'])
    data_shape = tuple(map(int, aux['data_shape']))
    img_feature_prefix = aux['img_feature_prefix']
    spatial_coord = aux['spatial_coord'] if 'spatial_coord' in aux else False
    glove = aux['glove'] if 'glove' in aux else False
    return model_path, proto_path, adict_path, vdict_path, batch_size, data_shape, img_feature_prefix, spatial_coord, glove

def get_pkl_fname(ques_file):
    if '_val2014_' in ques_file:
        return '/preds_val.pkl'
    elif '_test-dev2015_' in ques_file:
        return '/preds_test_dev.pkl'
    elif '_test2015_' in ques_file:
        return '/preds_test.pkl'
    else:
        raise NotImplementedError

def eval_one(folder_path, gpuid, ques_file):
    """
    Evaluates a single model (in folder_path) on the questions in ques_file.
    Returns an array of (QID, answer vector) tuples.
    """

    model_path, proto_path, adict_path, vdict_path, batch_size, data_shape, \
                    img_feature_prefix, spatial_coord, glove = verify_one(folder_path)
    
    dp = LoadVQADataProvider(ques_file, img_feature_prefix, vdict_path, \
        adict_path, mode='test', batchsize=batch_size, data_shape=data_shape)
    total_questions = len(dp.getQuesIds())
    print total_questions, 'total questions'

    if os.path.exists(folder_path + get_pkl_fname(ques_file)):
        print 'Found existing prediction file, trying to load...'
        with open(folder_path + get_pkl_fname(ques_file), 'r') as f:
            preds = cPickle.load(f)
        if len(preds) >= total_questions:
            print 'Loaded.'
            return preds
        else:
            print 'Number of saved answers does not match number of questions, continuing...'

    caffe.set_device(gpuid)
    caffe.set_mode_gpu()

    vqa_data_provider_layer.CURRENT_DATA_SHAPE = data_shape # This is a huge hack
    vqa_data_provider_layer.SPATIAL_COORD = spatial_coord
    vqa_data_provider_layer.GLOVE = glove

    net = caffe.Net(proto_path, model_path, caffe.TEST)

    print 'Model loaded:', model_path
    print 'Image feature prefix:', img_feature_prefix
    sys.stdout.flush()


    pred_layers = []

    epoch = 0
    while epoch == 0:
        t_word, t_cont, t_img_feature, t_answer, t_glove_matrix, t_qid_list, _, epoch = dp.get_batch_vec()
        net.blobs['data'].data[...] = np.transpose(t_word,(1,0))
        net.blobs['cont'].data[...] = np.transpose(t_cont,(1,0))
        net.blobs['img_feature'].data[...] = t_img_feature
        net.blobs['label'].data[...] = t_answer # dummy
        if glove:
            net.blobs['glove'].data[...] = np.transpose(t_glove_matrix, (1,0,2))
        net.forward()
        ans_matrix = net.blobs['prediction'].data

        for i in range(len(t_qid_list)):
            qid = t_qid_list[i]
            pred_layers.append((qid, np.copy(ans_matrix[i]))) # tricky!

        percent = 100 * float(len(pred_layers)) / total_questions
        sys.stdout.write('\r' + ('%.2f' % percent) + '%')
        sys.stdout.flush()

    #print 'Saving predictions...'
    #with open(folder_path + get_pkl_fname(ques_file), 'w') as f:
    #   cPickle.dump(pred_layers, f, protocol=-1)
    #print 'Saved.'

    return pred_layers

def make_rev_adict(adict):
    """
    An adict maps text answers to neuron indices. A reverse adict maps neuron
    indices to text answers.
    """
    rev_adict = {}
    for k,v in adict.items():
        rev_adict[v] = k
    return rev_adict

def softmax(arr):
    e = np.exp(arr)
    dist = e / np.sum(e)
    return dist

def get_qid_valid_answer_dict(ques_file, adict):
    """
    Returns a dictionary mapping question IDs to valid neuron indices.
    """
    print 'Multiple choice mode: making valid answer dictionary...'
    valid_answer_dict = {}
    with open(ques_file, 'r') as f:
        qdata = json.load(f)
        for q in qdata['questions']:
            valid_answer_dict[q['question_id']] = q['multiple_choices']
    for qid in valid_answer_dict:
        answers = valid_answer_dict[qid]
        valid_indices = []
        for answer in answers:
            if answer in adict:
                valid_indices.append(adict[answer])
        if len(valid_indices) == 0:
            print "we won't be able to answer qid", qid
        valid_answer_dict[qid] = valid_indices
    return valid_answer_dict

def dedupe(arr):
    print 'Deduping arr of len', len(arr)
    deduped = []
    seen = set()
    for qid, pred in arr:
        if qid not in seen:
            seen.add(qid)
            deduped.append((qid, pred))
    print 'New len', len(deduped)
    return deduped

def reorder_one(predictions, this_adict, canonical_adict):
    index_map = {}
    for idx, word in make_rev_adict(this_adict).iteritems():
        index_map[int(idx)] = int(canonical_adict[word])
    index_array = np.zeros(len(index_map), dtype=int)
    for src_idx, dest_idx in index_map.iteritems():
        index_array[src_idx] = dest_idx
    reordered = []
    for qid, output in predictions:
        reordered.append((qid, np.copy(output[index_array])))
    return reordered

def reorder_predictions(predictions, adicts):
    """
    Reorders prediction matrices so that the unit order matches that of the
    first answer dictionary.
    """
    if len(adicts) == 1:
        return predictions
    need_to_reorder = False
    for a2 in adicts[1:]:
        if adicts[0] != a2:
            need_to_reorder = True
    print 'Reordering...' if need_to_reorder else 'No need to reorder!'
    if not need_to_reorder:
        return predictions
    reordered = []
    for i in range(1, len(adicts)):
        if adicts[0] != adicts[i]:
            reordered.append(reorder_one(predictions[i], adicts[i], adicts[0]))
        else:
            reordered.append(predictions[i])
    return reordered

def average_outputs(arr_of_arr, rev_adict, qid_valid_answer_dict):
    """
    Given a list of lists, where each list contains (QID, answer vector) tuples,
    returns a single dictionary which maps a question ID to the text answer.
    """
    print 'Averaging outputs...'
    merged = defaultdict(list)
    for arr in arr_of_arr:
        for qid, ans_vec in arr:
            merged[qid].append(ans_vec)

    merged = {qid: softmax(np.vstack(ans_vecs).mean(axis=0)) for qid, ans_vecs in merged.iteritems()}
    mask_len = len(merged.values()[0])

    # Multiple choice filtering
    if qid_valid_answer_dict is not None:
        for qid in merged:
            valid_indices = qid_valid_answer_dict[qid]
            mask = np.zeros(mask_len)
            for idx in valid_indices:
                mask[idx] = 1
            merged[qid] *= mask

    merged = {qid: rev_adict[ans_vec.argmax()] for qid, ans_vec in merged.iteritems()}

    return merged

def save_json(qid_ans_dict, fname):
    tmp = []
    for qid, ans in qid_ans_dict.iteritems():
        tmp.append({u'answer': ans, u'question_id': qid})
    with open(fname, 'w') as f:
        json.dump(tmp, f)
    print 'Saved to', fname

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--ques_file', required=True)
    parser.add_argument('--gpu', type=int, required=True)
    parser.add_argument('--out_file', required=True)
    parser.add_argument('folders', nargs='*',
        help='space-separated list of folders containing models')
    args = parser.parse_args()
    assert len(args.folders) > 0, 'please specify at least one folder'
    print 'Folders', args.folders

    adicts = verify_all(args.folders)
    qid_valid_answer_dict = None
    if 'MultipleChoice' in args.ques_file:
        qid_valid_answer_dict = get_qid_valid_answer_dict(args.ques_file, adicts[0])

    arr_of_arr = [eval_one(folder_path, args.gpu, args.ques_file) for folder_path in args.folders]
    arr_of_arr = [dedupe(x) for x in arr_of_arr]
    reordered = reorder_predictions(arr_of_arr, adicts)
    qid_ans_dict = average_outputs(reordered, make_rev_adict(adicts[0]), qid_valid_answer_dict)
    save_json(qid_ans_dict, args.out_file)

if __name__ == '__main__':
    main()
