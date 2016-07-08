from flask import Flask, request, redirect, url_for, jsonify, send_from_directory
from time import time
import cv2
import hashlib
import caffe
import vqa_data_provider_layer
from vqa_data_provider_layer import LoadVQADataProvider
import numpy as np
import os
from skimage.transform import resize

# constants
GPU_ID = 3
RESNET_MEAN_PATH = "../00_data_preprocess/ResNet_mean.binaryproto"
RESNET_LARGE_PROTOTXT_PATH = "../00_data_preprocess/ResNet-152-448-deploy.prototxt"
RESNET_CAFFEMODEL_PATH = "/x/daylen/ResNet-152-model.caffemodel"
EXTRACT_LAYER = "res5c"
EXTRACT_LAYER_SIZE = (2048, 14, 14)
TARGET_IMG_SIZE = 448
VQA_PROTOTXT_PATH = "/x/daylen/saved_models/multi_att_2_glove/proto_test_batchsize1.prototxt"
VQA_CAFFEMODEL_PATH = "/x/daylen/saved_models/multi_att_2_glove/_iter_190000.caffemodel"
VDICT_PATH = "/x/daylen/saved_models/multi_att_2_glove/vdict.json"
ADICT_PATH = "/x/daylen/saved_models/multi_att_2_glove/adict.json"

ALLOWED_EXTENSIONS = set(['jpg', 'jpeg', 'JPG', 'JPEG', 'png', 'PNG'])
UPLOAD_FOLDER = './uploads/'
VIZ_FOLDER = './viz/'

# global variables
app = Flask(__name__, static_url_path='')
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
resnet_mean = None
resnet_net = None
vqa_net = None
feature_cache = {}
vqa_data_provider = LoadVQADataProvider(VDICT_PATH, ADICT_PATH, batchsize=1, \
    mode='test', data_shape=EXTRACT_LAYER_SIZE)

# helpers
def setup():
    global resnet_mean
    global resnet_net
    global vqa_net
    # data provider
    vqa_data_provider_layer.CURRENT_DATA_SHAPE = EXTRACT_LAYER_SIZE

    # mean substraction
    blob = caffe.proto.caffe_pb2.BlobProto()
    data = open( RESNET_MEAN_PATH , 'rb').read()
    blob.ParseFromString(data)
    resnet_mean = np.array( caffe.io.blobproto_to_array(blob)).astype(np.float32).reshape(3,224,224)
    resnet_mean = np.transpose(cv2.resize(np.transpose(resnet_mean,(1,2,0)), (448,448)),(2,0,1))

    # resnet
    caffe.set_device(GPU_ID)
    caffe.set_mode_gpu()

    resnet_net = caffe.Net(RESNET_LARGE_PROTOTXT_PATH, RESNET_CAFFEMODEL_PATH, caffe.TEST)

    # our net
    vqa_net = caffe.Net(VQA_PROTOTXT_PATH, VQA_CAFFEMODEL_PATH, caffe.TEST)

    # uploads
    if not os.path.exists(UPLOAD_FOLDER):
        os.makedirs(UPLOAD_FOLDER)

    if not os.path.exists(VIZ_FOLDER):
        os.makedirs(VIZ_FOLDER)

    print 'Finished setup'

def trim_image(img):
    y,x,c = img.shape
    if c != 3:
        raise Exception('Expected 3 channels in the image')
    resized_img = cv2.resize( img, (TARGET_IMG_SIZE, TARGET_IMG_SIZE))
    transposed_img = np.transpose(resized_img,(2,0,1)).astype(np.float32)
    ivec = transposed_img - resnet_mean
    return ivec

def make_rev_adict(adict):
    """
    An adict maps text answers to neuron indices. A reverse adict maps neuron
    indices to text answers.
    """
    rev_adict = {}
    for k,v in adict.items():
        rev_adict[v] = k
    return rev_adict

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1] in ALLOWED_EXTENSIONS

def softmax(arr):
    e = np.exp(arr)
    dist = e / np.sum(e)
    return dist

def downsample_image(img):
    img_h, img_w, img_c = img.shape
    img = resize(img, (448 * img_h / img_w, 448))
    return img

def save_attention_visualization(source_img_path, att_map, dest_name):
    """
    Visualize the attention map on the image and save the visualization.
    """
    img = cv2.imread(source_img_path) # cv2.imread does auto-rotate

    # downsample source image
    img = downsample_image(img)
    img_h, img_w, img_c = img.shape

    _, att_h, att_w = att_map.shape
    att_map = att_map.reshape((att_h, att_w))

    # upsample attention map to match original image
    upsample0 = resize(att_map, (img_h, img_w), order=3) # bicubic interpolation
    upsample0 = upsample0 / upsample0.max()

    # create rgb-alpha 
    rgba0 = np.zeros((img_h, img_w, img_c + 1))
    rgba0[..., 0:img_c] = img
    rgba0[..., 3] = upsample0

    path0 = os.path.join(VIZ_FOLDER, dest_name + '.png')
    cv2.imwrite(path0, rgba0 * 255.0)

    return path0

# routes
@app.route('/', methods=['GET'])
def index():
    return app.send_static_file('demo2.html')

@app.route('/api/upload_image', methods=['POST'])
def upload_image():
    file = request.files['file']
    if not file:
        return jsonify({'error': 'No file was uploaded.'})
    if allowed_file(file.filename):
        start = time()
        file_hash = hashlib.md5(file.read()).hexdigest()
        if file_hash in feature_cache:
            json = {'img_id': file_hash, 'time': time() - start}
            return jsonify(json)
        save_path = os.path.join(app.config['UPLOAD_FOLDER'], file_hash + '.jpg')
        file.seek(0)
        file.save(save_path)
        img = cv2.imread(save_path)
        if img is None:
            return jsonify({'error': 'Error reading image.'})
        small_img = downsample_image(img)
        cv2.imwrite(save_path, small_img * 255.0)
        preprocessed_img = trim_image(img)
        resnet_net.blobs['data'].data[0,...] = preprocessed_img
        resnet_net.forward()
        feature = resnet_net.blobs[EXTRACT_LAYER].data[0].reshape(EXTRACT_LAYER_SIZE)
        feature = ( feature / np.sqrt((feature**2).sum()) )
        feature_cache[file_hash] = feature
        json = {'img_id': file_hash, 'time': time() - start}
        return jsonify(json)
    else:
        return jsonify({'error': 'Please upload a JPG or PNG.'})

@app.route('/api/upload_question', methods=['POST'])
def upload_question():
    img_hash = request.form['img_id']
    if img_hash not in feature_cache:
        return jsonify({'error': 'Unknown image ID. Try uploading the image again.'})
    start = time()
    img_feature = feature_cache[img_hash]
    question = request.form['question']
    img_ques_hash = hashlib.md5(img_hash + question).hexdigest()
    qvec, cvec, avec, glove_matrix = vqa_data_provider.create_batch(question)
    vqa_net.blobs['data'].data[...] = np.transpose(qvec,(1,0))
    vqa_net.blobs['cont'].data[...] = np.transpose(cvec,(1,0))
    vqa_net.blobs['img_feature'].data[...] = img_feature
    vqa_net.blobs['label'].data[...] = avec # dummy
    vqa_net.blobs['glove'].data[...] = np.transpose(glove_matrix, (1,0,2))
    vqa_net.forward()
    scores = vqa_net.blobs['prediction'].data.flatten()

    # attention visualization
    att_map = vqa_net.blobs['att_map0'].data.copy()[0]
    source_img_path = os.path.join(app.config['UPLOAD_FOLDER'], img_hash + '.jpg')
    path0 = save_attention_visualization(source_img_path, att_map, img_ques_hash)

    scores = softmax(scores)
    top_indices = scores.argsort()[::-1][:5]
    top_answers = [vqa_data_provider.vec_to_answer(i) for i in top_indices]
    top_scores = [float(scores[i]) for i in top_indices]

    json = {'answer': top_answers[0],
        'answers': top_answers,
        'scores': top_scores,
        'viz': [path0],
        'time': time() - start}
    return jsonify(json)

@app.route('/viz/<filename>')
def get_visualization(filename):
    return send_from_directory(VIZ_FOLDER, filename)

if __name__ == '__main__':
    setup()
    app.run(host='0.0.0.0', port=5000, debug=False)
