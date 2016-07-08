import sys
import os
import cv2
import numpy as np
import time
import caffe
import config

TARGET_IMG_SIZE = 224
if config.USE_LARGE_INPUT_IMAGES:
    TARGET_IMG_SIZE = 448

def trim_image(img, resnet_mean):
    y,x,c = img.shape
    if c != 3:
        raise Exception('There are gray scale image in data.')
    resized_img = cv2.resize( img, (TARGET_IMG_SIZE, TARGET_IMG_SIZE))
    if config.FLIP_IMAGE:
        resized_img = cv2.flip(resized_img, 1)
    transposed_img = np.transpose(resized_img,(2,0,1)).astype(np.float32)

    ivec = transposed_img - resnet_mean
    return ivec

def extract_features(target_data, output_data):
    target_path = os.path.join(config.COCO_IMAGE_PATH, target_data)
    output_path = os.path.join(config.OUTPUT_PATH, output_data)
    os.makedirs(output_path) # will raise exception if directory exists

    caffe.set_device(config.GPU_ID)
    caffe.set_mode_gpu()

    net = caffe.Net(config.RESNET_LARGE_PROTOTXT_PATH if config.USE_LARGE_INPUT_IMAGES else config.RESNET_PROTOTXT_PATH,
        config.RESNET_CAFFEMODEL_PATH, caffe.TEST)
    n_data = len(os.listdir(target_path))

    # mean substraction
    blob = caffe.proto.caffe_pb2.BlobProto()
    data = open( config.RESNET_MEAN_PATH , 'rb').read()
    blob.ParseFromString(data)
    resnet_mean = np.array( caffe.io.blobproto_to_array(blob)).astype(np.float32).reshape(3,224,224)
    if config.USE_LARGE_INPUT_IMAGES:
        resnet_mean = np.transpose(cv2.resize(np.transpose(resnet_mean,(1,2,0)), (448,448)),(2,0,1))

    print resnet_mean[:,:4,:4]
    cv2.imwrite('./resnet_mean.png',np.transpose(resnet_mean,(1,2,0)))

    for i,t_img_filename in enumerate(os.listdir(target_path)):
        
        if os.path.isfile(os.path.join(output_path, t_img_filename + '.npz')):
            continue

        t_img_path =  os.path.join(target_path, t_img_filename)
        img = cv2.imread(t_img_path)
        if img is None:
            print(t_img_path, "is None!")

        preprocessed_img = trim_image(img, resnet_mean)

        net.blobs['data'].data[0,...] = preprocessed_img
        t_start = time.time()
        net.forward()
        feature = net.blobs[config.EXTRACT_LAYER].data[0].reshape(config.EXTRACT_LAYER_SIZE)
        t_end = time.time()
        print '-------------------------', t_end - t_start

        output_file_path = os.path.join(output_path, t_img_filename + '.npz')

        t_start = time.time()
        np.savez_compressed(output_file_path, x=feature)
        t_end = time.time()
        print '-------------------------', t_end - t_start
        print '   index            : ', i, n_data
        print '   target file      : ', t_img_path
        print '   output file      : ', output_file_path
        #print '   image overview   : ', preprocessed_img[:,100,100]
        print '   shape after pre  : ', preprocessed_img.shape, preprocessed_img.mean()
        print '   shape of feature : ', feature.shape
        print '   argmax,min sum f : ', feature.argmax(), feature.argmin(), feature.sum()
        #print '   feature[:5]      : ', feature[:5]
        #print '   feature[-5:]     : ', feature[-5:]
    print("DONE")

if __name__ == '__main__':
    extract_features('train2014', config.OUTPUT_PREFIX + 'train2014')
    extract_features('val2014', config.OUTPUT_PREFIX + 'val2014')
    extract_features('test2015', config.OUTPUT_PREFIX + 'test2015')
    #extract_features('', config.OUTPUT_PREFIX + '')
