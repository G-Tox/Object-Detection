#!/usr/bin/env python
from __future__ import print_function
import argparse
import numpy as np
import time
tt = time.time()

import cv2
import tensorflow as tf

from grpc.beta import implementations
from tensorflow_serving.apis import predict_pb2
from tensorflow_serving.apis import prediction_service_pb2

parser = argparse.ArgumentParser(description='incetion grpc client flags.')
parser.add_argument('--host', default='localhost', help='inception serving host')
parser.add_argument('--port', default='8500', help='inception serving port')
parser.add_argument('--image', default='', help='path to JPEG image file')
FLAGS = parser.parse_args()

def main(): 
  # create prediction service client stub
  channel = implementations.insecure_channel(FLAGS.host, int(FLAGS.port))
  stub = prediction_service_pb2.beta_create_PredictionService_stub(channel)
  
  # create request
  request = predict_pb2.PredictRequest()
  request.model_spec.name = 'retinanet'
  request.model_spec.signature_name = 'predict'
  
  # read image into numpy array
  img = cv2.imread(FLAGS.image).astype(np.float32)
  
  # convert to tensor proto and make request
  # shape is in NHWC (num_samples x height x width x channels) format
  tensor = tf.contrib.util.make_tensor_proto(img, shape=[1]+list(img.shape))
  request.inputs['images'].CopyFrom(tensor)
  resp = stub.Predict(request, 2.0)
  
  print('total time: {}s'.format(time.time() - tt))
  print(resp)
    
if __name__ == '__main__':
    main()

