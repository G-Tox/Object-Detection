#!/usr/bin/env python
from __future__ import print_function
import numpy as np
import pandas
import cv2
import tensorflow as tf

from grpc.beta import implementations
from tensorflow_serving.apis import predict_pb2
from tensorflow_serving.apis import prediction_service_pb2

from keras_retinanet.utils.colors import label_color


def main(image): 
  # create prediction service client stub
  channel = implementations.insecure_channel('0.0.0.0', int('8500'))
  stub = prediction_service_pb2.beta_create_PredictionService_stub(channel)
  #channel = grpc.insecure_channel('retinanet:8500')  # localhost:8500 in your case
  #stub = prediction_service_pb2_grpc.PredictionServiceStub(channel)
  
  # create request
  request = predict_pb2.PredictRequest()
  request.model_spec.name = 'retinanet'
  request.model_spec.signature_name = 'predict'
  
  # read image into numpy array
  #img = cv2.imread(image).astype(np.float32)
  img = image
  # convert to tensor proto and make request
  # shape is in NHWC (num_samples x height x width x channels) format
  tensor = tf.contrib.util.make_tensor_proto(img, shape=[1]+list(img.shape))
  request.inputs['images'].CopyFrom(tensor)
  resp = stub.Predict(request, 25.0)
  np_bound = np.array(resp.outputs['output1'].float_val)
  np_acc = np.array(resp.outputs['output2'].float_val)
  np_class = np.array(resp.outputs['output3'].int_val)

  print(np_class)

  arr=[]
  i=0
  while i < len(np_bound):
     varia=np_bound[i:i+4].tolist()
     i=i+4
     arr.append(varia)
  arrt=np.array(arr)

  df=pandas.DataFrame()
  df['label']=np_class
  df['scores']=np_acc
  df['boxes']=arr

  #filtro_acc=df_p['scores']>0.75
  #df=df_p[filtro_acc]

  filtro=df['label']==1
  filtro_1=df['scores']>0.20
  #filtro_2=df['boxes'][0]
  df_ro=df[filtro&filtro_1]

  filtro_c=df['label']==0
  filtro_c_1=df['scores']>0.2
  df_ca=df[filtro_c&filtro_c_1]

  labels_to_names = {0:'Casco',1:'Cabeza'}

  for i in range(len(df_ca)):
      coorde=df_ca.iloc[i][2]
      dt_fr_ca=df_ca.iloc[i]
        
      color_c = label_color(dt_fr_ca[0])
      b_c = np.array(dt_fr_ca[2],dtype=np.int)
      cv2.rectangle(img, (b_c[0], b_c[1]), (b_c[2], b_c[3]), (255, 0, 0), 3)
      caption_c = "{} {:.3f}".format(labels_to_names[dt_fr_ca[0]], dt_fr_ca[1])
      cv2.putText(img, caption_c, (b_c[2], b_c[3] - 10), cv2.FONT_HERSHEY_PLAIN, 1.5, (0, 0, 0), 3)
      cv2.putText(img, caption_c, (b_c[2], b_c[3] - 10), cv2.FONT_HERSHEY_PLAIN, 1.5, (255, 0, 0), 2)
        
  for j in range(len(df_ro)):
      coorde_ro=df_ro.iloc[j][2]
      dt_fr_ro=df_ro.iloc[j]
                           
      color = label_color(dt_fr_ro[0])
      b = np.array(dt_fr_ro[2],dtype=np.int)
      cv2.rectangle(img, (b[0], b[1]), (b[2], b[3]), (255, 255, 0), 3)
      caption = "{} {:.3f}".format(labels_to_names[dt_fr_ro[0]], dt_fr_ro[1])
      cv2.putText(img, caption, (b[0], b[3] - 10), cv2.FONT_HERSHEY_PLAIN, 1.5, (0, 0, 0), 3)
      cv2.putText(img, caption, (b[0], b[3] - 10), cv2.FONT_HERSHEY_PLAIN, 1.5, (255, 255, 0), 2)

  #imagen_t=image.split('/')
  #ruta='static/imagenes1/'+imagen_t[2]

  #cv2.imwrite(ruta,img)

  texto='NO se reconoce'
  for k in range(len(df_ro)):
     coorde_ro=np.array(df_ro.iloc[k][2],dtype=np.int)
     dt_fr_ro=df_ro.iloc[k]
     texto='NO tiene casco' 
     for l in range(len(df_ca)):
        coorde=np.array(df_ca.iloc[l][2],dtype=np.int)
        if ((coorde_ro[0]-5)<coorde[0]) and ((coorde_ro[1]-5)<coorde[1]) and ((coorde_ro[2]+5)>coorde[2]) and ((coorde_ro[3])>coorde[3]):
           texto='Tiene el casco puesto'
           break
        else:
           texto='NO tiene el casco puesto'

     b_d = np.array(dt_fr_ro[2],dtype=np.int)
     if texto == 'Tiene el casco puesto':
        cv2.putText(img, texto, (b_d[0], b_d[1]+30), cv2.FONT_HERSHEY_PLAIN, 1.5, (0, 0, 0), 3)
        cv2.putText(img, texto, (b_d[0], b_d[1]+30), cv2.FONT_HERSHEY_PLAIN, 1.5, (0, 255, 0), 2)
     elif texto == 'NO tiene el casco puesto':
        cv2.putText(img, texto, (b_d[0], b_d[1]+30), cv2.FONT_HERSHEY_PLAIN, 1.5, (0, 0, 0), 3)
        cv2.putText(img, texto, (b_d[0], b_d[1]+30), cv2.FONT_HERSHEY_PLAIN, 1.5, (0, 0, 255), 2)

  text1=texto 
    
  return img,text1



