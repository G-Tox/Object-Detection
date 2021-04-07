import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

import keras
from keras_retinanet import models
from keras_retinanet.utils.image import read_image_bgr, preprocess_image, resize_image

# import miscellaneous modules
import cv2
import numpy as np
import time

# set tf backend to allow memory to grow, instead of claiming everything
import tensorflow as tf

def get_session():
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    return tf.Session(config=config)

# use this environment flag to change which GPU to use
#os.environ["CUDA_VISIBLE_DEVICES"] = "1"

# set the modified tf session as backend in keras
keras.backend.tensorflow_backend.set_session(get_session())



from keras_retinanet.utils.visualization import draw_box, draw_caption
from keras_retinanet.utils.colors import label_color



# adjust this to point to your downloaded/trained model
model_path = os.path.join('snapshots', '21_resnet50_csv_06_inference.h5')

# load retinanet model
model = models.load_model(model_path,backbone_name='resnet50')
#print(model.summary())

# load label to names mapping for visualization purposes
labels_to_names = {0:'Casco',1:'Cabeza'}




cap = cv2.VideoCapture(0)
salida = cv2.VideoWriter('videoSalida.avi',cv2.VideoWriter_fourcc(*'XVID'),5.0,(640,480))

font = cv2.FONT_HERSHEY_PLAIN
starting_time = time.time()
frame_id = 0
counter = 0  
while True:
    ret, frame = cap.read()
    frame_id += 1

    if not ret:
        break
    bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
    
    # preprocess image for network
    image = preprocess_image(bgr)
    image, scale = resize_image(image)
    

    start = time.time()
    boxes, scores, labels = model.predict_on_batch(np.expand_dims(image, axis=0))
    
    t = time.time() - start
    
    
    # correct for image scale
    boxes /= scale
    label_1=[]
    scores_1=[]
    boxes_1=[]

    # visualize detections
    for box, score, label in zip(boxes[0], scores[0], labels[0]):
        # scores are sorted so we can break
        if score < 0.8:
            break
        
        boxes_1.append(box)
        scores_1.append(score)
        label_1.append(label)

    import pandas as pd
    df=pd.DataFrame()
    df['label']=label_1
    df['scores']=scores_1
    df['boxes']=boxes_1
        
    filtro=df['label']==1
    df_ro=df[filtro]
    filtro_c=df['label']==0
    df_ca=df[filtro_c]
    #filtro_d=df['label']==0
    #df_pe=df[filtro_d]


    for i in range(len(df_ca)):
        coorde=df_ca.iloc[i][2]
        dt_fr_ca=df_ca.iloc[i]
        
        color_c = label_color(dt_fr_ca[0])
        b_c = dt_fr_ca[2].astype(int)
        cv2.rectangle(frame, (b_c[0], b_c[1]), (b_c[2], b_c[3]), (255, 0, 0), 3)
        caption_c = "{} {:.3f}".format(labels_to_names[dt_fr_ca[0]], dt_fr_ca[1])
        cv2.putText(frame, caption_c, (b_c[2], b_c[3] - 10), cv2.FONT_HERSHEY_PLAIN, 1.5, (0, 0, 0), 3)
        cv2.putText(frame, caption_c, (b_c[2], b_c[3] - 10), cv2.FONT_HERSHEY_PLAIN, 1.5, (255, 0, 0), 2)
        
    for j in range(len(df_ro)):
        coorde_ro=df_ro.iloc[j][2]
        dt_fr_ro=df_ro.iloc[j]
                             
        color = label_color(dt_fr_ro[0])
        b = dt_fr_ro[2].astype(int)
        cv2.rectangle(frame, (b[0], b[1]), (b[2], b[3]), (255, 255, 0), 3)
        caption = "{} {:.3f}".format(labels_to_names[dt_fr_ro[0]], dt_fr_ro[1])
        capti = "{}".format(dt_fr_ro[2])
        cv2.putText(frame, caption, (b[0], b[1] - 10), cv2.FONT_HERSHEY_PLAIN, 1.5, (0, 0, 0), 3)
        cv2.putText(frame, caption, (b[0], b[1] - 10), cv2.FONT_HERSHEY_PLAIN, 1.5, (255, 255, 0), 2) 
        #cv2.putText(frame, capti, (b[0], b[1] + 20), cv2.FONT_HERSHEY_PLAIN, 1.5, (255, 255, 255), 2)    


    #for j in range(len(df_pe)):
    #    coorde_pe=df_pe.iloc[j][2]
    #    dt_fr_pe=df_pe.iloc[j]
    #                         
    #    color = label_color(dt_fr_pe[0])
    #    b = dt_fr_pe[2].astype(int)
    #    cv2.rectangle(frame, (b[0], b[1]), (b[2], b[3]), (150, 50, 150), 3)
    #    caption_pe = "{} {:.3f}".format(labels_to_names[dt_fr_pe[0]], dt_fr_pe[1])
    #    cv2.putText(frame, caption_pe, (b[0], b[1] - 10), cv2.FONT_HERSHEY_PLAIN, 1.5, (0, 0, 0), 3)
    #    cv2.putText(frame, caption_pe, (b[0], b[1] - 10), cv2.FONT_HERSHEY_PLAIN, 1.5, (150, 50, 150), 2)

            
    for k in range(len(df_ro)):
        coorde_ro=df_ro.iloc[k][2]
        dt_fr_ro=df_ro.iloc[k]
        texto=''
        for l in range(len(df_ca)):
            valor=True
            coorde=df_ca.iloc[l][2]

            if ((coorde_ro[0]-5)<coorde[0]) and ((coorde_ro[1]-5) <coorde[1]) and ((coorde_ro[2]+5) > coorde[2]) and (coorde_ro[3]>coorde[3]):
                texto='Tiene el casco'
                break

            else:
                texto='No tiene'
            

        b_d = dt_fr_ro[2].astype(int)

        if texto == 'Tiene el casco':
            cv2.putText(frame, texto, (b_d[0], b_d[1] - 30), cv2.FONT_HERSHEY_PLAIN, 1.5, (0, 0, 0), 3)
            cv2.putText(frame, texto, (b_d[0], b_d[1] - 30), cv2.FONT_HERSHEY_PLAIN, 1.5, (0, 255, 0), 2)
        elif texto == 'No tiene':
            cv2.putText(frame, texto, (b_d[0], b_d[1] - 30), cv2.FONT_HERSHEY_PLAIN, 1.5, (0, 0, 0), 3)
            cv2.putText(frame, texto, (b_d[0], b_d[1] - 30), cv2.FONT_HERSHEY_PLAIN, 1.5, (0, 0, 255), 2)  


        if texto == '':
            cv2.imwrite('tmp/img%08d.jpg'%counter,frame)

        elif texto=='No tiene':
            cv2.imwrite('tmp/img%08d.jpg'%counter,frame)

        counter += 1


            

    elapsed_time = time.time() - starting_time
    fps = frame_id / elapsed_time
    
    cv2.imshow("Image", frame)
    salida.write(frame)
    #key = cv2.waitKey(1)
    #if key == 27:
    #    break

    if cv2.waitKey(1) & 0xFF == ord('q'):
          break

cap.release()
salida.release()
cv2.destroyAllWindows()