from flask import Flask, render_template
from flask_socketio import SocketIO, emit
import io
import base64
#from PIL import Image
import cv2
import numpy as np
import imutils
from imageio import imread
from prueba import main

app = Flask(__name__)
socketio = SocketIO(app)

@app.route('/', methods=['POST', 'GET'])
def index():
    return render_template('index.html')

@socketio.on('image')
def image(data_image):

    # decode and convert into image

    #image_2 = base64.decodebytes(data_image)
    #image_64_encode = base64.b64encode(image_2).decode('ascii')

    pimg = imread(io.BytesIO(base64.b64decode(data_image)))



    #b = io.BytesIO(base64.b64decode(data_image))

    #pimg = Image.open(b)
    

    ## converting RGB to BGR, as opencv standards
    frame = cv2.cvtColor(np.array(pimg,dtype='float32'), cv2.COLOR_RGB2BGR)
    frame = cv2.flip(frame, 1)
    print('========',type(frame))
    frame,texto = main(frame)
    print('{{{{{{{{{{{{{{{{{{{{{{{{{{',texto)

    # Process the image frame
    frame = imutils.resize(frame, width=700)
    #
    imgencode = cv2.imencode('.jpg', frame)[1]

    # base64 encode
    stringData = base64.b64encode(imgencode).decode('utf-8')
    b64_src = 'data:image/jpg;base64,'
    stringData = b64_src + stringData

    # emit the frame back
    emit('response_back', stringData)


if __name__ == '__main__':
    socketio.run(app, host='0.0.0.0')
