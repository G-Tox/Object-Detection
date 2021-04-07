import json

import cv2
import numpy as np
import requests

image = cv2.imread("123.jpg")

print(image.shape)

image = image/255.

image = np.expand_dims(image, axis=0)

print(image.shape)

data = json.dumps({"signature_name": "predict",
                   "inputs": {"images": image.tolist()}})

json_response = requests.post(
    "http://localhost:8500/v1/models/1:predict", data=data)


print(json_response.json())
