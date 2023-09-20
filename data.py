import numpy as np
import os
import cv2

def get_data(data_dir):
    img_dim = 256
    labels = ["NORMAL","PNEUMONIA"]
    data = []
    for label in labels:
        path = os.path.join(data_dir, label)
        for file in os.listdir(path):
            img = cv2.imread(os.path.join(path, file), cv2.IMREAD_GRAYSCALE)
            img_resize = cv2.resize(img, (img_dim, img_dim), interpolation=cv2.INTER_CUBIC)
            data.append([np.array(img_resize), labels.index(label)])

    return np.array(data, dtype=object)

def preprocess(data):
    #must be in form [(x,y),...]
    x_data = []
    y_data = []
    for x, y in data:
        x_data.append(x)
        y_data.append(y)
    return np.expand_dims(np.array(x_data), axis=-1), np.expand_dims(np.array(y_data), axis=-1)
