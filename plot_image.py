import cv2
import keras
from keras.applications.imagenet_utils import preprocess_input
from keras.backend.tensorflow_backend import set_session
from keras.models import Model
from keras.preprocessing import image
import matplotlib.pyplot as plt
import numpy as np
import pickle
from random import shuffle
from scipy.misc import imread
from scipy.misc import imresize
import tensorflow as tf

from ssd_v2 import SSD300v2
from ssd_training import MultiboxLoss
from ssd_utils import BBoxUtility
import os

np.set_printoptions(suppress=True)

config = tf.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.45
set_session(tf.Session(config=config))

NUM_CLASSES = 3 #21 #4
voc_classes = ['maru', 'sankaku', 'x']
#NUM_CLASSES = len(voc_classes) + 1
print(NUM_CLASSES)

#input_shape = (None, None, 3)
img_rows,img_cols=300, 300   #300, 300
input_shape = (img_rows,img_cols,3)   #224, 224, 3)


priors = pickle.load(open('prior_boxes_ssd300_2.pkl', 'rb'))  #???
#priors = pickle.load(open('prior_boxes_ssd224.pkl', 'rb'))  #???
print("priors=", priors.shape, "len(priors)",len(priors))
bbox_util = BBoxUtility(NUM_CLASSES, priors)

model = SSD300v2(input_shape, num_classes=NUM_CLASSES)
#model.load_weights('weights_SSD300.hdf5', by_name=True)
model.load_weights('./checkpoints/params_maruF_epoch0.hdf5', by_name=True)


inputs = []
images = []
path_prefix = './VOCdevkit/JPEGImages/'
gt = pickle.load(open('Original4x.pkl', 'rb'))
keys = sorted(gt.keys())
num_train = int(round(0.9 * len(keys))) #0.8
#train_keys = keys[:num_train]
val_keys = keys[num_train:]
#num_val = len(val_keys)

img_path = path_prefix + sorted(val_keys)[0]
img = image.load_img(img_path, target_size=(img_rows,img_cols))
img = image.img_to_array(img)
images.append(imread(img_path))
inputs.append(img.copy())
"""
img_path = './pics/001004.jpg'
img = image.load_img(img_path, target_size=(img_rows,img_cols))
img = image.img_to_array(img)
images.append(imread(img_path))
inputs.append(img.copy())

for i in range(1,9,1):
    img_path = './pics/00000'+str(i)+'.jpg'
    img = image.load_img(img_path, target_size=(img_rows,img_cols))
    img = image.img_to_array(img)
    images.append(imread(img_path))
    inputs.append(img.copy())
#inputs = preprocess_input(np.array(inputs))

img_path = './pics/test0.jpg'
img = image.load_img(img_path, target_size=(img_rows,img_cols))
img = image.img_to_array(img)
images.append(imread(img_path))
inputs.append(img.copy())
img_path = './pics/test1.jpg'
img = image.load_img(img_path, target_size=(img_rows,img_cols))
img = image.img_to_array(img)
images.append(imread(img_path))
inputs.append(img.copy())
img_path = './pics/test2.jpg'
img = image.load_img(img_path, target_size=(img_rows,img_cols))
img = image.img_to_array(img)
images.append(imread(img_path))
inputs.append(img.copy())
img_path = './pics/test3.jpg'
img = image.load_img(img_path, target_size=(img_rows,img_cols))
img = image.img_to_array(img)
images.append(imread(img_path))
inputs.append(img.copy())
img_path = './pics/test4.jpg'
img = image.load_img(img_path, target_size=(img_rows,img_cols))
img = image.img_to_array(img)
images.append(imread(img_path))
inputs.append(img.copy())
img_path = './pics/test5.jpg'
img = image.load_img(img_path, target_size=(img_rows,img_cols))
img = image.img_to_array(img)
images.append(imread(img_path))
inputs.append(img.copy())
"""

img_path = './pics/001004.jpg'
img = image.load_img(img_path, target_size=(img_rows,img_cols))
img = image.img_to_array(img)
images.append(imread(img_path))
inputs.append(img.copy())

inputs = preprocess_input(np.array(inputs))


preds = model.predict(inputs, batch_size=1, verbose=1)
results = bbox_util.detection_out(preds)

# In[5]:
count=0
for i, img in enumerate(images):
    # Parse the outputs.
    det_label = results[i][:, 0]
    det_conf = results[i][:, 1]
    det_xmin = results[i][:, 2]
    det_ymin = results[i][:, 3]
    det_xmax = results[i][:, 4]
    det_ymax = results[i][:, 5]

    # Get detections with confidence higher than 0.6.
    top_indices = [i for i, conf in enumerate(det_conf) if conf >= 0.6]

    top_conf = det_conf[top_indices]
    top_label_indices = det_label[top_indices].tolist()
    top_xmin = det_xmin[top_indices]
    top_ymin = det_ymin[top_indices]
    top_xmax = det_xmax[top_indices]
    top_ymax = det_ymax[top_indices]

    colors = plt.cm.hsv(np.linspace(0, 1, NUM_CLASSES)).tolist()

    plt.imshow(img / 255.)
    currentAxis = plt.gca()

    for i in range(top_conf.shape[0]):
        xmin = int(round(top_xmin[i] * img.shape[1]))
        ymin = int(round(top_ymin[i] * img.shape[0]))
        xmax = int(round(top_xmax[i] * img.shape[1]))
        ymax = int(round(top_ymax[i] * img.shape[0]))
        score = top_conf[i]
        label = int(top_label_indices[i])
        label_name = voc_classes[label - 1]
        display_txt = '{:0.2f}, {}'.format(score, label_name)

        #display_txt = '{:0.2f}, {}'.format(score, label)
        coords = (xmin, ymin), xmax-xmin+1, ymax-ymin+1
        color = colors[label]
        currentAxis.add_patch(plt.Rectangle(*coords, fill=False, edgecolor=color, linewidth=2))
        currentAxis.text(xmin, ymin, display_txt, bbox={'facecolor':color, 'alpha':0.5})

    plt.pause(1)
    print('i=',i)
    count +=1
    plt.savefig("recognized_picture{0:03d}_".format(count)+"_{0:03d}.png".format(top_conf.shape[0])) 
    plt.close()