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

#from ssd_v2 import SSD
from ssd300VGG16_alt import SSD
from ssd_training import MultiboxLoss
from ssd_utils import BBoxUtility
import os

def save_history(history, result_file,epochs):
    loss = history.history['loss']
    #conv_loss = history.history['conv_out_loss']
    #acc = history.history['conv_out_acc']
    acc = history.history['acc']
    val_loss = history.history['val_loss']
    #val_conv_loss = history.history['val_out_recon_loss']
    #val_acc = history.history['val_conv_out_acc']
    val_acc = history.history['val_acc']
    nb_epoch = len(acc)

    with open(result_file, "a") as fp:
        if epochs==0:
            fp.write("i\tloss\tacc\tval_loss\tval_acc\n")
            for i in range(nb_epoch):
                fp.write("%d\t%f\t%f\t%f\t%f\n" % (epochs, loss[i],acc[i], val_loss[i],val_acc[i]))
        else:
            for i in range(nb_epoch):
                fp.write("%d\t%f\t%f\t%f\t%f\n" % (epochs, loss[i], acc[i], val_loss[i], val_acc[i]))
                         
plt.rcParams['figure.figsize'] = (8, 8)
plt.rcParams['image.interpolation'] = 'nearest'

np.set_printoptions(suppress=True)

# 21
NUM_CLASSES = 21 #4
#input_shape = (None, None, 3)
img_rows,img_cols=300, 300   #300, 300
input_shape = (img_rows,img_cols,3)   #224, 224, 3)

priors = pickle.load(open('prior_boxes_ssd300.pkl', 'rb'))  #???
#priors = pickle.load(open('prior_boxes_ssd224.pkl', 'rb'))  #???
print("priors=", priors.shape, "len(priors)",len(priors))
bbox_util = BBoxUtility(NUM_CLASSES, priors)

# gt = pickle.load(open('gt_pascal.pkl', 'rb'))
gt = pickle.load(open('VOC2007.pkl', 'rb'))
#gt = pickle.load(open('Original1.pkl', 'rb'))
#gt = pickle.load(open('Original4x.pkl', 'rb'))
keys = sorted(gt.keys())
num_train = int(round(0.9 * len(keys))) #0.8
train_keys = keys[:num_train]
val_keys = keys[num_train:]
num_val = len(val_keys)

class Generator(object):
    def __init__(self, gt, bbox_util,
                 batch_size, path_prefix,
                 train_keys, val_keys, image_size,
                 saturation_var=0.5,
                 brightness_var=0.5,
                 contrast_var=0.5,
                 lighting_std=0.5,
                 hflip_prob=0.5,
                 vflip_prob=0.5,
                 do_crop=True,
                 crop_area_range=[0.75, 1.0],
                 aspect_ratio_range=[3./4., 4./3.]):
        self.gt = gt
        self.bbox_util = bbox_util
        self.batch_size = batch_size
        self.path_prefix = path_prefix
        self.train_keys = train_keys
        self.val_keys = val_keys
        self.train_batches = len(train_keys)
        self.val_batches = len(val_keys)
        self.image_size = image_size
        self.color_jitter = []
        if saturation_var:
            self.saturation_var = saturation_var
            self.color_jitter.append(self.saturation)
        if brightness_var:
            self.brightness_var = brightness_var
            self.color_jitter.append(self.brightness)
        if contrast_var:
            self.contrast_var = contrast_var
            self.color_jitter.append(self.contrast)
        self.lighting_std = lighting_std
        self.hflip_prob = hflip_prob
        self.vflip_prob = vflip_prob
        self.do_crop = do_crop
        self.crop_area_range = crop_area_range
        self.aspect_ratio_range = aspect_ratio_range

    def grayscale(self, rgb):
        return rgb.dot([0.299, 0.587, 0.114])

    def saturation(self, rgb):
        gs = self.grayscale(rgb)
        alpha = 2 * np.random.random() * self.saturation_var 
        alpha += 1 - self.saturation_var
        rgb = rgb * alpha + (1 - alpha) * gs[:, :, None]
        return np.clip(rgb, 0, 255)

    def brightness(self, rgb):
        alpha = 2 * np.random.random() * self.brightness_var 
        alpha += 1 - self.saturation_var
        rgb = rgb * alpha
        return np.clip(rgb, 0, 255)

    def contrast(self, rgb):
        gs = self.grayscale(rgb).mean() * np.ones_like(rgb)
        alpha = 2 * np.random.random() * self.contrast_var 
        alpha += 1 - self.contrast_var
        rgb = rgb * alpha + (1 - alpha) * gs
        return np.clip(rgb, 0, 255)

    def lighting(self, img):
        cov = np.cov(img.reshape(-1, 3) / 255.0, rowvar=False)
        eigval, eigvec = np.linalg.eigh(cov)
        noise = np.random.randn(3) * self.lighting_std
        noise = eigvec.dot(eigval * noise) * 255
        img += noise
        return np.clip(img, 0, 255)

    def horizontal_flip(self, img, y):
        if np.random.random() < self.hflip_prob:
            img = img[:, ::-1]
            y[:, [0, 2]] = 1 - y[:, [2, 0]]
        return img, y

    def vertical_flip(self, img, y):
        if np.random.random() < self.vflip_prob:
            img = img[::-1]
            y[:, [1, 3]] = 1 - y[:, [3, 1]]
        return img, y

    def random_sized_crop(self, img, targets):
        img_w = img.shape[1]
        img_h = img.shape[0]
        img_area = img_w * img_h
        random_scale = np.random.random()
        random_scale *= (self.crop_area_range[1] -
                         self.crop_area_range[0])
        random_scale += self.crop_area_range[0]
        target_area = random_scale * img_area
        random_ratio = np.random.random()
        random_ratio *= (self.aspect_ratio_range[1] -
                         self.aspect_ratio_range[0])
        random_ratio += self.aspect_ratio_range[0]
        w = np.round(np.sqrt(target_area * random_ratio))     
        h = np.round(np.sqrt(target_area / random_ratio))
        if np.random.random() < 0.5:
            w, h = h, w
        w = min(w, img_w)
        w_rel = w / img_w
        w = int(w)
        h = min(h, img_h)
        h_rel = h / img_h
        h = int(h)
        x = np.random.random() * (img_w - w)
        x_rel = x / img_w
        x = int(x)
        y = np.random.random() * (img_h - h)
        y_rel = y / img_h
        y = int(y)
        img = img[y:y+h, x:x+w]
        new_targets = []
        for box in targets:
            cx = 0.5 * (box[0] + box[2])
            cy = 0.5 * (box[1] + box[3])
            if (x_rel < cx < x_rel + w_rel and
                y_rel < cy < y_rel + h_rel):
                xmin = (box[0] - x_rel) / w_rel
                ymin = (box[1] - y_rel) / h_rel
                xmax = (box[2] - x_rel) / w_rel
                ymax = (box[3] - y_rel) / h_rel
                xmin = max(0, xmin)
                ymin = max(0, ymin)
                xmax = min(1, xmax)
                ymax = min(1, ymax)
                box[:4] = [xmin, ymin, xmax, ymax]
                new_targets.append(box)
        new_targets = np.asarray(new_targets).reshape(-1, targets.shape[1])
        return img, new_targets

    def generate(self, train=True):
        while True:
            if train:
                shuffle(self.train_keys)
                keys = self.train_keys
            else:
                shuffle(self.val_keys)
                keys = self.val_keys
            inputs = []
            targets = []
            for key in keys:            
                img_path = self.path_prefix + key
                img = imread(img_path).astype('float32')
                y = self.gt[key].copy()
                if train and self.do_crop:
                    img, y = self.random_sized_crop(img, y)
                img = imresize(img, self.image_size).astype('float32')
                # boxの位置は正規化されているから画像をリサイズしても
                # 教師信号としては問題ない
                if train:
                    shuffle(self.color_jitter)
                    for jitter in self.color_jitter:
                        img = jitter(img)
                    if self.lighting_std:
                        img = self.lighting(img)
                    if self.hflip_prob > 0:
                        img, y = self.horizontal_flip(img, y)
                    if self.vflip_prob > 0:
                        img, y = self.vertical_flip(img, y)
                # 訓練データ生成時にbbox_utilを使っているのはここだけらしい
                #print(y)
                y = self.bbox_util.assign_boxes(y)
                #print(y)
                inputs.append(img)                
                targets.append(y)
                if len(targets) == self.batch_size:
                    tmp_inp = np.array(inputs)
                    tmp_targets = np.array(targets)
                    inputs = []
                    targets = []
                    yield preprocess_input(tmp_inp), tmp_targets

path_prefix = '../VOCdevkit/VOC2007/JPEGImages/'
#path_prefix = './VOCdevkit/JPEGImages/'
#path_prefix = './VOCdevkit/Original1/JPEGImages/'

gen = Generator(gt, bbox_util, 4, path_prefix,
                train_keys, val_keys,
                (input_shape[0], input_shape[1]), do_crop=False)

model = SSD(input_shape, num_classes=NUM_CLASSES)
#model.load_weights('weights_SSD300.hdf5', by_name=True)
#model.load_weights('./checkpoints/weights.03-0.21.hdf5', by_name=True)

freeze = ['input_1',
          #'conv1_1', 'conv1_2', 'pool1',
          #'conv2_1', 'conv2_2', 'pool2',
          #'conv3_1', 'conv3_2', 'conv3_3', 'pool3',
          'conv4_1', 'conv4_2', 'conv4_3', 'pool4',
          'conv5_1', 'conv5_2', 'conv5_3', 'pool5',
          'fc6','fc7',
          'conv6_1', 'conv6_2',
          'conv7_1','conv7_2',
          'conv8_1', 'conv8_2', 'pool6']

"""
for L in model.layers:
    if L.name in freeze:
        L.trainable = False
"""
 
        
def schedule(epoch, decay=0.99):   #0.9
    return base_lr * decay**(epoch)

csv_logger = keras.callbacks.CSVLogger('./checkpoints/training.log', separator=',', append=True)
weights_save=keras.callbacks.ModelCheckpoint('./checkpoints/weights.{val_loss:.2f}-{val_acc:.3f}.hdf5',
                                             verbose=1,
                                             save_weights_only=True)
learnRateSchedule=keras.callbacks.LearningRateScheduler(schedule)

callbacks = [ csv_logger, learnRateSchedule] #weights_save,

base_lr = 0.0001 #3e-4
#optim = keras.optimizers.Adam(lr=base_lr, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.)
optim = keras.optimizers.Adam(lr=base_lr)
model.compile(optimizer=optim,
              loss=MultiboxLoss(NUM_CLASSES, neg_pos_ratio=2.0).compute_loss, metrics=['accuracy'])
model.summary()

for j in range(10):
    nb_epoch = 3
    history = model.fit_generator(gen.generate(True), gen.train_batches,
                                  nb_epoch,
                                  callbacks=callbacks,
                                  workers=1, 
                                  validation_data=gen.generate(False),
                                  validation_steps=gen.val_batches,
                                  verbose=1)
    base_lr=base_lr/2
    model.save_weights('./checkpoints/params_maruF_epoch{0:0d}.hdf5'.format(j), True)
    # 学習履歴を保存
    save_history(history, os.path.join("./checkpoints/", 'history_maruF{0:0d}.txt'.format(j)),j)
    
inputs = []
images = []
img_path = path_prefix + sorted(val_keys)[0]
img = image.load_img(img_path, target_size=(img_rows,img_cols))
img = image.img_to_array(img)
images.append(imread(img_path))
inputs.append(img.copy())
inputs = preprocess_input(np.array(inputs))

preds = model.predict(inputs, batch_size=1, verbose=1)
results = bbox_util.detection_out(preds)

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
        # label_name = voc_classes[label - 1]
        display_txt = '{:0.2f}, {}'.format(score, label)
        coords = (xmin, ymin), xmax-xmin+1, ymax-ymin+1
        color = colors[label]
        currentAxis.add_patch(plt.Rectangle(*coords, fill=False, edgecolor=color, linewidth=2))
        currentAxis.text(xmin, ymin, display_txt, bbox={'facecolor':color, 'alpha':0.5})

    plt.show()

"""
4009/4009 [==============================] - 769s 192ms/step - loss: 2.8250 - acc: 0.3286 - val_loss: 2.1945 - val_acc: 0.3245

Epoch 00001: saving model to ./checkpoints/weights.01-2.19.hdf5
Epoch 2/10
4009/4009 [==============================] - 756s 189ms/step - loss: 1.9975 - acc: 0.3423 - val_loss: 1.8301 - val_acc: 0.3422

Epoch 00002: saving model to ./checkpoints/weights.02-1.83.hdf5
Epoch 3/10
4009/4009 [==============================] - 760s 190ms/step - loss: 1.5943 - acc: 0.3560 - val_loss: 1.6592 - val_acc: 0.3768

Epoch 00003: saving model to ./checkpoints/weights.03-1.66.hdf5
Epoch 4/10
4009/4009 [==============================] - 753s 188ms/step - loss: 1.3751 - acc: 0.3696 - val_loss: 1.6339 - val_acc: 0.3709

Epoch 00004: saving model to ./checkpoints/weights.04-1.63.hdf5
Epoch 5/10
4009/4009 [==============================] - 760s 190ms/step - loss: 1.2539 - acc: 0.3722 - val_loss: 1.6071 - val_acc: 0.3718

Epoch 00005: saving model to ./checkpoints/weights.05-1.61.hdf5
Epoch 6/10
4009/4009 [==============================] - 760s 190ms/step - loss: 1.1872 - acc: 0.3759 - val_loss: 1.6069 - val_acc: 0.3750

Epoch 00006: saving model to ./checkpoints/weights.06-1.61.hdf5
Epoch 7/10
4009/4009 [==============================] - 741s 185ms/step - loss: 1.1534 - acc: 0.3783 - val_loss: 1.6063 - val_acc: 0.3761

Epoch 00007: saving model to ./checkpoints/weights.07-1.61.hdf5
Epoch 8/10
4009/4009 [==============================] - 743s 185ms/step - loss: 1.1352 - acc: 0.3797 - val_loss: 1.6081 - val_acc: 0.3808

Epoch 00008: saving model to ./checkpoints/weights.08-1.61.hdf5
Epoch 9/10
4009/4009 [==============================] - 738s 184ms/step - loss: 1.1259 - acc: 0.3803 - val_loss: 1.6074 - val_acc: 0.3787

Epoch 00009: saving model to ./checkpoints/weights.09-1.61.hdf5
Epoch 10/10
4009/4009 [==============================] - 739s 184ms/step - loss: 1.1282 - acc: 0.3804 - val_loss: 1.5982 - val_acc: 0.3783

Epoch 00010: saving model to ./checkpoints/weights.10-1.60.hdf5
__________________________________________________________________________________________________
Layer (type)                    Output Shape         Param #     Connected to
==================================================================================================
input_1 (InputLayer)            (None, 300, 300, 3)  0
__________________________________________________________________________________________________
conv1_1 (Conv2D)                (None, 300, 300, 64) 1792        input_1[0][0]
__________________________________________________________________________________________________
conv1_2 (Conv2D)                (None, 300, 300, 64) 36928       conv1_1[0][0]
__________________________________________________________________________________________________
pool1 (MaxPooling2D)            (None, 150, 150, 64) 0           conv1_2[0][0]
__________________________________________________________________________________________________
conv2_1 (Conv2D)                (None, 150, 150, 128 73856       pool1[0][0]
__________________________________________________________________________________________________
conv2_2 (Conv2D)                (None, 150, 150, 128 147584      conv2_1[0][0]
__________________________________________________________________________________________________
pool2 (MaxPooling2D)            (None, 75, 75, 128)  0           conv2_2[0][0]
__________________________________________________________________________________________________
conv3_1 (Conv2D)                (None, 75, 75, 256)  295168      pool2[0][0]
__________________________________________________________________________________________________
conv3_2 (Conv2D)                (None, 75, 75, 256)  590080      conv3_1[0][0]
__________________________________________________________________________________________________
conv3_3 (Conv2D)                (None, 75, 75, 256)  590080      conv3_2[0][0]
__________________________________________________________________________________________________
pool3 (MaxPooling2D)            (None, 38, 38, 256)  0           conv3_3[0][0]
__________________________________________________________________________________________________
conv4_1 (Conv2D)                (None, 38, 38, 512)  1180160     pool3[0][0]
__________________________________________________________________________________________________
conv4_2 (Conv2D)                (None, 38, 38, 512)  2359808     conv4_1[0][0]
__________________________________________________________________________________________________
conv4_3 (Conv2D)                (None, 38, 38, 512)  2359808     conv4_2[0][0]
__________________________________________________________________________________________________
pool4 (MaxPooling2D)            (None, 19, 19, 512)  0           conv4_3[0][0]
__________________________________________________________________________________________________
conv5_1 (Conv2D)                (None, 19, 19, 512)  2359808     pool4[0][0]
__________________________________________________________________________________________________
conv5_2 (Conv2D)                (None, 19, 19, 512)  2359808     conv5_1[0][0]
__________________________________________________________________________________________________
conv5_3 (Conv2D)                (None, 19, 19, 512)  2359808     conv5_2[0][0]
__________________________________________________________________________________________________
pool5 (MaxPooling2D)            (None, 19, 19, 512)  0           conv5_3[0][0]
__________________________________________________________________________________________________
fc6 (Conv2D)                    (None, 19, 19, 1024) 4719616     pool5[0][0]
__________________________________________________________________________________________________
fc7 (Conv2D)                    (None, 19, 19, 1024) 1049600     fc6[0][0]
__________________________________________________________________________________________________
conv6_1 (Conv2D)                (None, 19, 19, 256)  262400      fc7[0][0]
__________________________________________________________________________________________________
conv6_2 (Conv2D)                (None, 10, 10, 512)  1180160     conv6_1[0][0]
__________________________________________________________________________________________________
conv7_1 (Conv2D)                (None, 10, 10, 128)  65664       conv6_2[0][0]
__________________________________________________________________________________________________
conv7_1z (ZeroPadding2D)        (None, 12, 12, 128)  0           conv7_1[0][0]
__________________________________________________________________________________________________
conv7_2 (Conv2D)                (None, 5, 5, 256)    295168      conv7_1z[0][0]
__________________________________________________________________________________________________
conv8_1 (Conv2D)                (None, 5, 5, 128)    32896       conv7_2[0][0]
__________________________________________________________________________________________________
conv4_3_norm (Normalize)        (None, 38, 38, 512)  512         conv4_3[0][0]
__________________________________________________________________________________________________
conv8_2 (Conv2D)                (None, 3, 3, 256)    295168      conv8_1[0][0]
__________________________________________________________________________________________________
pool6 (GlobalAveragePooling2D)  (None, 256)          0           conv8_2[0][0]
__________________________________________________________________________________________________
conv4_3_norm_mbox_conf (Conv2D) (None, 38, 38, 63)   290367      conv4_3_norm[0][0]
__________________________________________________________________________________________________
fc7_mbox_conf (Conv2D)          (None, 19, 19, 126)  1161342     fc7[0][0]
__________________________________________________________________________________________________
conv6_2_mbox_conf (Conv2D)      (None, 10, 10, 126)  580734      conv6_2[0][0]
__________________________________________________________________________________________________
conv7_2_mbox_conf (Conv2D)      (None, 5, 5, 126)    290430      conv7_2[0][0]
__________________________________________________________________________________________________
conv8_2_mbox_conf (Conv2D)      (None, 3, 3, 126)    290430      conv8_2[0][0]
__________________________________________________________________________________________________
conv4_3_norm_mbox_loc (Conv2D)  (None, 38, 38, 12)   55308       conv4_3_norm[0][0]
__________________________________________________________________________________________________
fc7_mbox_loc (Conv2D)           (None, 19, 19, 24)   221208      fc7[0][0]
__________________________________________________________________________________________________
conv6_2_mbox_loc (Conv2D)       (None, 10, 10, 24)   110616      conv6_2[0][0]
__________________________________________________________________________________________________
conv7_2_mbox_loc (Conv2D)       (None, 5, 5, 24)     55320       conv7_2[0][0]
__________________________________________________________________________________________________
conv8_2_mbox_loc (Conv2D)       (None, 3, 3, 24)     55320       conv8_2[0][0]
__________________________________________________________________________________________________
conv4_3_norm_mbox_conf_flat (Fl (None, 90972)        0           conv4_3_norm_mbox_conf[0][0]
__________________________________________________________________________________________________
fc7_mbox_conf_flat (Flatten)    (None, 45486)        0           fc7_mbox_conf[0][0]
__________________________________________________________________________________________________
conv6_2_mbox_conf_flat (Flatten (None, 12600)        0           conv6_2_mbox_conf[0][0]
__________________________________________________________________________________________________
conv7_2_mbox_conf_flat (Flatten (None, 3150)         0           conv7_2_mbox_conf[0][0]
__________________________________________________________________________________________________
conv8_2_mbox_conf_flat (Flatten (None, 1134)         0           conv8_2_mbox_conf[0][0]
__________________________________________________________________________________________________
pool6_mbox_conf_flat (Dense)    (None, 126)          32382       pool6[0][0]
__________________________________________________________________________________________________
conv4_3_norm_mbox_loc_flat (Fla (None, 17328)        0           conv4_3_norm_mbox_loc[0][0]
__________________________________________________________________________________________________
fc7_mbox_loc_flat (Flatten)     (None, 8664)         0           fc7_mbox_loc[0][0]
__________________________________________________________________________________________________
conv6_2_mbox_loc_flat (Flatten) (None, 2400)         0           conv6_2_mbox_loc[0][0]
__________________________________________________________________________________________________
conv7_2_mbox_loc_flat (Flatten) (None, 600)          0           conv7_2_mbox_loc[0][0]
__________________________________________________________________________________________________
conv8_2_mbox_loc_flat (Flatten) (None, 216)          0           conv8_2_mbox_loc[0][0]
__________________________________________________________________________________________________
pool6_mbox_loc_flat (Dense)     (None, 24)           6168        pool6[0][0]
__________________________________________________________________________________________________
mbox_conf (Concatenate)         (None, 153468)       0           conv4_3_norm_mbox_conf_flat[0][0]
                                                                 fc7_mbox_conf_flat[0][0]
                                                                 conv6_2_mbox_conf_flat[0][0]
                                                                 conv7_2_mbox_conf_flat[0][0]
                                                                 conv8_2_mbox_conf_flat[0][0]
                                                                 pool6_mbox_conf_flat[0][0]
__________________________________________________________________________________________________
pool6_reshaped (Reshape)        (None, 1, 1, 256)    0           pool6[0][0]
__________________________________________________________________________________________________
mbox_loc (Concatenate)          (None, 29232)        0           conv4_3_norm_mbox_loc_flat[0][0]
                                                                 fc7_mbox_loc_flat[0][0]
                                                                 conv6_2_mbox_loc_flat[0][0]
                                                                 conv7_2_mbox_loc_flat[0][0]
                                                                 conv8_2_mbox_loc_flat[0][0]
                                                                 pool6_mbox_loc_flat[0][0]
__________________________________________________________________________________________________
mbox_conf_logits (Reshape)      (None, 7308, 21)     0           mbox_conf[0][0]
__________________________________________________________________________________________________
conv4_3_norm_mbox_priorbox (Pri (None, 4332, 8)      0           conv4_3_norm[0][0]
__________________________________________________________________________________________________
fc7_mbox_priorbox (PriorBox)    (None, 2166, 8)      0           fc7[0][0]
__________________________________________________________________________________________________
conv6_2_mbox_priorbox (PriorBox (None, 600, 8)       0           conv6_2[0][0]
__________________________________________________________________________________________________
conv7_2_mbox_priorbox (PriorBox (None, 150, 8)       0           conv7_2[0][0]
__________________________________________________________________________________________________
conv8_2_mbox_priorbox (PriorBox (None, 54, 8)        0           conv8_2[0][0]
__________________________________________________________________________________________________
pool6_mbox_priorbox (PriorBox)  (None, 6, 8)         0           pool6_reshaped[0][0]
__________________________________________________________________________________________________
mbox_loc_final (Reshape)        (None, 7308, 4)      0           mbox_loc[0][0]
__________________________________________________________________________________________________
mbox_conf_final (Activation)    (None, 7308, 21)     0           mbox_conf_logits[0][0]
__________________________________________________________________________________________________
mbox_priorbox (Concatenate)     (None, 7308, 8)      0           conv4_3_norm_mbox_priorbox[0][0]
                                                                 fc7_mbox_priorbox[0][0]
                                                                 conv6_2_mbox_priorbox[0][0]
                                                                 conv7_2_mbox_priorbox[0][0]
                                                                 conv8_2_mbox_priorbox[0][0]
                                                                 pool6_mbox_priorbox[0][0]
__________________________________________________________________________________________________
predictions (Concatenate)       (None, 7308, 33)     0           mbox_loc_final[0][0]
                                                                 mbox_conf_final[0][0]
                                                                 mbox_priorbox[0][0]
==================================================================================================
Total params: 25,765,497
Trainable params: 11,050,809
Non-trainable params: 14,714,688
__________________________________________________________________________________________________

Layer (type)                    Output Shape         Param #     Connected to
==================================================================================================
input_1 (InputLayer)            (None, 300, 300, 3)  0
__________________________________________________________________________________________________
conv1_1 (Conv2D)                (None, 300, 300, 64) 1792        input_1[0][0]
__________________________________________________________________________________________________
conv1_2 (Conv2D)                (None, 300, 300, 64) 36928       conv1_1[0][0]
__________________________________________________________________________________________________
dropout_1 (Dropout)             (None, 300, 300, 64) 0           conv1_2[0][0]
__________________________________________________________________________________________________
pool1 (MaxPooling2D)            (None, 150, 150, 64) 0           dropout_1[0][0]
__________________________________________________________________________________________________
conv2_1 (Conv2D)                (None, 150, 150, 128 73856       pool1[0][0]
__________________________________________________________________________________________________
conv2_2 (Conv2D)                (None, 150, 150, 128 147584      conv2_1[0][0]
__________________________________________________________________________________________________
dropout_2 (Dropout)             (None, 150, 150, 128 0           conv2_2[0][0]
__________________________________________________________________________________________________
pool2 (MaxPooling2D)            (None, 75, 75, 128)  0           dropout_2[0][0]
__________________________________________________________________________________________________
conv3_1 (Conv2D)                (None, 75, 75, 256)  295168      pool2[0][0]
__________________________________________________________________________________________________
conv3_2 (Conv2D)                (None, 75, 75, 256)  590080      conv3_1[0][0]
__________________________________________________________________________________________________
conv3_3 (Conv2D)                (None, 75, 75, 256)  590080      conv3_2[0][0]
__________________________________________________________________________________________________
dropout_3 (Dropout)             (None, 75, 75, 256)  0           conv3_3[0][0]
__________________________________________________________________________________________________
pool3 (MaxPooling2D)            (None, 38, 38, 256)  0           dropout_3[0][0]
__________________________________________________________________________________________________
conv4_1 (Conv2D)                (None, 38, 38, 512)  1180160     pool3[0][0]
__________________________________________________________________________________________________
conv4_2 (Conv2D)                (None, 38, 38, 512)  2359808     conv4_1[0][0]
__________________________________________________________________________________________________
conv4_3 (Conv2D)                (None, 38, 38, 512)  2359808     conv4_2[0][0]
__________________________________________________________________________________________________
dropout_4 (Dropout)             (None, 38, 38, 512)  0           conv4_3[0][0]
__________________________________________________________________________________________________
pool4 (MaxPooling2D)            (None, 19, 19, 512)  0           dropout_4[0][0]
__________________________________________________________________________________________________
conv5_1 (Conv2D)                (None, 19, 19, 512)  2359808     pool4[0][0]
__________________________________________________________________________________________________
conv5_2 (Conv2D)                (None, 19, 19, 512)  2359808     conv5_1[0][0]
__________________________________________________________________________________________________
conv5_3 (Conv2D)                (None, 19, 19, 512)  2359808     conv5_2[0][0]
__________________________________________________________________________________________________
dropout_5 (Dropout)             (None, 19, 19, 512)  0           conv5_3[0][0]
__________________________________________________________________________________________________
pool5 (MaxPooling2D)            (None, 19, 19, 512)  0           dropout_5[0][0]
__________________________________________________________________________________________________
fc6 (Conv2D)                    (None, 19, 19, 1024) 4719616     pool5[0][0]
__________________________________________________________________________________________________
drop6 (Dropout)                 (None, 19, 19, 1024) 0           fc6[0][0]
__________________________________________________________________________________________________
fc7 (Conv2D)                    (None, 19, 19, 1024) 1049600     drop6[0][0]
__________________________________________________________________________________________________
drop7 (Dropout)                 (None, 19, 19, 1024) 0           fc7[0][0]
__________________________________________________________________________________________________
conv6_1 (Conv2D)                (None, 19, 19, 512)  524800      drop7[0][0]
__________________________________________________________________________________________________
conv6_2 (Conv2D)                (None, 10, 10, 1024) 4719616     conv6_1[0][0]
__________________________________________________________________________________________________
dropout_6 (Dropout)             (None, 10, 10, 1024) 0           conv6_2[0][0]
__________________________________________________________________________________________________
conv7_1 (Conv2D)                (None, 10, 10, 256)  262400      dropout_6[0][0]
__________________________________________________________________________________________________
conv7_1z (ZeroPadding2D)        (None, 12, 12, 256)  0           conv7_1[0][0]
__________________________________________________________________________________________________
conv7_2 (Conv2D)                (None, 5, 5, 256)    590080      conv7_1z[0][0]
__________________________________________________________________________________________________
dropout_7 (Dropout)             (None, 5, 5, 256)    0           conv7_2[0][0]
__________________________________________________________________________________________________
conv8_1 (Conv2D)                (None, 5, 5, 128)    32896       dropout_7[0][0]
__________________________________________________________________________________________________
conv8_2 (Conv2D)                (None, 3, 3, 256)    295168      conv8_1[0][0]
__________________________________________________________________________________________________
conv4_3_norm (Normalize)        (None, 38, 38, 512)  512         dropout_4[0][0]
__________________________________________________________________________________________________
dropout_8 (Dropout)             (None, 3, 3, 256)    0           conv8_2[0][0]
__________________________________________________________________________________________________
pool6 (GlobalAveragePooling2D)  (None, 256)          0           dropout_8[0][0]
__________________________________________________________________________________________________
conv4_3_norm_mbox_conf (Conv2D) (None, 38, 38, 63)   290367      conv4_3_norm[0][0]
__________________________________________________________________________________________________
fc7_mbox_conf (Conv2D)          (None, 19, 19, 126)  1161342     drop7[0][0]
__________________________________________________________________________________________________
conv6_2_mbox_conf (Conv2D)      (None, 10, 10, 126)  1161342     dropout_6[0][0]
__________________________________________________________________________________________________
conv7_2_mbox_conf (Conv2D)      (None, 5, 5, 126)    290430      dropout_7[0][0]
__________________________________________________________________________________________________
conv8_2_mbox_conf (Conv2D)      (None, 3, 3, 126)    290430      dropout_8[0][0]
__________________________________________________________________________________________________
conv4_3_norm_mbox_loc (Conv2D)  (None, 38, 38, 12)   55308       conv4_3_norm[0][0]
__________________________________________________________________________________________________
fc7_mbox_loc (Conv2D)           (None, 19, 19, 24)   221208      drop7[0][0]
__________________________________________________________________________________________________
conv6_2_mbox_loc (Conv2D)       (None, 10, 10, 24)   221208      dropout_6[0][0]
__________________________________________________________________________________________________
conv7_2_mbox_loc (Conv2D)       (None, 5, 5, 24)     55320       dropout_7[0][0]
__________________________________________________________________________________________________
conv8_2_mbox_loc (Conv2D)       (None, 3, 3, 24)     55320       dropout_8[0][0]
__________________________________________________________________________________________________
conv4_3_norm_mbox_conf_flat (Fl (None, 90972)        0           conv4_3_norm_mbox_conf[0][0]
__________________________________________________________________________________________________
fc7_mbox_conf_flat (Flatten)    (None, 45486)        0           fc7_mbox_conf[0][0]
__________________________________________________________________________________________________
conv6_2_mbox_conf_flat (Flatten (None, 12600)        0           conv6_2_mbox_conf[0][0]
__________________________________________________________________________________________________
conv7_2_mbox_conf_flat (Flatten (None, 3150)         0           conv7_2_mbox_conf[0][0]
__________________________________________________________________________________________________
conv8_2_mbox_conf_flat (Flatten (None, 1134)         0           conv8_2_mbox_conf[0][0]
__________________________________________________________________________________________________
pool6_mbox_conf_flat (Dense)    (None, 126)          32382       pool6[0][0]
__________________________________________________________________________________________________
conv4_3_norm_mbox_loc_flat (Fla (None, 17328)        0           conv4_3_norm_mbox_loc[0][0]
__________________________________________________________________________________________________
fc7_mbox_loc_flat (Flatten)     (None, 8664)         0           fc7_mbox_loc[0][0]
__________________________________________________________________________________________________
conv6_2_mbox_loc_flat (Flatten) (None, 2400)         0           conv6_2_mbox_loc[0][0]
__________________________________________________________________________________________________
conv7_2_mbox_loc_flat (Flatten) (None, 600)          0           conv7_2_mbox_loc[0][0]
__________________________________________________________________________________________________
conv8_2_mbox_loc_flat (Flatten) (None, 216)          0           conv8_2_mbox_loc[0][0]
__________________________________________________________________________________________________
pool6_mbox_loc_flat (Dense)     (None, 24)           6168        pool6[0][0]
__________________________________________________________________________________________________
mbox_conf (Concatenate)         (None, 153468)       0           conv4_3_norm_mbox_conf_flat[0][0]
                                                                 fc7_mbox_conf_flat[0][0]
                                                                 conv6_2_mbox_conf_flat[0][0]
                                                                 conv7_2_mbox_conf_flat[0][0]
                                                                 conv8_2_mbox_conf_flat[0][0]
                                                                 pool6_mbox_conf_flat[0][0]
__________________________________________________________________________________________________
pool6_reshaped (Reshape)        (None, 1, 1, 256)    0           pool6[0][0]
__________________________________________________________________________________________________
mbox_loc (Concatenate)          (None, 29232)        0           conv4_3_norm_mbox_loc_flat[0][0]
                                                                 fc7_mbox_loc_flat[0][0]
                                                                 conv6_2_mbox_loc_flat[0][0]
                                                                 conv7_2_mbox_loc_flat[0][0]
                                                                 conv8_2_mbox_loc_flat[0][0]
                                                                 pool6_mbox_loc_flat[0][0]
__________________________________________________________________________________________________
mbox_conf_logits (Reshape)      (None, 7308, 21)     0           mbox_conf[0][0]
__________________________________________________________________________________________________
conv4_3_norm_mbox_priorbox (Pri (None, 4332, 8)      0           conv4_3_norm[0][0]
__________________________________________________________________________________________________
fc7_mbox_priorbox (PriorBox)    (None, 2166, 8)      0           drop7[0][0]
__________________________________________________________________________________________________
conv6_2_mbox_priorbox (PriorBox (None, 600, 8)       0           dropout_6[0][0]
__________________________________________________________________________________________________
conv7_2_mbox_priorbox (PriorBox (None, 150, 8)       0           dropout_7[0][0]
__________________________________________________________________________________________________
conv8_2_mbox_priorbox (PriorBox (None, 54, 8)        0           dropout_8[0][0]
__________________________________________________________________________________________________
pool6_mbox_priorbox (PriorBox)  (None, 6, 8)         0           pool6_reshaped[0][0]
__________________________________________________________________________________________________
mbox_loc_final (Reshape)        (None, 7308, 4)      0           mbox_loc[0][0]
__________________________________________________________________________________________________
mbox_conf_final (Activation)    (None, 7308, 21)     0           mbox_conf_logits[0][0]
__________________________________________________________________________________________________
mbox_priorbox (Concatenate)     (None, 7308, 8)      0           conv4_3_norm_mbox_priorbox[0][0]
                                                                 fc7_mbox_priorbox[0][0]
                                                                 conv6_2_mbox_priorbox[0][0]
                                                                 conv7_2_mbox_priorbox[0][0]
                                                                 conv8_2_mbox_priorbox[0][0]
                                                                 pool6_mbox_priorbox[0][0]
__________________________________________________________________________________________________
predictions (Concatenate)       (None, 7308, 33)     0           mbox_loc_final[0][0]
                                                                 mbox_conf_final[0][0]
                                                                 mbox_priorbox[0][0]
==================================================================================================
Total params: 30,750,201
Trainable params: 23,114,937
Non-trainable params: 7,635,264
__________________________________________________________________________________________________
"""