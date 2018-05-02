import keras
import pickle
from videotest_alt import VideoTest
import time

import sys
sys.path.append("..")
from ssd_v2 import SSD300v2 as SSD

input_shape = (500,500,3)

# Change this if you run with other classes than VOC
class_names = ["background", "aeroplane", "bicycle", "bird", "boat", "bottle", "bus", "car", "cat", "chair", "cow", "diningtable", "dog", "horse", "motorbike", "person", "pottedplant", "sheep", "sofa", "train", "tvmonitor"];
NUM_CLASSES = len(class_names)

model = SSD(input_shape, num_classes=NUM_CLASSES)

# Change this path if you want to use your own trained weights
model.load_weights('weights_SSD300.hdf5') 
        
vid_test = VideoTest(class_names, model, input_shape)

# To test on webcam 0, remove the parameter (or change it to another number
# to test on that webcam)
#vid_test.run(0)

for i in range(1):
    print(i)
    #vid_test.run('1.png')    #doga_nogisaka.mp4')
    #time.sleep(1.5)
    #vid_test.run(0)
    #vid_test.run('2.jpg')
    #time.sleep(15)
    #vid_test.run('3.jpg')
    #time.sleep(1.5)
    #vid_test.run('douga_car2.mp4')
    #vid_test.run('doga_scuba.mp4')
    vid_test.run('dougasozai_car.mp4')
    #vid_test.run('doga_nogisaka.mp4') 
