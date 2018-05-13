from __future__ import print_function
import keras
from keras.datasets import cifar10
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import BatchNormalization
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Input
from keras.models import Model
import h5py

batch_size = 32
num_classes = 10
epochs = 50
data_augmentation = True

# The data, shuffled and split between train and test sets:
(x_train, y_train), (x_test, y_test) = cifar10.load_data()

#x_train=x_train[0:1000]
#y_train=y_train[0:1000]
#x_test=x_test[0:1000]
#y_test=y_test[0:1000]

print('x_train shape:', x_train.shape)
print(x_train.shape[0], 'train samples')
print(x_test.shape[0], 'test samples')

    

# Convert class vectors to binary class matrices.
y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)

def model_family_cnn(input_shape, num_classes=10):
    input_layer = Input(shape=input_shape)

    # Block 1
    conv1_1 = Conv2D(64, (3, 3),name='conv1_1', activation='relu', padding='same')(input_layer)
    conv1_2 = Conv2D(64, (3, 3),name='conv1_2', activation='relu', padding='same')(conv1_1)
    bn1 = BatchNormalization(axis=3)(conv1_2)
    pool1 = MaxPooling2D(pool_size=(2, 2))(bn1)
    drop1 = Dropout(0.5)(pool1)

    # Block 2
    conv2_1 = Conv2D(128, (3, 3),name='conv2_1', activation='relu', padding='same')(drop1)
    conv2_2 = Conv2D(128, (3, 3),name='conv2_2', activation='relu', padding='same')(conv2_1)
    bn2 = BatchNormalization(axis=3)(conv2_2)
    pool2 = MaxPooling2D(pool_size=(2, 2))(bn2)
    drop2 = Dropout(0.5)(pool2)

    # Block 3
    conv3_1 = Conv2D(256, (3, 3),name='conv3_1', activation='relu', padding='same')(drop2)
    conv3_2 = Conv2D(256, (3, 3),name='conv3_2', activation='relu', padding='same')(conv3_1)
    conv3_3 = Conv2D(256, (3, 3),name='conv3_3', activation='relu', padding='same')(conv3_2)
    conv3_4 = Conv2D(256, (3, 3),name='conv3_4', activation='relu', padding='same')(conv3_3)
    bn3 = BatchNormalization(axis=3)(conv3_4)
    pool3 = MaxPooling2D(pool_size=(2, 2))(bn3)
    drop3 = Dropout(0.5)(pool3)

    # Block 4
    conv4_1 = Conv2D(512, (3, 3),name='conv4_1', activation='relu', padding='same')(drop3)
    conv4_2 = Conv2D(512, (3, 3),name='conv4_2', activation='relu', padding='same')(conv4_1)
    conv4_3 = Conv2D(512, (3, 3),name='conv4_3', activation='relu', padding='same')(conv4_2)
    conv4_4 = Conv2D(512, (3, 3),name='conv4_4', activation='relu', padding='same')(conv4_3)
    bn4 = BatchNormalization(axis=3)(conv4_4)
    pool4 = MaxPooling2D(pool_size=(2, 2))(bn4)
    drop4 = Dropout(0.5)(pool4)

    # Block 5
    conv5_1 = Conv2D(512, (3, 3),name='conv5_1', activation='relu', padding='same')(drop4)
    conv5_2 = Conv2D(512, (3, 3),name='conv5_2', activation='relu', padding='same')(conv5_1)
    conv5_3 = Conv2D(512, (3, 3),name='conv5_3', activation='relu', padding='same')(conv5_2)
    conv5_4 = Conv2D(512, (3, 3),name='conv5_4', activation='relu', padding='same')(conv5_3)
    bn5 = BatchNormalization(axis=3)(conv5_4)
    pool5 = MaxPooling2D(pool_size=(2, 2))(bn5)
    drop5 = Dropout(0.5)(pool5)
    
    x = Flatten()(drop5)
    x = Dense(4096)(x)
    x = Activation('relu')(x)
    x = Dropout(0.5)(x)
    x = Dense(num_classes)(x)
    x = Activation('softmax')(x)
    model = Model(inputs=input_layer, outputs=x)
    return model

img_rows,img_cols=32, 32   #300, 300
input_shape = (img_rows,img_cols,3)   #224, 224, 3)

model = model_family_cnn(input_shape, num_classes = 10)

# initiate RMSprop optimizer
#opt = keras.optimizers.rmsprop(lr=0.0001, decay=1e-6)
base_lr = 0.0001 #3e-4
#optim = keras.optimizers.Adam(lr=base_lr, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.)
opt = keras.optimizers.Adam(lr=base_lr)

# load the weights from the last epoch
model.load_weights('weights_SSD300.hdf5', by_name=True)
"""
weights_path = 'weights_SSD300.hdf5'
f = h5py.File(weights_path,mode="r")

layer_names = [n.decode('utf8') for n in f.attrs['layer_names']]

for k in range(0,17):
    print("L",k," ",layer_names[k],model.layers[k-8])
    

for k in range(0,17):  #len(layer_names)):
#k=12
    print(k,layer_names[k])
    g = f[layer_names[k]]
    print(g)
    weight_names = [n.decode('utf8') for n in g.attrs['weight_names']]
    weight_values = [g[weight_name] for weight_name in weight_names]
    print(weight_names)
    print(model.layers[k-0])
    model.layers[k-0].set_weights(weight_values)
f.close()

for layer in model.layers[0:12]:
    layer.trainable = False    
"""
print('Model loaded.')

def schedule(epoch, decay=0.9):   #0.9
    return base_lr * decay**(epoch)
csv_logger = keras.callbacks.CSVLogger('./checkpoints/training.log', separator=',', append=True)
weights_save=keras.callbacks.ModelCheckpoint('./checkpoints/weights.{val_loss:.2f}-{val_acc:.3f}.hdf5',
                                             verbose=1,
                                             save_weights_only=True)
learnRateSchedule=keras.callbacks.LearningRateScheduler(schedule)
callbacks = [ csv_logger, learnRateSchedule] #weights_save,

# Let's train the model using RMSprop
model.compile(loss='categorical_crossentropy',
              optimizer=opt,
              metrics=['accuracy'])

model.summary()

x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
x_train /= 255
x_test /= 255

if not data_augmentation:
    print('Not using data augmentation.')
    model.fit(x_train, y_train,
              batch_size=batch_size,
              epochs=epochs,
              validation_data=(x_test, y_test),
              shuffle=True)
else:
    print('Using real-time data augmentation.')
    # This will do preprocessing and realtime data augmentation:
    datagen = ImageDataGenerator(
        featurewise_center=False,  # set input mean to 0 over the dataset
        samplewise_center=False,  # set each sample mean to 0
        featurewise_std_normalization=False,  # divide inputs by std of the dataset
        samplewise_std_normalization=False,  # divide each input by its std
        zca_whitening=False,  # apply ZCA whitening
        rotation_range=0,  # randomly rotate images in the range (degrees, 0 to 180)
        width_shift_range=0.1,  # randomly shift images horizontally (fraction of total width)
        height_shift_range=0.1,  # randomly shift images vertically (fraction of total height)
        horizontal_flip=True,  # randomly flip images
        vertical_flip=False)  # randomly flip images

    # Compute quantities required for feature-wise normalization
    # (std, mean, and principal components if ZCA whitening is applied).
    datagen.fit(x_train)

    # Fit the model on the batches generated by datagen.flow().
    model.fit_generator(datagen.flow(x_train, y_train,
                                     batch_size=batch_size),
                        steps_per_epoch=x_train.shape[0] // batch_size,
                        epochs=epochs,callbacks=callbacks,
                        validation_data=(x_test, y_test))



# save weights every epoch
model.save_weights('params_cifar10model_epoch.hdf5')
model.save_weights(
      'params_cifar10model_epoch_{0:03d}.hdf5'.format(epochs), True)

score = model.evaluate(x_test, y_test, verbose=0)
print('Test loss:', score[0])
print('Test accuracy:', score[1])

"""
Epoch 50/50
1562/1562 [==============================] - 80s 51ms/step - loss: 0.5912 - acc: 0.7949 - val_loss: 0.7396 - val_acc: 0.7544
Test loss: 0.739627401638031
Test accuracy: 0.7544
Layer (type)                 Output Shape              Param #
=================================================================
input_1 (InputLayer)         (None, 32, 32, 3)         0
_________________________________________________________________
conv1_1 (Conv2D)             (None, 32, 32, 64)        1792
_________________________________________________________________
batch_normalization_1 (Batch (None, 32, 32, 64)        256
_________________________________________________________________
max_pooling2d_1 (MaxPooling2 (None, 16, 16, 64)        0
_________________________________________________________________
dropout_1 (Dropout)          (None, 16, 16, 64)        0
_________________________________________________________________
conv2_1 (Conv2D)             (None, 16, 16, 128)       73856
_________________________________________________________________
batch_normalization_2 (Batch (None, 16, 16, 128)       512
_________________________________________________________________
max_pooling2d_2 (MaxPooling2 (None, 8, 8, 128)         0
_________________________________________________________________
dropout_2 (Dropout)          (None, 8, 8, 128)         0
_________________________________________________________________
conv3_1 (Conv2D)             (None, 8, 8, 256)         295168
_________________________________________________________________
conv3_2 (Conv2D)             (None, 8, 8, 256)         590080
_________________________________________________________________
batch_normalization_3 (Batch (None, 8, 8, 256)         1024
_________________________________________________________________
max_pooling2d_3 (MaxPooling2 (None, 4, 4, 256)         0
_________________________________________________________________
dropout_3 (Dropout)          (None, 4, 4, 256)         0
_________________________________________________________________
conv4_1 (Conv2D)             (None, 4, 4, 512)         1180160
_________________________________________________________________
conv4_2 (Conv2D)             (None, 4, 4, 512)         2359808
_________________________________________________________________
batch_normalization_4 (Batch (None, 4, 4, 512)         2048
_________________________________________________________________
max_pooling2d_4 (MaxPooling2 (None, 2, 2, 512)         0
_________________________________________________________________
dropout_4 (Dropout)          (None, 2, 2, 512)         0
_________________________________________________________________
conv5_1 (Conv2D)             (None, 2, 2, 512)         2359808
_________________________________________________________________
conv5_2 (Conv2D)             (None, 2, 2, 512)         2359808
_________________________________________________________________
batch_normalization_5 (Batch (None, 2, 2, 512)         2048
_________________________________________________________________
max_pooling2d_5 (MaxPooling2 (None, 1, 1, 512)         0
_________________________________________________________________
dropout_5 (Dropout)          (None, 1, 1, 512)         0
_________________________________________________________________
flatten_1 (Flatten)          (None, 512)               0
_________________________________________________________________
dense_1 (Dense)              (None, 4096)              2101248
_________________________________________________________________
activation_1 (Activation)    (None, 4096)              0
_________________________________________________________________
dropout_6 (Dropout)          (None, 4096)              0
_________________________________________________________________
dense_2 (Dense)              (None, 10)                40970
_________________________________________________________________
activation_2 (Activation)    (None, 10)                0
=================================================================
Total params: 11,368,586
Trainable params: 11,365,642
Non-trainable params: 2,944
_________________________________________________________________

Epoch 50/50
1562/1562 [==============================] - 112s 72ms/step - loss: 0.2400 - acc: 0.9172 - val_loss: 0.3077 - val_acc: 0.8949
Test loss: 0.3077232243150473
Test accuracy: 0.8949
Layer (type)                 Output Shape              Param #
=================================================================
input_1 (InputLayer)         (None, 32, 32, 3)         0
_________________________________________________________________
conv1_1 (Conv2D)             (None, 32, 32, 64)        1792
_________________________________________________________________
conv1_2 (Conv2D)             (None, 32, 32, 64)        36928
_________________________________________________________________
batch_normalization_1 (Batch (None, 32, 32, 64)        256
_________________________________________________________________
max_pooling2d_1 (MaxPooling2 (None, 16, 16, 64)        0
_________________________________________________________________
dropout_1 (Dropout)          (None, 16, 16, 64)        0
_________________________________________________________________
conv2_1 (Conv2D)             (None, 16, 16, 128)       73856
_________________________________________________________________
conv2_2 (Conv2D)             (None, 16, 16, 128)       147584
_________________________________________________________________
batch_normalization_2 (Batch (None, 16, 16, 128)       512
_________________________________________________________________
max_pooling2d_2 (MaxPooling2 (None, 8, 8, 128)         0
_________________________________________________________________
dropout_2 (Dropout)          (None, 8, 8, 128)         0
_________________________________________________________________
conv3_1 (Conv2D)             (None, 8, 8, 256)         295168
_________________________________________________________________
conv3_2 (Conv2D)             (None, 8, 8, 256)         590080
_________________________________________________________________
conv3_3 (Conv2D)             (None, 8, 8, 256)         590080
_________________________________________________________________
batch_normalization_3 (Batch (None, 8, 8, 256)         1024
_________________________________________________________________
max_pooling2d_3 (MaxPooling2 (None, 4, 4, 256)         0
_________________________________________________________________
dropout_3 (Dropout)          (None, 4, 4, 256)         0
_________________________________________________________________
conv4_1 (Conv2D)             (None, 4, 4, 512)         1180160
_________________________________________________________________
conv4_2 (Conv2D)             (None, 4, 4, 512)         2359808
_________________________________________________________________
conv4_3 (Conv2D)             (None, 4, 4, 512)         2359808
_________________________________________________________________
batch_normalization_4 (Batch (None, 4, 4, 512)         2048
_________________________________________________________________
max_pooling2d_4 (MaxPooling2 (None, 2, 2, 512)         0
_________________________________________________________________
dropout_4 (Dropout)          (None, 2, 2, 512)         0
_________________________________________________________________
conv5_1 (Conv2D)             (None, 2, 2, 512)         2359808
_________________________________________________________________
conv5_2 (Conv2D)             (None, 2, 2, 512)         2359808
_________________________________________________________________
conv5_3 (Conv2D)             (None, 2, 2, 512)         2359808
_________________________________________________________________
batch_normalization_5 (Batch (None, 2, 2, 512)         2048
_________________________________________________________________
max_pooling2d_5 (MaxPooling2 (None, 1, 1, 512)         0
_________________________________________________________________
dropout_5 (Dropout)          (None, 1, 1, 512)         0
_________________________________________________________________
flatten_1 (Flatten)          (None, 512)               0
_________________________________________________________________
dense_1 (Dense)              (None, 4096)              2101248
_________________________________________________________________
activation_1 (Activation)    (None, 4096)              0
_________________________________________________________________
dropout_6 (Dropout)          (None, 4096)              0
_________________________________________________________________
dense_2 (Dense)              (None, 10)                40970
_________________________________________________________________
activation_2 (Activation)    (None, 10)                0
=================================================================
Total params: 16,862,794
Trainable params: 16,859,850
Non-trainable params: 2,944
_________________________________________________________________
Using real-time data augmentation.
"""
