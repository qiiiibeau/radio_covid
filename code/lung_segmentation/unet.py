import os

import numpy as np
from timeit import default_timer as timer
import matplotlib.pyplot as plt
from skimage.io import imread, imsave, imshow
from skimage.transform import resize
from sklearn.model_selection import train_test_split

import tensorflow as tf
from tensorflow.keras.layers import *
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import Callback, EarlyStopping, ReduceLROnPlateau, ModelCheckpoint

from utils import DIR_DATA_LOCAL, DIR_IMAGES, DIR_MASKS, LST_GROUP, LST_FOLDERS, NUM_ALL_IMG, get_fname, LABEL_MAPPER
from preprocessing import crop_black_border

DIR_OUTPUT = 'result/mask_predicted'


def pipeline_unet(groups=list, num_images=list, image_size=(50, 50), use_mask=True,
                  crop=True, output_type='matrix', **kwargs):
    """
    Parameters
    ------------
    groups: list of categories (labels) of input data
    num_images: list of number of images in each group. pass NUM_ALL_IMG to use all the images in current dataset.
    image_size: tuple: the destinated size of each image. If none, the image will be 256*256.
    use_mask: bool: whether apply the masks to the radio images.
    crop: bool: whether crop the black border after applying mask.
    output_type: 'matrix' or 'vector', the output type of an image

    Return:
    ------------
    X_train
    X_test
    y_train
    y_test
    todo

    """

    X = np.zeros((sum(num_images), 256, 256))
    y = np.zeros(sum(num_images))
    sum_num = 0

    # load the images
    print('image loading ...')
    for group, num in zip(groups, num_images):
        X[sum_num:(sum_num + num), :, :] = [
            resize(imread(os.path.join(DIR_IMAGES[group], get_fname(group, idx)), as_gray=True), (256, 256),
                   anti_aliasing=True)
            for idx in range(1, num + 1)]
        y[sum_num:sum_num + num] = [LABEL_MAPPER[group]] * num
        sum_num += num

    # masking
    if use_mask:
        print('unet mask calculating ...')
        # load unet model
        model = tf.keras.models.load_model("projet_radio_covid/callbacks_unet_256_5200/unet_256_5200_20ep_batch_32.h5")
        # reshape X for prediction
        X_1 = X.reshape(len(X), 256, 256, 1)
        masks = model.predict(X_1)
        masks = tf.argmax(masks, axis=-1)
        masks = masks[..., tf.newaxis]
        masks = tf.cast(masks, tf.float32).numpy()
        X = X_1 * masks
        X = X.reshape(-1, 256, 256)

    # crop the black border and resize
    if crop:
        print('border cropping ...')
        X = np.array([resize(crop_black_border(image), image_size, anti_aliasing=True) for image in X])
    else:
        print('resizing ...')
        X = resize(X, (X.shape[0], *image_size), anti_aliasing=True)

    # data split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=23, shuffle=True, stratify=y)

    if output_type == 'matrix':
        return X_train, X_test, y_train, y_test

    elif output_type == 'vector':
        # reshape each image to 1-D vector
        X_train, X_test = X_train.reshape(X_train.shape[0], -1), X_test.reshape(X_test.shape[0], -1)

        plt.subplot(221)
        imshow(X_test[1].reshape(image_size))
        plt.subplot(222)
        plt.plot(np.histogram(X_test[1])[0])


x_img = 256
y_img = 256
image_size = (x_img, y_img)

# frac=10
n_covid = 1300  # 3616//frac
n_normal = 1300  # 10192//frac
n_viral = 1300  # 1345//frac
n_opac = 1300  # 6012//frac

groups = LST_GROUP
num_images = [n_covid, n_normal, n_viral, n_opac]

# load images
X = np.zeros((sum(num_images), 256, 256))
y = np.zeros(sum(num_images))
sum_num = 0

print('image loading ...')
for group, num in zip(groups, num_images):
    X[sum_num:(sum_num + num), :, :] = [
        resize(imread(os.path.join(DIR_IMAGES[group], get_fname(group, idx)), as_gray=True), (256, 256),
               anti_aliasing=True)
        for idx in range(1, num + 1)]
    y[sum_num:sum_num + num] = [LABEL_MAPPER[group]] * num
    sum_num += num
# img = tf.io.decode_jpeg(img, channels=3)

# load masks
print('mask loading ...')
masks = np.zeros((sum(num_images), 256, 256))
sum_num = 0
for group, num in zip(groups, num_images):
    masks[sum_num:(sum_num + num), :, :] = [imread(os.path.join(DIR_MASKS[group], get_fname(group, idx)), as_gray=True)
                                            for idx in range(1, num + 1)]
    sum_num += num

# resize images and masks
# not to be run in case x_img = 256
print('resizing ...')
X_rs = resize(X, (X.shape[0], *image_size), anti_aliasing=True)
masks_rs = resize(masks, (masks.shape[0], *image_size), anti_aliasing=True)

# in case x_img = y_img = 256
X_rs = X
masks_rs = masks

# images and masks split (test_size = 0.1 then validation_split = 0.2)
X_train, X_test, masks_train, masks_test = train_test_split(X_rs, masks_rs, test_size=0.1, random_state=23,
                                                            shuffle=True, stratify=y)

# reshape 1 channel
X_train_1 = X_train.reshape(-1, x_img, y_img, 1)
X_test_1 = X_test.reshape(-1, x_img, y_img, 1)

# reshape 1 channel
masks_train_1 = masks_train.reshape(-1, x_img, y_img, 1)
masks_test_1 = masks_test.reshape(-1, x_img, y_img, 1)

# check shape
print("check shape", X_test_1.shape, masks_test_1.shape)

#### ** U-net model 256 x 256 input with 12 levels**

# Unet model 256


nb_class = 2

inputs = Input((256, 256, 1))  # gray images

conv1 = Conv2D(16, 3, activation='relu', padding='same', kernel_initializer='he_normal')(inputs)
conv1 = Conv2D(16, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv1)
pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)

conv2 = Conv2D(32, 3, activation='relu', padding='same', kernel_initializer='he_normal')(pool1)
conv2 = Conv2D(32, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv2)
pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)

conv3 = Conv2D(64, 3, activation='relu', padding='same', kernel_initializer='he_normal')(pool2)
conv3 = Conv2D(64, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv3)
pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)

conv4 = Conv2D(128, 3, activation='relu', padding='same', kernel_initializer='he_normal')(pool3)
conv4 = Conv2D(128, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv4)
pool4 = MaxPooling2D(pool_size=(2, 2))(conv4)

conv5 = Conv2D(256, 3, activation='relu', padding='same', kernel_initializer='he_normal')(pool4)
conv5 = Conv2D(256, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv5)
drop5 = Dropout(0.5)(conv5)
pool5 = MaxPooling2D(pool_size=(2, 2))(drop5)

conv6 = Conv2D(512, 3, activation='relu', padding='same', kernel_initializer='he_normal')(pool5)
conv6 = Conv2D(512, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv6)
drop6 = Dropout(0.5)(conv6)

up7 = Conv2D(256, 2, activation='relu', padding='same', kernel_initializer='he_normal')(
    UpSampling2D(size=(2, 2))(drop6))
merge7 = concatenate([drop5, up7], axis=3)
conv7 = Conv2D(256, 3, activation='relu', padding='same', kernel_initializer='he_normal')(merge7)
conv7 = Conv2D(256, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv7)

up8 = Conv2D(128, 2, activation='relu', padding='same', kernel_initializer='he_normal')(
    UpSampling2D(size=(2, 2))(conv7))
merge8 = concatenate([conv4, up8], axis=3)
conv8 = Conv2D(128, 3, activation='relu', padding='same', kernel_initializer='he_normal')(merge8)
conv8 = Conv2D(128, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv8)

up9 = Conv2D(64, 2, activation='relu', padding='same', kernel_initializer='he_normal')(UpSampling2D(size=(2, 2))(conv8))
merge9 = concatenate([conv3, up9], axis=3)
conv9 = Conv2D(64, 3, activation='relu', padding='same', kernel_initializer='he_normal')(merge9)
conv9 = Conv2D(64, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv9)

up10 = Conv2D(32, 2, activation='relu', padding='same', kernel_initializer='he_normal')(
    UpSampling2D(size=(2, 2))(conv9))
merge10 = concatenate([conv2, up10], axis=3)
conv10 = Conv2D(32, 3, activation='relu', padding='same', kernel_initializer='he_normal')(merge10)
conv10 = Conv2D(32, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv10)

up11 = Conv2D(16, 2, activation='relu', padding='same', kernel_initializer='he_normal')(
    UpSampling2D(size=(2, 2))(conv10))
merge11 = concatenate([conv1, up11], axis=3)
conv11 = Conv2D(16, 3, activation='relu', padding='same', kernel_initializer='he_normal')(merge11)
conv11 = Conv2D(16, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv11)

conv11 = Conv2D(2, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv11)
conv12 = Conv2D(nb_class, 1, activation='softmax')(conv11)

model = Model(inputs=inputs, outputs=conv12)

model.summary()


# **callbacks**

class TimingCallback(Callback):
    def __init__(self, logs={}):
        self.logs = []

    def on_epoch_begin(self, epoch, logs={}):
        self.starttime = timer()

    def on_epoch_end(self, epoch, logs={}):
        self.logs.append(timer() - self.starttime)


model_checkpoint_unet = ModelCheckpoint(filepath='projet_radio_covid/callbacks_unet_256_5200_batch_32',
                                        monitor='val_loss',
                                        save_best_only=True,
                                        mode='min')

early_stopping = EarlyStopping(monitor='val_loss',
                               min_delta=0.005,
                               patience=5,
                               verbose=1)

reduce_learning_rate = ReduceLROnPlateau(monitor='val_loss',
                                         min_delta=0.005,
                                         patience=3,
                                         factor=0.5,
                                         cooldown=4,
                                         verbose=1)

time_callback = TimingCallback()

# compile and fit
model.compile(optimizer='adam',
              loss="sparse_categorical_crossentropy",
              # nos masques avec seulement 0 et 1, sparse_categorical_crossentropy
              metrics=['SparseCategoricalAccuracy'])

batch_size = 32
history_unet = model.fit(X_train_1, masks_train_1,
                         epochs=20,
                         #                        steps_per_epoch = Xm_train.shape[0]//batch_size,
                         batch_size=batch_size,
                         validation_split=0.2,
                         callbacks=[model_checkpoint_unet,
                                    early_stopping,
                                    reduce_learning_rate, time_callback],
                         verbose=True)

# model.save('projet_radio_covid/callbacks_unet_256_5200/unet_256_5200_20ep_batch_32.h5')  # path for collab
model.save('../result/models/callbacks_unet_256_5200/unet_256_5200_20ep_batch_32.h5')  # path for collab

train_loss = history_unet.history['loss']
val_loss = history_unet.history['val_loss']

train_acc = history_unet.history['sparse_categorical_accuracy']
val_acc = history_unet.history['val_sparse_categorical_accuracy']

plt.figure(figsize=(9, 3))

plt.subplot(1, 2, 1)
plt.plot(train_loss, label='train loss')
plt.plot(val_loss, label='test loss')
plt.xlabel('epoch')
plt.ylabel('loss function')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(train_acc, label='train accuracy')
plt.plot(val_acc, label='test accuracy')
plt.xlabel('epoch')
plt.ylabel('accuracy')
plt.legend()
plt.show()

# probabilités
pred_mask = model.predict(X_test_1)

score = model.evaluate(X_test_1, masks_test_1)
print('Loss function value', score[0], '\n Model accuracy :', score[1])

# from probablities to classification 0 and 1
pred_mask_class = tf.argmax(pred_mask, axis=-1)
pred_mask_class = pred_mask_class[..., tf.newaxis]
pred_mask_class = tf.cast(pred_mask_class, tf.float32)

for i in range(len(X_test)):
    imsave(os.path.join(DIR_OUTPUT, f'mask_{i}.png'), pred_mask_class[i])
