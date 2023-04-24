import numpy as np
import keras
from tensorflow.keras import datasets, layers, models, optimizers, initializers, callbacks
from sklearn.model_selection import train_test_split
from keras.regularizers import l2
import tensorflow as tf

import os
import matplotlib.pyplot as plt
import pathlib
from skimage import io
from skimage.transform import resize
from tensorflow.keras import regularizers
import torch
import cv2

threshold = 1000
path_str = "C:/Users/mzarodn2/Documents/DeepLearningCellClassif/testing_codes/train_segments/"
images_lst = [file for file in os.listdir(path_str) if file.endswith("tif")]

bf_imgs = []
for filenames in images_lst:
  im = io.imread(os.path.join(path_str,filenames))
  bf_imgs.append(im)
input_size = bf_imgs[0].shape[0]

print(bf_imgs[0][10:35, 10:35])

labels = np.loadtxt('labels.txt',delimiter = ',')
labels = labels[:len(bf_imgs)]


bin_label = np.where(labels > threshold, 1, 0)

def init_model(input_size, threshold):
    if threshold == -1:
        loss_type = 'poisson'
        metric_type = 'mean_squared_error'
        fin_activation = 'relu'
    else:
        loss_type = 'binary_crossentropy'
        metric_type = 'accuracy'
        fin_activation = 'sigmoid'

    initializer = initializers.RandomNormal(mean=0., stddev=1.)

    model = models.Sequential()
    model.add(layers.InputLayer(input_shape=(input_size, input_size, 1)))
    model.add(layers.Conv2D(64, kernel_size = (3,3), activation='relu'))
    model.add(layers.MaxPooling2D(pool_size = 2))
    model.add(layers.Conv2D(64, kernel_size = (3,3), activation='relu'))
    model.add(layers.MaxPooling2D(pool_size = 2))
    model.add(layers.Conv2D(128, kernel_size = (5,5), activation='relu'))
    model.add(layers.GlobalMaxPooling2D())
    model.add(layers.Dense(256, kernel_initializer=initializer, activation = 'relu'))
    # model.add(layers.Dropout(0.2))
    # model.add(layers.Dense(128, kernel_initializer=initializer, activation = 'relu'))
    model.add(layers.Dense(64, kernel_initializer=initializer, activation = 'relu'))
    # model.add(layers.Dense(32, kernel_initializer=initializer, activation = 'relu'))
    model.add(layers.Dense(1, activation = fin_activation))
    opt = optimizers.Adam(learning_rate=0.00001)
    model.compile(optimizer = opt, loss = loss_type, metrics = [metric_type])

    return model
  
  
  
def init_AlexNet(input_size, threshold):
    if threshold == -1:
        loss_type = 'poisson'
        metric_type = 'mean_squared_error'
        fin_activation = 'relu'
    else:
        loss_type = 'sparse_categorical_crossentropy'
        metric_type = 'accuracy'
        fin_activation = 'softmax'

    initializer = initializers.RandomNormal(mean=0., stddev=1.)
    
    data_augmentation = tf.keras.Sequential([
      layers.RandomFlip("horizontal_and_vertical")
    ])

    model = keras.models.Sequential([
    data_augmentation,
    keras.layers.Conv2D(filters=96, kernel_size=(11,11), strides=(4,4), activation='relu', input_shape=(input_size, input_size, 1)),
    keras.layers.BatchNormalization(),
    keras.layers.MaxPool2D(pool_size=(3,3), strides=(2,2)),
    keras.layers.Conv2D(filters=256, kernel_size=(5,5), strides=(1,1), activation='relu', padding="same"),
    keras.layers.BatchNormalization(),
    keras.layers.MaxPool2D(pool_size=(3,3), strides=(2,2)),
    keras.layers.Conv2D(filters=384, kernel_size=(3,3), strides=(1,1), activation='relu', padding="same"),
    keras.layers.BatchNormalization(),
    keras.layers.Conv2D(filters=384, kernel_size=(3,3), strides=(1,1), activation='relu', padding="same"),
    keras.layers.BatchNormalization(),
    keras.layers.Conv2D(filters=256, kernel_size=(3,3), strides=(1,1), activation='relu', padding="same"),
    keras.layers.BatchNormalization(),
    keras.layers.MaxPool2D(pool_size=(3,3), strides=(2,2)),
    keras.layers.Flatten(),
    keras.layers.Dense(4096, activation='relu'),
    keras.layers.Dropout(0.5),
    keras.layers.Dense(4096, activation='relu'),
    keras.layers.Dropout(0.5),
    keras.layers.Dense(2, activation=fin_activation)])
    
    opt = optimizers.Adam(learning_rate=0.00001)
    model.compile(optimizer = opt, loss = loss_type, metrics = [metric_type])

    return model
  

def fit_model(model, data_x, data_y, input_size, modelName):
    x,y = data_org(data_x, data_y, input_size)
    best_callback = [callbacks.ModelCheckpoint(filepath= modelName + '.h5', monitor='val_accuracy', save_best_only = True, mode='max')]

    history = model.fit(x, y, batch_size = 32, epochs = 100, validation_split=0.2, shuffle=True, callbacks = best_callback)
    print('Model has been saved in the supplied directory as ' + modelName + '.h5')
    return(history)

def data_org(data_x, data_y, size_selection):
    data_x = np.stack(data_x,axis=0)
    x = data_x.reshape(-1, size_selection, size_selection, 1)
    return x,data_y
  
def plot_train_curves(history):
  accuracy = history.history["accuracy"]
  val_accuracy = history.history["val_accuracy"]
  loss = history.history["loss"]
  val_loss = history.history["val_loss"]
  epochs = range(1, len(accuracy) + 1)
  
  plt.plot(epochs, accuracy, "b--", label="Training accuracy")
  plt.plot(epochs, val_accuracy, "b", label="Validation accuracy")
  plt.title("Training and Validation Accuracy")
  plt.legend()
  plt.figure()
  
  plt.plot(epochs, loss, "b--", label = "Training loss")
  plt.plot(epochs,val_loss, "b", label = "Validation loss")
  plt.title("Training and Validation Loss")
  plt.legend()
  plt.show()
  
  
model = init_model(input_size, 1000)
history = fit_model(model, bf_imgs, bin_label, input_size, 'AlexNet_median_filter_augment')
plot_train_curves(history)



