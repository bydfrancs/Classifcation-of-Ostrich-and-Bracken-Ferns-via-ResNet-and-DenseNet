'''# python librairies installation
!pip install split-folders matplotlib opencv-python spicy'''

# display, transform, read, split ...
import numpy as np
import pandas as pd
import cv2 as cv
import os
import matplotlib.pyplot as plt

# tensorflow
import tensorflow.keras as keras
import tensorflow as tf

# model / neural network
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.applications.resnet50 import preprocess_input

#train data
from PIL import Image 
train=pd.read_csv("train.csv")
train_images=[]
path="train_images/"
for i in train.id:
    image=plt.imread(path+i+".jpg")
    train_images.append(image)
    
train_images=np.asarray(train_images)
X=train_images
y=train.is_ostrich

#Data split 70-20-10
from sklearn.model_selection import train_test_split
from keras import utils

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1)
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.22) # 0.22 x 0.9 = 0.198

Cat_test_y = utils.to_categorical(y_test)
Cat_val_y = utils.to_categorical(y_val)
y_train = utils.to_categorical(y_train)

# ResNet50 model
resnet_50 = ResNet50(include_top=False, weights='imagenet', input_shape=(400,600,3))
for layer in resnet_50.layers:
    layer.trainable = False

# build the entire model
x = resnet_50.output
x = layers.GlobalAveragePooling2D()(x)
x = layers.Dense(512, activation='relu')(x) 
x = layers.Dropout(0.5)(x)
x = layers.Dense(256, activation='relu')(x) 
x = layers.Dropout(0.5)(x)
x = layers.Dense(128, activation='relu')(x) 
x = layers.Dropout(0.5)(x)
x = layers.Dense(64, activation='relu')(x) 
x = layers.Dropout(0.5)(x)
predictions = layers.Dense(2, activation='softmax')(x)
model = Model(inputs = resnet_50.input, outputs = predictions)

# define training function
def trainModel(model, epochs, optimizer):
    batch_size = 10 #from 24
    model.compile(optimizer=optimizer, loss="binary_crossentropy", metrics=["accuracy"])
    return model.fit(X_train, y_train, validation_data=(X_val, Cat_val_y), epochs=epochs, batch_size=batch_size, shuffle=True)

# launch the training
model_history = trainModel(model = model, epochs = 20, optimizer = "Adam") #from 10 epoch

loss_train_curve = model_history.history["loss"]
loss_val_curve = model_history.history["val_loss"]
plt.plot(loss_train_curve, label = "Train")
plt.plot(loss_val_curve, label = "Validation")
plt.legend(loc = 'upper right')
plt.title("Loss")
plt.show()

acc_train_curve = model_history.history["accuracy"]
acc_val_curve = model_history.history["val_accuracy"]
plt.plot(acc_train_curve, label = "Train")
plt.plot(acc_val_curve, label = "Validation")
plt.legend(loc = 'lower right')
plt.title("Accuracy")
plt.show()

print("fit_generator")
model.save('model.keras')
model.save_weights('weights.keras')

print("saved model and weights")
model.summary()

#Testing
from sklearn import metrics
label_pred = model.predict(X_test)

pred = []
for i in range(len(label_pred)):
    pred.append(np.argmax(label_pred[i]))

Y_test = np.argmax(Cat_test_y, axis=1) # Convert one-hot to index

print(metrics.classification_report(Y_test, pred))
print(metrics.accuracy_score(Y_test, pred))

