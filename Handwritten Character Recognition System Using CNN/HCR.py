import tensorflow as tf
from tensorflow import keras
import numpy as np
import pandas as pd
print(tf.__version__)

mnist = keras.datasets.mnist
(train_images_mnist,train_labels_mnist),(test_images_mnist,test_labels_mnist) = mnist.load_data()
train_images_mnist = np.reshape(train_images_mnist,(train_images_mnist.shape[0],28,28,1))  
test_images_mnist = np.reshape(test_images_mnist,(test_images_mnist.shape[0],28,28,1))

AZ_data = pd.read_csv("A_Z Handwritten Data.csv",header = None)
AZ_labels = AZ_data.values[:,0]
AZ_images = AZ_data.values[:,1:]
AZ_images = np.reshape(AZ_images,(AZ_images.shape[0],28,28,1))

from sklearn.model_selection import train_test_split

test_size = float(len(test_labels_mnist))/len(train_labels_mnist)
print(f'test set size: {test_size}')
train_images_AZ, test_images_AZ, train_labels_AZ, test_labels_AZ = train_test_split(AZ_images,AZ_labels, test_size=test_size)
train_labels_mnist = train_labels_mnist + max(AZ_labels)+1
test_labels_mnist = test_labels_mnist + max(AZ_labels)+1

train_images = np.concatenate((train_images_AZ,train_images_mnist),axis=0)
train_labels = np.concatenate((train_labels_AZ,train_labels_mnist))
test_images = np.concatenate((test_images_AZ,test_images_mnist),axis=0)
test_labels = np.concatenate((test_labels_AZ,test_labels_mnist))

print('Data ready')

import tensorflow as tf
from tensorflow.keras.optimizers import RMSprop
model = tf.keras.models.Sequential([
    tf.keras.layers.Conv2D(32, (3,3), activation='relu', input_shape=(28, 28, 1)),
    tf.keras.layers.MaxPooling2D(2,2),
    tf.keras.layers.Conv2D(32, (3,3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2,2), 
    tf.keras.layers.Flatten(), 
    tf.keras.layers.Dense(512, activation='relu'), 
    tf.keras.layers.Dense(len(np.unique(train_labels)), activation='softmax')  
])

model.compile(optimizer=RMSprop(learning_rate=1e-4),
              loss='sparse_categorical_crossentropy',
              metrics = ['accuracy'])

model.summary()

from tensorflow.keras.preprocessing.image import ImageDataGenerator
train_datagen = ImageDataGenerator(
      rescale=1./255,
      rotation_range=15,
      width_shift_range=0.1,
      height_shift_range=0.1,
      shear_range=0.1,
      zoom_range=0.2,
      horizontal_flip=False,
      fill_mode='nearest')

test_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow(train_images, train_labels, batch_size=50, shuffle=True)
validation_generator = test_datagen.flow(test_images, test_labels, batch_size=50, shuffle=True)

history = model.fit(
      train_generator,
      steps_per_epoch=500,  
      epochs=30,
      validation_data=validation_generator,
      validation_steps=50,  
      verbose=2)
model.save('model_v2')

from tensorflow.keras.models import load_model
import tensorflow as tf
import cv2 
import matplotlib.pyplot as plt

import imutils
from imutils.contours import sort_contours

model_path = 'model_v2'
print("Loading NN model...")
model = load_model(model_path)
print("Done")

image_path = "Letters.jpg"
image = cv2.imread(image_path)

gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
cropped = gray[120:,:]
blurred = cv2.GaussianBlur(cropped, (5, 5), 0)

from matplotlib import cm
fig = plt.figure(figsize=(16,4))
ax = plt.subplot(1,4,1)
ax.imshow(image)
ax.set_title('original image');

ax = plt.subplot(1,4,2)
ax.imshow(gray,cmap=cm.binary_r)
ax.set_axis_off()
ax.set_title('grayscale image');

ax = plt.subplot(1,4,3)
ax.imshow(cropped,cmap=cm.binary_r)
ax.set_axis_off()
ax.set_title('cropped image');

ax = plt.subplot(1,4,4)
ax.imshow(blurred,cmap=cm.binary_r)
ax.set_axis_off()
ax.set_title('blurred image');

edged = cv2.Canny(blurred, 30, 250) 
cnts = cv2.findContours(edged.copy(), cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
cnts = imutils.grab_contours(cnts)
cnts = sort_contours(cnts, method="left-to-right")[0]

figure = plt.figure(figsize=(7,7))
plt.imshow(edged,cmap=cm.binary_r);

chars = []

for c in cnts:
  (x, y, w, h) = cv2.boundingRect(c)
  roi = cropped[y:y + h, x:x + w]
  
  thresh = cv2.threshold(roi, 0, 255,cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]
  
  (tH, tW) = thresh.shape
  if tW > tH:
    thresh = imutils.resize(thresh, width=28)
  else:
    thresh = imutils.resize(thresh, height=28)

  (tH, tW) = thresh.shape
  dX = int(max(0, 28 - tW) / 2.0)
  dY = int(max(0, 28 - tH) / 2.0)
  padded = cv2.copyMakeBorder(thresh, top=dY, bottom=dY,
    left=dX, right=dX, borderType=cv2.BORDER_CONSTANT,
    value=(0, 0, 0))
  padded = cv2.resize(padded, (28, 28))
  padded = padded.astype("float32") / 255.0
  padded = np.expand_dims(padded, axis=-1)
  chars.append((padded, (x, y, w, h)))

n_cols = 10
n_rows = np.floor((len(chars)/ n_cols)+1). astype(int)
fig = plt.figure(figsize=(2*n_cols,2*n_rows))
for i,char in enumerate(chars):
  ax = plt.subplot(n_rows,n_cols,i+1)
  ax.imshow(char[0][:,:],cmap=cm.binary,aspect='auto')
plt.tight_layout()

boxes = [b[1] for b in chars]
chars = np.array([c[0] for c in chars], dtype="float32")
preds = model.predict(chars)
labelNames = "ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"

image = cv2.imread("Letters.jpg")
cropped = image[200:,:]

for (pred, (x, y, w, h)) in zip(preds, boxes):
  i = np.argmax(pred)
  prob = pred[i]
  label = labelNames[i]
  label_text = f"{label},{prob * 100:.1f}%"
  cv2.rectangle(blurred, (x, y), (x + w, y + h), (0,255,0), 2)
  cv2.putText(blurred, label_text, (x - 10, y - 10),cv2.FONT_HERSHEY_SIMPLEX,1, (0,255, 0), 2,)
plt.figure(figsize=(15,10))
plt.imshow(blurred)