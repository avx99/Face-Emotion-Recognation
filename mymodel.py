import os
import cv2
import numpy as np
from keras.models import Sequential
from keras.layers import Dense,Dropout,Activation,Flatten,BatchNormalization
from keras.layers import Conv2D,MaxPooling2D
from keras.optimizers import Adam

BATCH_SIZE = 30
EPOCHS = 20
IMAGE_SIZE = (48,48,3)
DATASET = r'dataset/'
TRAIN = r'dataset/train'
TEST = r'dataset/validation'
CLASSES = ['angry', 'happy', 'neutral', 'sad', 'surprise']
CLASSES_NUMBER = 5

X_test = []
X_train = []
y_test = []
y_train = []



for classe in os.listdir(TRAIN):
    for image in os.listdir(TRAIN + '/' + classe):
        img = cv2.imread(TRAIN + '/' + classe + '/' + image)       
        # img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        img = img / 255
        X_train.append(img)
        y_train.append(CLASSES.index(classe))

for classe in os.listdir(TEST):
    for image in os.listdir(TEST + '/' + classe):
        img = cv2.imread(TEST + '/' + classe + '/' + image)       
        # img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        img = img / 255
        X_test.append(img)
        y_test.append(CLASSES.index(classe))

X_train = np.array(X_train)
y_train = np.array(y_train)
X_test = np.array(X_test)
y_test = np.array(y_test)

X_train = X_train[:,:,:,np.newaxis]
X_test = X_test[:,:,:,np.newaxis]
y_train = y_train[:,np.newaxis]
y_test = y_test[:,np.newaxis]






model = Sequential()

# Block-1

model.add(Conv2D(32,(3,3),padding='same',kernel_initializer='he_normal',input_shape=IMAGE_SIZE))
model.add(Activation('elu'))
model.add(BatchNormalization())
model.add(Conv2D(32,(3,3),padding='same',kernel_initializer='he_normal',input_shape=IMAGE_SIZE))
model.add(Activation('elu'))
model.add(BatchNormalization())
model.add(MaxPooling2D())
model.add(Dropout(0.2))

# Block-2 

model.add(Conv2D(64,(3,3),padding='same',kernel_initializer='he_normal'))
model.add(Activation('elu'))
model.add(BatchNormalization())
model.add(Conv2D(64,(3,3),padding='same',kernel_initializer='he_normal'))
model.add(Activation('elu'))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.2))

# Block-3

# model.add(Conv2D(128,(3,3),padding='same',kernel_initializer='he_normal'))
# model.add(Activation('elu'))
# model.add(BatchNormalization())
# model.add(Conv2D(128,(3,3),padding='same',kernel_initializer='he_normal'))
# model.add(Activation('elu'))
# model.add(BatchNormalization())
# model.add(MaxPooling2D(pool_size=(2,2)))
# model.add(Dropout(0.2))

# # Block-4 

# model.add(Conv2D(256,(3,3),padding='same',kernel_initializer='he_normal'))
# model.add(Activation('elu'))
# model.add(BatchNormalization())
# model.add(Conv2D(256,(3,3),padding='same',kernel_initializer='he_normal'))
# model.add(Activation('elu'))
# model.add(BatchNormalization())
# model.add(MaxPooling2D(pool_size=(2,2)))
# model.add(Dropout(0.2))

# Block-5

model.add(Flatten())
model.add(Dense(512,kernel_initializer='he_normal'))
model.add(Activation('elu'))
model.add(BatchNormalization())
model.add(Dropout(0.5))

# Block-6

model.add(Dense(256,kernel_initializer='he_normal'))
model.add(Activation('elu'))
model.add(BatchNormalization())
model.add(Dropout(0.5))

# Block-7

model.add(Dense(CLASSES_NUMBER,kernel_initializer='he_normal'))
model.add(Activation('softmax'))

print(model.summary())


model.compile(loss='binary_crossentropy',optimizer = Adam(lr=0.001),metrics=['accuracy'])
model.fit(X_train,y_train ,batch_size = BATCH_SIZE , epochs = EPOCHS)

a = X_test[1500]
a = a[np.newaxis,:,:,:]
y_pred = model.predict(a)
true_val = y_test[1500]