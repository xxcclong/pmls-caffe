import sys
sys.path.insert(1,"/home/kezhao/usr/python3/lib/python3.6/site-packages")
import os
import random
dev =  str(random.randint(0,7))

os.environ["CUDA_VISIBLE_DEVICES"] = dev
print("device : " + dev)
import tensorflow as tf
import json
import copy
import time
import numpy as np
import pandas as pd
import keras
from keras.models import Sequential
from keras.utils import plot_model

from keras.layers import Dense, Dropout, Activation, Reshape, Conv2D, AveragePooling2D, Flatten, MaxPool2D
from keras.optimizers import adam, RMSprop
from keras.callbacks import ReduceLROnPlateau

class Logger(object):
    def __init__(self, filename='default_log'):
        self.terminal = sys.stdout
        self.log=open(filename,'a')
        self.rule = ['ETA','=>','...','===']
    def write(self, message):
        for item in self.rule:
            if message.count(item) > 0:
                return
        if message == '\n':
            return 
        self.terminal.write(message)
        self.log.write(message)
    def flush(self):
        pass


def Print(k):
    s = ''
    for i in range(28):
        for j in range(28):
            if k[i*28 + j] > 0:
                s+='*'
            else:
                s+=' '
        s+='\n'
    print(s)


def output_prediction(y_pred, model_name):
    global t
    data_predict = {"ImageId":range(1, test_sample_num+1), "Label":y_pred}
    data_predict = pd.DataFrame(data_predict)
    data_predict.to_csv("Result/result_%s_" %(model_name) + t+".csv", index = False)

def Compare(pre, real):
    global x_test, json_string, t
    ALL = 0
    RIGHT = 0
    for i in range(len(pre)):
        if pre[i] == real[i]:
            RIGHT += 1 
        else:
            Print(x_test[i])
        ALL += 1
        #else print number
    print(ALL)
    print(RIGHT)
    print('accuracy!! = ' + str( float(RIGHT) / float(ALL)))
    outf = open('MODEL/'+ str( float(RIGHT) / float(ALL))+ t,'a')
    outf.write(json_string)
    return  float(RIGHT) / float(ALL)

def Move(arr):
    left = np.zeros((33600, 784),dtype = np.int)   
    right = np.zeros((33600, 784),dtype = np.int)   
    up = np.zeros((33600, 784),dtype = np.int)   
    down = np.zeros((33600, 784),dtype = np.int)   
    for j in range(left.shape[0]):
        for i in range(left.shape[1]):
            if i % 28 == 27:
                continue
            left[j][i] = arr[j][i+1]
    print("finish left")
    for j in range(left.shape[0]):
        for i in range(left.shape[1]):
            if i % 28 == 0:
                continue
            right[j][i] = arr[j][i-1]
    for j in range(left.shape[0]):
        for i in range(left.shape[1]):
            if int(i / 28) == 0:
                continue
            up[j][i] = arr[j][i-28]
    for j in range(left.shape[0]):
        for i in range(left.shape[1]):
            if int(i / 28) == 27:
                continue
            down[j][i] = arr[j][i+28]
    return right, left, up, down
        

def Add_data(train_x, train_y):
    r,l,u,d = Move(train_x)
    train_x = np.concatenate((train_x,r))
    train_x = np.concatenate((train_x,l))
    train_x = np.concatenate((train_x,u))
    train_x = np.concatenate((train_x,d))
    temp = copy.deepcopy(train_y)
    for k in range(4):
        train_y = np.concatenate((train_y, temp))
    return train_x, train_y
model_name = "CNN"
t = str(time.asctime( time.localtime(time.time()) ))
# sys.stdout = Logger('LOG/'+ t + '.txt')


x_part_train = np.array(json.load(open("x_train.json")))
y_part_train = np.array(json.load(open("y_train.json")))
x_part_test = np.array(json.load(open("x_test.json")))
y_part_test = np.array(json.load(open("y_test.json")))
x_test = np.array(json.load(open("question.json")))

test_sample_num = x_test.shape[0]
x_part_train = x_part_train / 255.0
x_part_test = x_part_test / 255.0
x_test = x_test / 255.0
model = Sequential()
y_part_train = keras.utils.to_categorical(y_part_train, num_classes=10)
opt_y_part_test = keras.utils.to_categorical(y_part_test, num_classes=10)
model.add(Reshape(target_shape=(1, 28, 28), input_shape=(784,)))
'''
model.add(Conv2D(kernel_size=(5, 5), filters=32, padding="same", data_format="channels_first", kernel_initializer="uniform", use_bias=False))
model.add(AveragePooling2D(pool_size=(2, 2), data_format="channels_first"))
model.add(Conv2D(kernel_size=(2, 2), filters=32, padding="same", data_format="channels_first", kernel_initializer="uniform", use_bias=False))
model.add(AveragePooling2D(pool_size=(3, 3), data_format="channels_first"))
model.add(Flatten())
model.add(Dense(output_dim=1000, activation='relu'))
model.add(Dense(output_dim=100, activation='relu'))
model.add(Dense(output_dim=10, activation='softmax'))
adam = keras.optimizers.Adam(lr=0.0005, beta_1=0.9, beta_2=0.999, epsilon=1e-08) 
model.compile(loss='binary_crossentropy', optimizer=adam, metrics=['accuracy'])
'''
model.add(Conv2D(filters = 32, kernel_size = (5,5),padding = 'Same', 
                     activation ='relu', input_shape = (28,28,1)))

model.add(AveragePooling2D(pool_size=(2,2), data_format="channels_first") )
model.add(Conv2D(filters = 64, kernel_size = (3,3),padding = 'Same', 
                     activation ='relu'))
model.add(MaxPool2D(pool_size=(2,2), data_format="channels_first") )
model.add(Dropout(0.25))
#model.add(Conv2D(filters = 128, kernel_size = (3,3),padding = 'Same', 
#                     activation ='relu'))
#model.add(Conv2D(filters = 64, kernel_size = (3,3),padding = 'Same', 
#                     activation ='relu'))
#model.add(MaxPool2D(pool_size=(3,3), strides=(2,2), data_format="channels_first"))
# model.add(AveragePooling2D(pool_size=(2,2),strides=(2,2), data_format="channels_first") )
#model.add(Dropout(0.25))
model.add(Flatten())
model.add(Dense(1024, activation = "relu"))
model.add(Dropout(0.5))
model.add(Dense(100, activation = "relu"))
model.add(Dense(10, activation = "softmax"))
optimizer = RMSprop(lr=0.001, rho=0.9, epsilon=1e-08, decay=0.0)
model.compile(optimizer = optimizer , loss = "categorical_crossentropy", metrics=["accuracy"])
learning_rate_reduction = ReduceLROnPlateau(monitor='val_acc', patience=3,verbose=1, factor=0.5, min_lr=0.00001)
model.fit(x_part_train, y_part_train,validation_data = (x_part_test,opt_y_part_test), epochs=30, batch_size=64,verbose = 2, callbacks=[learning_rate_reduction])
#history = model.fit_generator(x_part_train, y_part_train, 
#epochs = epochs, validation_data = (x_part_test,y_part_test),
#verbose = 2)



json_string = model.to_json()  
y_pred = model.predict_classes(x_part_test)
fl = Compare(y_pred, y_part_test)
if fl > 0.9: 
    y_pred = model.predict_classes(x_test)
    output_prediction(y_pred, str(fl) + model_name)

