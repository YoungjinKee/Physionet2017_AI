from sklearn.metrics import confusion_matrix, accuracy_score
from keras.callbacks import ModelCheckpoint
from biosppy.signals import ecg
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import MinMaxScaler, RobustScaler
import scipy.io as sio
from os import listdir
from os.path import isfile, join
import keras
from keras.models import Sequential
from keras.layers import Dense,Add, UpSampling2D,ZeroPadding1D,Multiply,Activation,concatenate,Concatenate, Dropout, Conv2D, MaxPooling2D, Flatten, LSTM, BatchNormalization,Conv1D,AveragePooling1D, GlobalAveragePooling1D, MaxPooling1D, Input
from keras import regularizers
from keras import backend as K
import os
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import wfdb
import ast
from keras.models import Model
import tensorflow as tf
from keras.callbacks import EarlyStopping
import datetime as dt
from tensorflow.python.keras.callbacks import TensorBoard
from time import time

## Main parameter

sampling_rate=500
num_classes=4
channel=1

band_passfilter=True
normalization=True
dropout_rate=0.5
my_optimizer='Nadam'

experiment_setting=("sampling_rate =",sampling_rate,"num_classes =",num_classes, "channel =",channel,"Band_passfilter =",band_passfilter,"Normalization =",normalization,"Dropout_rate =",dropout_rate,"Optimizer =",my_optimizer
      )






#################################  Step 1 : Load data ##############################
################################ PTB-XL 2018 data set ##############################
path='c://Deeplearning/data/'

X_data=np.load('c://Deeplearning/data/physionet_2017_X.npy')
Y_data=np.load('c://Deeplearning/data/physionet_2017_Y.npy')

filtered_ecg_measurements=X_data.reshape(X_data.shape[0],X_data.shape[1],1)
new_labels=Y_data
#################################  Step 2 : Select class  ################################# 

#################################  Step 3 : Data denoising  & normalization (option) ################################# 
'''
from scipy.signal import butter, lfilter
from scipy import stats

def bandpass_filter(data, lowcut, highcut, signal_freq, filter_order):
        """
        Method responsible for creating and applying Butterworth filter.
        :param deque data: raw data
        :param float lowcut: filter lowcut frequency value
        :param float highcut: filter highcut frequency value
        :param int signal_freq: signal frequency in samples per second (Hz)
        :param int filter_order: filter order
        :return array: filtered data
        """
        nyquist_freq = 0.5 * signal_freq
        low = lowcut / nyquist_freq
        high = highcut / nyquist_freq
        b, a = butter(filter_order, [low, high], btype="band")
        y = lfilter(b, a, data)
        return y
    


filter_lowcut = 0.001
filter_highcut = 50
filter_order = 1

#filtered_ecg_measurements = bandpass_filter(New_X, lowcut=filter_lowcut, highcut=filter_highcut, signal_freq=sampling_rate, filter_order=filter_order)

def feature_normalize(dataset):

    mu = np.mean(dataset, axis=0)
    sigma = np.std(dataset, axis=0)
    return (dataset - mu)/sigma

filtered_ecg_measurements=feature_normalize(New_X)

#################################  Step 4 : Channel selection & shuffling data ################################# 
#channel=3 # channel 6 # channel 12
if channel==1:
    filtered_ecg_measurements=filtered_ecg_measurements[:,:,0]
    filtered_ecg_measurements=filtered_ecg_measurements.reshape(filtered_ecg_measurements.shape[0],filtered_ecg_measurements.shape[1],1)
else:
    filtered_ecg_measurements=filtered_ecg_measurements[:,:,0:channel]

def data_shuffling(input_data,y_label):
    
    k=np.random.randint(0, len(input_data),len(input_data))

    shuffled_data=input_data[k]
    shuffled_label=y_label[k]

    return shuffled_data, shuffled_label

filtered_ecg_measurements,new_labels=data_shuffling(filtered_ecg_measurements,new_labels)
print(len(filtered_ecg_measurements))

'''
#################################  Step 5 : Data split ################################# 
def train_test_split(input_dat,Y_label,portion):
    
    X_train=input_dat[0:int(portion*input_dat.shape[0]),:]
    y_train=Y_label[0:int(portion*input_dat.shape[0]),:]
    
    X_test=input_dat[int(portion*input_dat.shape[0]):,:]
    print(int(portion*input_dat.shape[0]))
    y_test=Y_label[int(portion*input_dat.shape[0]):,:]
    
    return X_train, y_train, X_test, y_test

X_train, y_train, X_test, y_test=train_test_split(filtered_ecg_measurements,new_labels,0.9)

#################################  Step 6 : Deep learning Model ##################################### 
## Modified Loss function
## For sovling data imbalacing.
# 1) focal_loss, 2) weight balancing 

def categorical_focal_loss(gamma=2.0, alpha=0.25):
    """
    Implementation of Focal Loss from the paper in multiclass classification
    Formula:
        loss = -alpha*((1-p)^gamma)*log(p)
    Parameters:
        alpha -- the same as wighting factor in balanced cross entropy
        gamma -- focusing parameter for modulating factor (1-p)
    Default value:
        gamma -- 2.0 as mentioned in the paper
        alpha -- 0.25 as mentioned in the paper
    """
    def focal_loss(y_true, y_pred):
        # Define epsilon so that the backpropagation will not result in NaN
        # for 0 divisor case
        epsilon = K.epsilon()
        # Add the epsilon to prediction value
        #y_pred = y_pred + epsilon
        # Clip the prediction value
        y_pred = K.clip(y_pred, epsilon, 1.0-epsilon)
        # Calculate cross entropy
        cross_entropy = -y_true*K.log(y_pred)
        # Calculate weight that consists of  modulating factor and weighting factor
        weight = alpha * y_true * K.pow((1-y_pred), gamma)
        # Calculate focal loss
        loss = weight * cross_entropy
        # Sum the losses in mini_batch
        loss = K.sum(loss, axis=1)
        return loss
    
    return focal_loss




'''
def focal_loss(y_true, y_pred):
    gamma = 2.0
    alpha = 0.25
    pt_1 = tf.where(tf.equal(y_true, 1), y_pred, tf.ones_like(y_pred))
    pt_0 = tf.where(tf.equal(y_true, 0), y_pred, tf.zeros_like(y_pred))
    return -K.sum(alpha * K.pow(1. - pt_1, gamma) * K.log(pt_1))-K.sum((1-alpha) * K.pow( pt_0, gamma) * K.log(1. - pt_0))
'''
'''
class_weight = {"buy": 0.75,
                "don't buy": 0.25}

model.fit(X_train, Y_train, epochs=10, batch_size=32, class_weight=class_weight)
'''
#######################################  Model 1: Resnet 50  ########################################
## Dropout 정확하게 할것 0.5 / 0.2 / 0.8




def initial_layer(x):
    #x = ZeroPadding1D(padding=3)(x)
    x = Conv1D(filters=128, kernel_size=5) (x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    #x = ZeroPadding1D(padding=1)(x)
    
    return x

def Short_cutLayer(x,filter_value, kernel_value, iteration):
    
    x = MaxPooling1D(3)(x)
    shortcut =x 
    
    for i in range(iteration):
        if i ==0:
            x = Conv1D(filters=filter_value, kernel_size=kernel_value,padding='same') (x)
            x = BatchNormalization()(x)
            x = Activation('relu')(x)
            
            x = Conv1D(filters=filter_value, kernel_size=kernel_value*3,padding='same') (x)
            x = BatchNormalization()(x)
            x = Activation('relu')(x)
            
            x = Conv1D(filters=filter_value*4, kernel_size=kernel_value,padding='same') (x)
            shortcut = Conv1D(filters=filter_value*4,padding='same', kernel_size=kernel_value) (shortcut)
            x = BatchNormalization()(x)
            shortcut = BatchNormalization()(shortcut)
            
            x = Add()([x, shortcut])
            x = Activation('relu')(x)      
            shortcut = x
            
        else:
            x = Conv1D(filters=filter_value, kernel_size=kernel_value,padding='same') (x)
            x = BatchNormalization()(x)
            x = Activation('relu')(x)
            
            x = Conv1D(filters=filter_value, kernel_size=kernel_value*3,padding='same') (x)
            x = BatchNormalization()(x)
            x = Activation('relu')(x)
            
            x = Conv1D(filters=filter_value*4, kernel_size=kernel_value,padding='same') (x)
            x = BatchNormalization()(x)
            
            x = Add()([x, shortcut])
            x = Activation('relu')(x)      
            shortcut = x
   
    return x

input_shape=(filtered_ecg_measurements.shape[1],filtered_ecg_measurements.shape[2])
inputs=Input(shape=input_shape)

layer1= initial_layer(inputs)
layer2=Short_cutLayer(layer1,filter_value=64, kernel_value=1, iteration=3)
#layer2 = Dropout(0.2)(layer2)

layer3=Short_cutLayer(layer2,filter_value=128, kernel_value=1, iteration=4)
layer4=Short_cutLayer(layer3,filter_value=256, kernel_value=1, iteration=6)
layer5=Short_cutLayer(layer4,filter_value=512, kernel_value=1, iteration=3)
x=GlobalAveragePooling1D()(layer5)
x = Dropout(dropout_rate)(x)

output = Dense(num_classes, activation='softmax')(x)


model = Model(inputs=inputs, outputs=output)
#model.compile(loss=categorical_focal_loss(gamma=2.0, alpha=0.25), optimizer='adam', metrics=['accuracy']) # -> focal_loss version
#model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
#model.load_weights('C:/Deeplearning/ckpt/physionet2017_2020-08-14 09_channel_1_class_4DR_0.5_Nadam/epoch_50-val_acc_0.8547-val_loss_0.4433.h5')

model.compile(loss='categorical_crossentropy', optimizer=my_optimizer, metrics=['accuracy'])

model.summary()

#################################  Step 7 : Deep learning training ################################# 
## checkpoint path, batch size. Callbacks setting  
# to-do list -> early stop. 
date_time_obj=dt.datetime.now()


model_save_folder='c://Deeplearning/ckpt/physionet2017_'+str(date_time_obj)[:-13]+'_channel_'+str(channel)+'_class_'+str(num_classes)+'DR_0.5'+'_Nadam/'
if not os.path.exists(model_save_folder):
    os.mkdir(model_save_folder)
else:
    model_save_folder=model_save_folder+'v2/'
    
    os.mkdir(model_save_folder)

text_file = open(model_save_folder+'Experiment_setting.txt', "w")
text_file.write(str(experiment_setting))
text_file.close()


model_path=model_save_folder+'epoch_{epoch:02d}-val_acc_{val_acc:.4f}-val_loss_{val_loss:.4f}.h5'
es = EarlyStopping(monitor='val_loss',patience=20)
checkpointer = ModelCheckpoint(filepath=model_path, monitor='val_acc', verbose=1, save_best_only=True)
hist = model.fit(X_train, y_train, validation_data=(X_test, y_test), batch_size=20, epochs=100, verbose=1, shuffle=False, callbacks=[checkpointer,es])



## For tensorboard monitoring
#tensorboard=TensorBoard(log_dir=model_save_folder+"{}".format(time()))
'''
checkpointer = ModelCheckpoint(filepath=model_path, monitor='val_acc', verbose=1, save_best_only=True)
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    hist = model.fit(X_train, y_train, validation_data=(X_test, y_test), batch_size=50, epochs=100, verbose=1, shuffle=True, callbacks=[checkpointer,es,tensorboard])
'''
##

#checkpointer = ModelCheckpoint(filepath=model_path, monitor='val_loss', verbose=1, save_best_only=True)
#hist = model.fit(X_train, y_train, validation_data=(X_test, y_test), batch_size=50, epochs=100, verbose=1, shuffle=True, callbacks=[checkpointer,es])

np.save(model_save_folder+'hist.npy',hist)
#################################  Step 8 : Deep learning Evaluation ################################# 
## TO do list : Evaluation package ->  ROC, AUC ...  
predictions=model.predict(X_test)

from sklearn.metrics import classification_report

def change(x):  #From boolean arrays to decimal arrays
    answer = np.zeros((np.shape(x)[0]))
    for i in range(np.shape(x)[0]):
        max_value = max(x[i, :])
        max_index = list(x[i, :]).index(max_value)
        answer[i] = max_index
    return answer.astype(np.int)

text_file = open(model_save_folder+"report.txt", "w")
text_file.write(classification_report(change(y_test),change(predictions), target_names=['Normal', 'AF', 'Other','Noise']))
text_file.close()

text_file = open(model_save_folder+"confusion_matrix.txt", "w")
text_file.write(confusion_matrix(change(y_test), change(predictions)))
text_file.close()
