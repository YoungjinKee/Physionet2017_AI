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
'''
sampling_rate
num_classes
num_channel
band_passfilter
normalization
dropout

임진영
optimizer
loss
'''


#################################  Step 1 : Load data ##############################
################################ PTB-XL 2018 data set ##############################


## v1
def load_raw_data(df, sampling_rate, path):
    if sampling_rate == 100:
        data = [wfdb.rdsamp(path+f) for f in df.filename_lr]
    else:
        data = [wfdb.rdsamp(path+f) for f in df.filename_hr]
    data = np.array([signal for signal, meta in data])
    return data

path = 'C://Users/yjkee/Downloads/ptb-xl-a-large-publicly-available-electrocardiography-dataset-1.0.1/ptb-xl/'
sampling_rate=500

# load and convert annotation data
Y = pd.read_csv(path+'ptbxl_database.csv', index_col='ecg_id')
Y.scp_codes = Y.scp_codes.apply(lambda x: ast.literal_eval(x))

# Load raw signal data
X = load_raw_data(Y, sampling_rate, path)

# Load scp_statements.csv for diagnostic aggregation
agg_df = pd.read_csv(path+'scp_statements.csv', index_col=0)
agg_df = agg_df[agg_df.diagnostic == 1]

def aggregate_diagnostic(y_dic):
    tmp = []
    for key in y_dic.keys():
        if key in agg_df.index:
            tmp.append(agg_df.loc[key].diagnostic_class)
    return list(set(tmp))

# Apply diagnostic superclass
Y['diagnostic_superclass'] = Y.scp_codes.apply(aggregate_diagnostic)

kk=Y.diagnostic_superclass


#################################  Step 2 : Select class  ################################# 
num_classes=5
labels=list()
No_labels=list()
for i in range(1,len(kk)+1):
    label_act=np.zeros(num_classes)
      
    if kk[i]==['NORM']:
        label_act=[1,0,0,0,0]
    elif kk[i]==['CD']:
        label_act=[0,1,0,0,0]

    elif kk[i]==['STTC']:
        label_act=[0,0,1,0,0]

    elif kk[i]==['MI']:
        label_act=[0,0,0,1,0]
        
    elif kk[i]==['HYP']:
        label_act=[0,0,0,0,1]
    else:
        No_labels.append(i-1)
   
    labels.append(label_act)

    
labels = np.array(labels)
Nolabels=np.array(No_labels)

new_labels=np.delete(labels,Nolabels,axis=0)
New_X=np.delete(X,Nolabels,axis=0)
'''
from utils import utils

datafolder = 'C://Users/yjkee/Downloads/ptb-xl-a-large-publicly-available-electrocardiography-dataset-1.0.1/ptb-xl-a-large-publicly-available-electrocardiography-dataset-1.0.1/'
sampling_rate=500
task='superdiagnostic'


# Load PTB-XL data
data, raw_labels = utils.load_dataset(datafolder, sampling_frequency)
# Preprocess label data
labels = utils.compute_label_aggregations(raw_labels, datafolder, task)
# Select relevant data and convert to one-hot
data, labels, Y, _ = utils.select_data(data, labels, task, min_samples=0, outputfolder=outputfolder)

# 1-9 for training 
X_train = data[labels.strat_fold < 10]
y_train = Y[labels.strat_fold < 10]
# 10 for validation
X_val = data[labels.strat_fold == 10]
y_val = Y[labels.strat_fold == 10]
'''
#################################  Step 3 : Data denoising  & normalization (option) ################################# 

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
    

signal_frequency=500
filter_lowcut = 0.001
filter_highcut = 50
filter_order = 1

#filtered_ecg_measurements = bandpass_filter(New_X, lowcut=filter_lowcut, highcut=filter_highcut, signal_freq=signal_frequency, filter_order=filter_order)

def feature_normalize(dataset):

    mu = np.mean(dataset, axis=0)
    sigma = np.std(dataset, axis=0)
    return (dataset - mu)/sigma

filtered_ecg_measurements=feature_normalize(New_X)

#################################  Step 4 : Channel selection & shuffling data ################################# 
channel=12 # channel 6 # channel 12
if channel==3:
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
x = Dropout(0.5)(x)

output = Dense(num_classes, activation='softmax')(x)


model = Model(inputs=inputs, outputs=output)
#model.compile(loss=categorical_focal_loss(gamma=2.0, alpha=0.25), optimizer='adam', metrics=['accuracy']) # -> focal_loss version
#model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
model.compile(loss='categorical_crossentropy', optimizer='Nadam', metrics=['accuracy'])

model.summary()

#################################  Step 7 : Deep learning training ################################# 
## checkpoint path, batch size. Callbacks setting  
# to-do list -> early stop. 
date_time_obj=dt.datetime.now()


model_save_folder='c://Deeplearning/ckpt/ptb-xl_'+str(date_time_obj)[:-13]+'_channel_'+str(channel)+'_class_'+str(num_classes)+'DR_0.5'+'_Nadam/'
if not os.path.exists(model_save_folder):
    os.mkdir(model_save_folder)
else:
    model_save_folder=model_save_folder+'v2/'
    
    os.mkdir(model_save_folder)

model_path=model_save_folder+'epoch_{epoch:02d}-val_acc_{val_acc:.4f}-val_loss_{val_loss:.4f}.h5'
es = EarlyStopping(monitor='val_loss',patience=20)

## For tensorboard monitoring
tensorboard=TensorBoard(log_dir=model_save_folder+"{}".format(time()))

checkpointer = ModelCheckpoint(filepath=model_path, monitor='val_acc', verbose=1, save_best_only=False)
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    hist = model.fit(X_train, y_train, validation_data=(X_test, y_test), batch_size=50, epochs=100, verbose=1, shuffle=True, callbacks=[checkpointer,es,tensorboard])

##

#checkpointer = ModelCheckpoint(filepath=model_path, monitor='val_loss', verbose=1, save_best_only=True)
#hist = model.fit(X_train, y_train, validation_data=(X_test, y_test), batch_size=50, epochs=100, verbose=1, shuffle=True, callbacks=[checkpointer,es])

np.save(model_save_folder+'hist.npy',hist)
#################################  Step 8 : Deep learning Evaluation ################################# 
## TO do list : Evaluation package ->  ROC, AUC ...  

