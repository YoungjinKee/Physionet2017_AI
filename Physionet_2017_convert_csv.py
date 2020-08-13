import matplotlib.pyplot as plt
import numpy as np
from os import listdir
from os.path import isfile, join
import pandas as pd
import scipy.io as sio


mypath = '../Downloads/DeepECG-master/DeepECG-master/training2017/'
onlyfiles = [f for f in listdir(mypath) if (isfile(join(mypath, f)) and f[0] == 'A')]
bats = [f for f in onlyfiles if f[7] == 'm']

data_length=np.zeros((len(bats),1))
target_train = np.zeros((len(bats),1))
Train_data = pd.read_csv(mypath + 'REFERENCE.csv', sep=',', header=None, names=None)

limit=20000
X=np.zeros((len(bats),limit))

for i in range(len(bats)):
        
        data_length[i]=(np.shape(sio.loadmat(mypath + bats[i])['val'])[1])
        
        if Train_data[1][i] == 'N':
            target_train[i] = 0
        elif Train_data[1][i] == 'A':
            target_train[i] = 1
        elif Train_data[1][i] == 'O':
            target_train[i] = 2
        else:
            target_train[i] = 3
        
        temp=sio.loadmat(mypath + bats[i])['val']
        zero_padding=np.zeros((1,limit-temp.shape[1]))
        data=np.hstack((temp, zero_padding))
        X[i]=data

#bats=np.array(bats)
#bats=bats.reshape(len(bats),1)
#final_arr=np.hstack((bats,data_length))


final_arr=np.hstack((data_length,target_train))
final_arr=np.hstack((final_arr,X))
np.savetxt('final_arr.csv',final_arr,fmt='%i', delimiter=',')
        