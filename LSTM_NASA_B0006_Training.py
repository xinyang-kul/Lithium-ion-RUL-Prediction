# -*- coding: utf-8 -*-
"""
Created on Mon Sep 30 21:17:57 2019

@author: Yang
"""

import numpy
import pandas as pd
import numpy as np
#import matplotlib.pyplot as plt
#import seaborn as sns
from sklearn.externals import joblib

#from sklearn import metrics # for the check the error and accuracy of the model
#from sklearn.metrics import mean_squared_error

## for Deep-learing:
#import keras
from keras.layers import Dense
from keras.models import Sequential
#from keras.utils import to_categorical
#from keras.optimizers import SGD 
#from keras.callbacks import EarlyStopping
#from keras.utils import np_utils
#import itertools
from keras.layers import LSTM
#from keras.layers import Bidirectional
#from keras.layers.convolutional import Conv1D
#from keras.layers.convolutional import MaxPooling1D
from keras.layers import Dropout
io = r'C:\Users\Yang\Desktop\毕业设计\电池容量数据\NASA实验数据\Battery Dataset One\B0005&6&7&18_discharge&capacity.xlsx'
B0006_data=pd.read_excel(io,sheet_name='B0006',header=0)
#features=['Batt_name','cycle','amb_temp','voltage_battery','current_battery','temp_battery','current_load','voltage_load','time','Capacity','H']
f1=['cycle','Capacity']

dataset=B0006_data[f1]
data_train=dataset[(dataset['cycle']>0)]
data_set_train=data_train.iloc[:,1:2].values
#data_test=dataset[(dataset['cycle']>=100)]
#data_set_test=data_test.iloc[:,1:2].values

from sklearn.preprocessing import MinMaxScaler   #标准化
sc=MinMaxScaler(feature_range=(0,1))
data_set_train=sc.fit_transform(data_set_train)
#data_set_test=sc.transform(data_set_test)   # ????

X_train=[]
y_train=[]
#take the last 10t to predict 10t+1
for i in range(10,168):
    X_train.append(data_set_train[i-10:i,0])
    y_train.append(data_set_train[i,0])
X_train,y_train=np.array(X_train),np.array(y_train)
X_train=np.reshape(X_train,(X_train.shape[0],X_train.shape[1],1))#X_train.shape[0]=69,X_train.shape[1]=10,将X_train变成（69,10,1）矩阵；


#Apply LSTM
regress=Sequential()
regress.add(LSTM(units=50, return_sequences=True, input_shape=(X_train.shape[1],1)))
regress.add(Dropout(0.3))

regress.add(LSTM(units=50, return_sequences=True))
regress.add(Dropout(0.3))


regress.add(LSTM(units=50, return_sequences=True))
regress.add(Dropout(0.3))

regress.add(LSTM(units=50))
regress.add(Dropout(0.3))


regress.add(Dense(units=1))

regress.compile(optimizer='Adam',loss='mean_squared_error')

regress.fit(X_train,y_train,epochs=200,batch_size=50)

filename='B0006training model.csv'
joblib.dump(regress,filename)