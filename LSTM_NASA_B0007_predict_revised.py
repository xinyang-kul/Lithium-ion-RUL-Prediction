# -*- coding: utf-8 -*-
"""
Created on Wed Nov 27 19:43:35 2019

@author: Yang
"""

import numpy
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


from sklearn import metrics # for the check the error and accuracy of the model
from sklearn.metrics import mean_squared_error

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
B0007_data=pd.read_excel(io,sheet_name='B0007',header=0)
#features=['Batt_name','cycle','amb_temp','voltage_battery','current_battery','temp_battery','current_load','voltage_load','time','Capacity','H']
f1=['cycle','Capacity']

dataset=B0007_data[f1]
data_train=dataset[(dataset['cycle']<60)]
data_set_train=data_train.iloc[:,1:2].values
data_test=dataset[(dataset['cycle']>=60)]
data_set_test=data_test.iloc[:,1:2].values

from sklearn.preprocessing import MinMaxScaler   #标准化
sc=MinMaxScaler(feature_range=(0,1))
data_set_train=sc.fit_transform(data_set_train)
data_set_test=sc.transform(data_set_test)   # ????

X_train=[]
y_train=[]
#take the last 10t to predict 10t+1
for i in range(5,59):
    X_train.append(data_set_train[i-5:i,0])
    y_train.append(data_set_train[i,0])
X_train,y_train=np.array(X_train),np.array(y_train)
X_train=np.reshape(X_train,(X_train.shape[0],X_train.shape[1],1))#X_train.shape[0]=69,X_train.shape[1]=10,将X_train变成（69,10,1）矩阵；

rate=0.5
#Apply LSTM
regress=Sequential()
regress.add(LSTM(units=50, return_sequences=True, input_shape=(X_train.shape[1],1)))
regress.add(Dropout(rate))

regress.add(LSTM(units=50, return_sequences=True))
regress.add(Dropout(rate))

regress.add(LSTM(units=50, return_sequences=True))
regress.add(Dropout(rate))

regress.add(LSTM(units=50))
regress.add(Dropout(rate))

regress.add(Dense(units=1))

regress.compile(optimizer='adam',loss='mean_squared_error')
regress.fit(X_train,y_train,epochs=100,batch_size=50)

regress.save('B0007_cycle60_training model')
#Test the trained model
#Predictions
data_total=pd.concat((data_train['Capacity'],data_test['Capacity']),axis=0)
inputs=numpy.zeros(shape=(114,))
inputs[:5]=data_total[len(data_total)-len(data_test)-5:len(data_total)-len(data_test)].values
inputs=inputs.reshape(-1,1)
inputs=sc.transform(inputs)

#每一次的迭代输入

X_test=[]
X_test=np.array(X_test)
for i in range(5,114):
    X_test=inputs[i-5:i]
    X_test=np.reshape(X_test,(1,X_test.shape[0],1))
    pred=regress.predict(X_test)
    inputs[i]=pred

inputs=sc.inverse_transform(inputs)
inputs=inputs[:,0]
prediction=inputs[5:]

tests=data_test.iloc[:,1:2]
rmse = np.sqrt(mean_squared_error(tests, prediction))
print('Test RMSE: %.3f' % rmse)
metrics.r2_score(tests,prediction)

len(prediction)
data_test['pre']=prediction

#作图
plot_df = dataset.loc[(dataset['cycle']>=1),['cycle','Capacity']]
plot_per=data_test.loc[(data_test['cycle']>=60),['cycle','pre']]
sns.set_style("darkgrid")
plt.figure(figsize=(10, 5))
plt.plot(plot_df['cycle'], plot_df['Capacity'], label="Actual data", color='blue')
plt.plot(plot_per['cycle'],plot_per['pre'],label="Prediction data", color='red')
#plt.plot(pred)
#Draw threshold
plt.plot([0.,168], [1.4, 1.4])
plt.ylabel('Capacity')
# make x-axis ticks legible
adf = plt.gca().get_xaxis().get_major_formatter()
plt.xlabel('cycle')
