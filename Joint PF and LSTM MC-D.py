# -*- coding: utf-8 -*-
"""
Created on Sat Dec  7 21:17:47 2019

@author: Yang
"""

import numpy
import pandas as pd
import numpy as np
import keras.backend as K
from keras.models import load_model
from sklearn.preprocessing import MinMaxScaler 
sc=MinMaxScaler(feature_range=(0,1))
###############################################################################观测函数
def hfun(X=None,k=None,*args,**kwargs):
    #varargin = hfun.varargin
    #nargin = hfun.nargin
    Q=np.dot(X[0],np.exp(np.dot(X[1],k))) + np.dot(X[2],np.exp(np.dot(X[3],k)))
    return(Q)
###############################################################################重采样
def residualR(inIndex=None,q=None,*args,**kwargs):
    S=q.shape[0]
    N_babies=np.zeros([1,S])
    q_res=np.multiply(S,np.transpose(q))
    N_babies=np.fix(q_res)
    N_res=int(S-sum(N_babies))
    if (N_res!= 0):
        q_res=(q_res - N_babies) / N_res
        cumDist=q_res.cumsum(0)
        u=np.fliplr((np.random.rand(1,N_res)**(1 / (np.arange(N_res,0,- 1).reshape(1,N_res)))).cumprod(1))  #u为1*N_res的二维数组
        j=0
        for i in range(0,N_res):
            while (u[0,i] > cumDist[j]):
                j=j + 1
            N_babies[j]=N_babies[j] + 1
    index=0
    outIndex=np.zeros([100])
    for i in range(0,S):
        if (int(N_babies[i]) > 0):
            for j in range(index,index + int(N_babies[i])):
                outIndex[j]=inIndex[i]
        index=index + int(N_babies[i])
    return(outIndex)

###############################################################################MC-dropout模拟
regress=load_model('B0005_cycle60_training model_1')                          #载入训练的LSTM模型

class KerasDropoutPrediction(object):
    def __init__(self,model):
        self.f = K.function([model.layers[0].input, K.learning_phase()], 
                            [model.layers[-1].output])

    def predict(self, x, n_iter = 100):
        result = []
        for i in range(n_iter):
            result.append(self.f((x, 1)))
        result=np.array(result)
        prediction = result.mean(axis=0)
        uncertainty = result.std(axis=0)
        return prediction, uncertainty    

kdp = KerasDropoutPrediction(regress)



###############################################################################主函数
io = r'C:\Users\Yang\Desktop\毕业设计\电池容量数据\NASA实验数据\Battery Dataset One\B0005&6&7&18_discharge&capacity.xlsx'
B0005_data=pd.read_excel(io,sheet_name='B0005',header=0)
f1=['cycle','Capacity']
dataset=B0005_data[f1]

B0005_Cycle=dataset['cycle']
B0005_Capacity=dataset['Capacity']


#B0006_Cycle= xlsread(filename,2,'B2:B169');B0006_Capacity= xlsread(filename,2,'K2:K169')
#B0007_Cycle= xlsread(filename,3,'B2:B169');B0007_Capacity= xlsread(filename,3,'C2:C169')
#B0018_Cycle= xlsread(filename,4,'B2:B133');B0018_Capacity= xlsread(filename,4,'C2:C133')

N=len(B0005_Cycle)
M=100
cita=1e-4
wa=0.000001
wb=0.01
wc=0.1
wd=0.0001
w= [wa,wb,wc,wd]
Q=numpy.dot(cita,np.diag(w))    #过程噪声协方差
F=np.eye(4)           #驱动矩阵
R=0.001          #观测噪声协方差

a=1.78967
b=-0.010507
c=0.1285245
d=-0.017541 #B0005参数
#a=1.93033;b=-0.002556;c=-0.0889088;d=-0.040967     #B0006参数
#a=1.84611;b=-0.005507;c=0.0371874667;d=-0.038583   #B0007参数
#a=1.81833;b=-0.003458;c=0.0719467;d=-0.057241      #B00018参数

X0=np.transpose([a,b,c,d])
Xpf=numpy.zeros([4,N])
Xpf[:,0]=X0
Xm=numpy.zeros([4,M,N])

for i in range(0,M):
    Xm[:,i,0]=(np.reshape(X0,(4,1)) + np.dot(np.sqrt(Q),np.random.randn(4,1))).reshape(4,)


Z=numpy.zeros([1,N])
Z[:,:]=(np.transpose(np.mat(B0005_Capacity))).reshape(-1)
Zm=np.zeros([1,M,N])
Zpf=np.zeros([1,N])
W=np.zeros([N,M])


##########联合预测
SP=59           #联合预测起点循环数为80，python中计算向前一位取79  
  
Prep=Z[0,:]
Prep=sc.fit_transform(Prep.reshape(-1,1))

for k in range(1,N):
    for i in range(0,M):
        Xm[:,i,k]=np.dot(F,Xm[:,i,k-1])+ np.dot(np.sqrt(Q),np.random.randn(4,1)).reshape(4,)
    for i in range(0,M):
        Zm[0,i,k]=hfun(Xm[:,i,k],k)
        if k>=SP:
            X_input=Prep[k-5:k]
            X_input=np.reshape(X_input,(1,X_input.shape[0],1))
            Predict_k=kdp.predict(X_input,n_iter = 10)                         #LSTM one-step ahead prediction
            mean_k=sc.inverse_transform(Predict_k[0]).reshape(1,-1)
            sigma_k=sc.inverse_transform(Predict_k[1]).reshape(1,-1)                  
            Z[0,k]=sc.inverse_transform(mean_k).reshape(1,)
        W[k,i]=np.exp(- (Z[0,k] - Zm[0,i,k]) ** 2 / 2 / R) + 1e-99
    W[k,:]=W[k,:] / sum(W[k,:])
    inputs=np.arange(0,M,1)
    outIndex=residualR(inputs,np.transpose(W[k,:]))
    Xm[:,:,k]=Xm[:,outIndex[:].astype(int),k]                                 #输出重采样粒子排序outIndex需要转为int
    Xpf[:,k]=[np.mean(Xm[0,:,k]),np.mean(Xm[1,:,k]),np.mean(Xm[2,:,k]),np.mean(Xm[3,:,k])]
    Zpf[0,k]=hfun(Xpf[:,k],k)
    if k>=SP:
        Z[0,k]=Zpf[0,k]                                                       #PF输出的滤波值重新反代，成为下一次预测的输入


###############################################################################画图
import matplotlib.pyplot as plt
plt.plot(B0005_Cycle,B0005_Capacity,'.k')
X=np.array(B0005_Cycle[SP:N])
plt.plot(X,Z[0,SP:N],'g')


###############################################################################预测的RMSE
from sklearn.metrics import mean_squared_error
test=numpy.zeros([1,N])
test[:,:]=(np.transpose(np.mat(B0005_Capacity))).reshape(-1)
RMSE = np.sqrt(mean_squared_error(test[0,SP:N], Z[0,SP:N]))
print('Test RMSE: %.3f' % RMSE)
