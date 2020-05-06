# -*- coding: utf-8 -*-
"""
Created on Sun Dec  1 10:59:53 2019

@author: Yang
"""

import numpy
import pandas as pd
import numpy as np

##########观测函数
def hfun(X=None,k=None,*args,**kwargs):
    #varargin = hfun.varargin
    #nargin = hfun.nargin
    Q=np.dot(X[0],np.exp(np.dot(X[1],k))) + np.dot(X[2],np.exp(np.dot(X[3],k)))
    return(Q)
##########重采样
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

########主函数
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
Q=numpy.dot(cita,np.diag(w))
F=np.eye(4)
R=0.001

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



for k in range(1,N):
    for i in range(0,M):
        Xm[:,i,k]=np.dot(F,Xm[:,i,k-1])+ np.dot(np.sqrt(Q),np.random.randn(4,1)).reshape(4,)
    for i in range(0,M):
        Zm[0,i,k]=hfun(Xm[:,i,k],k)
        W[k,i]=np.exp(- (Z[0,k] - Zm[0,i,k]) ** 2 / 2 / R) + 1e-99
    W[k,:]=W[k,:] / sum(W[k,:])
    inputs=np.arange(0,M,1)
    outIndex=residualR(inputs,np.transpose(W[k,:]))
    Xm[:,:,k]=Xm[:,outIndex[:].astype(int),k]
    Xpf[:,k]=[np.mean(Xm[0,:,k]),np.mean(Xm[1,:,k]),np.mean(Xm[2,:,k]),np.mean(Xm[3,:,k])]
    Zpf[0,k]=hfun(Xpf[:,k],k)
    
import matplotlib.pyplot as plt
plt.plot(B0005_Cycle,B0005_Capacity,'.k')
X=np.array(B0005_Cycle[0:N])
plt.plot(X,Zpf[0,:],'g')


