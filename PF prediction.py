# -*- coding: utf-8 -*-
"""
Created on Sun Dec  1 13:28:36 2019

@author: Yang
"""

import matplotlib.pyplot as plt
import numpy as np
#def hfun(X=None,k=None,*args,**kwargs):
    #varargin = hfun.varargin
    #nargin = hfun.nargin
    #Q=np.dot(X[0],np.exp(np.dot(X[1],k))) + np.dot(X[2],np.exp(np.dot(X[3],k)))
    #return(Q)
start=79
Xf=np.zeros([1,N-start])
Zf=np.zeros([1,N-start])
for k in range(start,N):
    Zf[0,k - start ]=hfun(Xpf[:,start],k)
    Xf[0,k - start ]=k

plt.figure(figsize=(10, 5))
#plt.plot([0,0:start],Zpf[0,:start],'-b.')
plt.plot(B0005_Cycle,B0005_Capacity,'.b')
plt.plot(Xf,Zf,'-g.')
plt.xlim([1.4,1.4],'r--','linewidth',1)
plt.legend('实验测量数据','滤波估计数据','预测电池容量')
plt.ylim([1.2,2])