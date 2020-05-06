# -*- coding: utf-8 -*-
"""
Created on Mon Dec  2 21:34:38 2019

@author: Yang
"""

weights = 0.01* np.ones([1,M])
Zfm=np.zeros([M,N-SP])
RULs=np.zeros([M,1])
for j in range(0,M):  
    for k in range(SP,N):
        Zfm[j,k-SP]=hfun(Xm[:,j,k],k)
for i in range(0,M):
    for j in range(0,N-SP):
        if (Zfm[i,j]<=1.4):
            RULs[i]=j
            break

RULs=RULs.reshape(M,)+np.ones([M,])*SP
junzhi=np.mean(RULs)#计算每一行
sigma=np.std(RULs)
