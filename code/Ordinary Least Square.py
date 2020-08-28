# -*- coding: utf-8 -*-
"""
Created on Mon Aug 10 18:36:43 2020

@author: Nclab
"""
import numpy as np
import matplotlib.pyplot as plt
#y=ax+b
rng=np.random.RandomState(1)
#numpy.random.randn(d0, d1, …, dn)是從常態分配中返回一個或多個值 
x=10*rng.rand(50)
#numpy.random.rand(d0, d1, …, dn)的數值會產生在(0,1)之間
y=2*x-5+rng.rand(50)

def linear_regression(x,y):
  x=np.concatenate((np.ones((x.shape[0],1)),x[:,np.newaxis]),axis=1)
  y=y[:,np.newaxis]
  beta=np.matmul(np.matmul(np.linalg.inv(np.matmul(x.T,x)),x.T),y)
  return beta
#將x,y帶入
by_hand =linear_regression(x,y)
#print出截距項
print(by_hand[0])
#print出斜率項
print(by_hand[1])
#任意建立新的點
xs=np.linspace(0,10,200)
ys=by_hand[0]+by_hand[1]*xs
#原本的data point
plt.scatter(x,y,s=100,alpha=0.3)
plt.plot(xs,ys,'r',linewidth=3)
plt.xlabel('x',fontsize=20)
plt.ylabel('y',rotation=0,fontsize=20)
