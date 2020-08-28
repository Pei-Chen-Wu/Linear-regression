# -*- coding: utf-8 -*-
"""
Created on Wed Aug 12 15:33:59 2020

@author: Nclab
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

rng=np.random.RandomState(1)
#numpy.random.randn(d0, d1, …, dn)是從常態分配中返回一個或多個值 
x=10*rng.rand(50)
#numpy.random.rand(d0, d1, …, dn)的數值會產生在(0,1)之間
y=2*x-5+rng.rand(50)

model=LinearRegression()
model.fit(x[:,np.newaxis],y)
print('intercept:',model.intercept_)
print('coefficient:',model.coef_)
print('score:',model.score(x[:,np.newaxis],y))

plt.plot(x,y,'o',markersize=15,alpha=0.3)
plt.plot(x,model.intercept_+model.coef_*x,linewidth=5)
plt.xlabel('x',fontsize=20)
plt.ylabel('y',fontsize=20,rotation=0)
