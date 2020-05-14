# -*- coding: utf-8 -*-
"""
Created on Sun Sep 18 16:17:22 2016

@author: User
"""
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import numpy as np
import pandas as pt

    
def select(k,x,y):
    
    '''x=(0:198)
    y=(0:198) '''

    xtrain=x[:k]
    ytrain=y[:k]
    xtest=x[k:]
    ytest=y[k:]
    
    return xtrain,ytrain,xtest,ytest

def cost_computation(xtrain,ytrain,n,m,b):
	sum1=0
	z=0
	for i in range(xtrain.size):
		sum1=sum1+(((n*xtrain[i]*xtrain[i]+m*xtrain[i]+b)-ytrain[i])**2)
		z=sum1/(2*(xtrain.size))
	return z

def stepGradient(b_current, m_current,n_current,xtrain,ytrain, learningRate):
    b_gradient = 0
    m_gradient,n_gradient = 0,0
    N = float(xtrain.size)
    for i in range(xtrain.size):
        b_gradient-=(2/N)*(ytrain[i]-((n_current*(xtrain[i]**2)+m_current*xtrain[i])+b_current))
        m_gradient-=(2/N)*xtrain[i]*(ytrain[i]-((n_current*(xtrain[i]**2)+m_current*xtrain[i])+b_current));n_gradient-=(2/N)*xtrain[i]*xtrain[i]*(ytrain[i]-((n_current*(xtrain[i]**2)+m_current*xtrain[i])+b_current))
    new_b=b_current-(learningRate*b_gradient)
    new_m=m_current-(learningRate*m_gradient)
    new_n=n_current-(learningRate*n_gradient)
    return (new_b, new_m,new_n)
def finderror(xtrain,ytrain,xtest,ytest):
	alpha=0.00000000455
	mvalue,nvalue=1,1
	bvalue,error=1,[]
	z,newz=1000000,100000
      
     
	while(z>150):
               
		newbvalue,newmvalue,newnvalue=stepGradient(bvalue,mvalue,nvalue,xtrain,ytrain,alpha);z=newz
		newz=cost_computation(xtrain,ytrain,nvalue,mvalue,bvalue)
		nvalue,mvalue,bvalue=newnvalue,newmvalue,newbvalue
	return cost_computation(xtest,ytest,nvalue,mvalue,bvalue)
	
def main():
	#initializing the data set
	data=pt.read_excel('BreastCancerData.xlsx')
      
	x=np.array(data[u'Perimeter'],dtype=np.float64)
	y=np.array(data[u'Compactness'],dtype=np.float64);k=80
      
	for i in range(2,10):
			newerror=[]
			k=x.size*(10-i)/10;xtrain,ytrain,xtest,ytest=select(k,x,y);newerror.append(finderror(xtrain,ytrain,xtest,ytest));print newerror
            #print len(xtrain),len(xtest),len(ytrain),len(ytest)
            
            
	
	
if __name__=="__main__":
	main()
