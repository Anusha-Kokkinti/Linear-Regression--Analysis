# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.

"""
#%matplotlib qt
import matplotlib.pyplot as plt
import numpy as np
import pandas as pt

def cost_computation(x,y,m,b):
	sum1=0
	z=0
	for i in range(x.size):
		sum1=sum1+(((m*x[i]+b)-y[i])**2)
		z=sum1/(2*(x.size))
	return z

def stepGradient(b_current, m_current,x,y, learningRate):
    b_gradient = 0
    m_gradient = 0
    N = float(x.size)
    for i in range(x.size):
        b_gradient-=(2/N)*(y[i]-((m_current*x[i])+b_current))
        m_gradient-=(2/N)*x[i]*(y[i]-((m_current*x[i])+b_current))
    new_b=b_current-(learningRate*b_gradient)
    new_m=m_current-(learningRate*m_gradient)
    return (new_b, new_m)
def main():
	#initializing the data set
	data=pt.read_excel('BreastCancerData.xlsx')
     
	x=np.array(data[u'Perimeter'],dtype=np.float64)
	y=np.array(data[u'Compactness'],dtype=np.float64)
	#x=np.array([1,2,3,4])
	#y=np.array([2,4,7,8])
	mvalues=np.zeros(x.size+1)
	bvalues=np.zeros(x.size+1)
	alpha=0.00007326
	mvalue=1
	bvalue,error=1,[]
	z,newz=1000000,100000
	i=0	
      
     
	while(z>0.009):
		#print mvalue,bvalue,z 
               
		newbvalue,newmvalue=stepGradient(bvalue,mvalue,x,y,alpha);z=newz
		newz=cost_computation(x,y,mvalue,bvalue);error.append(newz)
		mvalue,bvalue=newmvalue,newbvalue
		print mvalue,bvalue,z,i
		i=i+1
		plt.figure(1),plt.title('Cost Over Iterations'),plt.xlabel('No. Of Iterations'),plt.ylabel('Cost Function'),plt.scatter(i,newz,color='r')      
      
	x1=np.linspace(10,200,num=50)
	newx1=x1; 
	y1=[mvalue*i+bvalue for i in newx1]
 	plt.figure(2),plt.title('Linear Regression With One Variable'),plt.xlabel('Perimeter'),plt.ylabel('Compactness'),plt.plot(x1,y1,color='g');plt.scatter(x,y,color='r')
      #plt.scatter(y,x)
       
	plt.show()
	
if __name__=="__main__":
	main()

