# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.

"""
#%matplotlib qt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import numpy as np
import pandas as pt

def cost_computation(x1,x2,y,n,m,b):
	sum1=0
	z=0
	for i in range(x1.size):
		sum1=sum1+(((n*x2[i]+m*x1[i]+b)-y[i])**2)
		z=sum1/(2*(x1.size))
	return z

def stepGradient(b_current, m_current,n_current,x2,x1,y, learningRate):
    b_gradient = 0
    m_gradient,n_gradient = 0, 0
    
    N = float(x1.size)
    for i in range(x1.size):
        b_gradient-=(2/N)*(y[i]-((n_current*x2[i])+(m_current*x1[i])+b_current))
        m_gradient-=(2/N)*x1[i]*(y[i]-((n_current*x2[i])+(m_current*x1[i])+b_current));n_gradient-=(2/N)*x2[i]*(y[i]-((n_current*x2[i])+(m_current*x1[i])+b_current))
    new_b=b_current-(learningRate*b_gradient)
    new_m=m_current-(learningRate*m_gradient);new_n=n_current-(learningRate*n_gradient)
    return (new_b, new_m, new_n)
def main():
	#initializing the data set
	data=pt.read_excel('BreastCancerData.xlsx')
     
	x1=np.array(data[u'Perimeter'],dtype=np.float64)
	x2=np.array(data[u'Area'],dtype=np.float64);y=np.array(data[u'Compactness'],dtype=np.float64)
	#x=np.array([1,2,3,4])
	#y=np.array([2,4,7,8])
	mvalues=np.zeros(x1.size+1)
	bvalues,nvalues=np.zeros(x1.size+1),np.zeros(x1.size+1)
	learningRate=0.00000092
	mvalue,nvalue=1,1
	bvalue,error=1,[]
	z,newz=1000000,100000
	i=0	
      
     
	while(z>0.004):
		#print mvalue,bvalue,z 
               
		newbvalue,newmvalue,newnvalue=stepGradient(bvalue,mvalue,nvalue,x2,x1,y,learningRate);z=newz
		newz=cost_computation(x1,x2,y,nvalue,mvalue,bvalue);error.append(newz)
		mvalue,bvalue,nvalue=newmvalue,newbvalue,newnvalue
		print nvalue,mvalue,bvalue,z,i
		i=i+1;plt.figure(1),plt.title('Cost Over Iterations'),plt.xlabel('No. Of Iterations'),plt.ylabel('Cost Function'),plt.scatter(i,newz,color='r')     
            
    #plt.scatter(x1,x2)
	#plt.plot(range(30),error[:30])
	#print mvalues[x.size],bvalues[x.size]
	xaxis=np.linspace(60,200,num=50);yaxis=np.linspace(250,2500,num=50)
     
	newx1=xaxis;zaxis=[nvalue*yaxis[i]+mvalue*xaxis[i]+bvalue for i in range(50)]
 	fig=plt.figure()
      
	ax=fig.add_subplot(111,projection='3d')
	ax.scatter(x1, x2, y,c='r',marker='o'),ax.scatter(xaxis, yaxis, zaxis,marker='*'),ax.set_xlabel('Perimeter'),ax.set_ylabel('Area'),ax.set_zlabel('Compactness'),ax.set_title('Performance of Model when Area is added')
   
	plt.show()
	
if __name__=="__main__":
	main()

