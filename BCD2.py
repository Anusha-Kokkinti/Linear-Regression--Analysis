import matplotlib.pyplot as plt
import numpy as np
import pandas as pt

def cost_computation(x,y,n,m,b):
    sum1=0
    z=0
    for i in range(x.size):
        sum1=sum1+(((n*x[i]*x[i]+m*x[i]+b)-y[i])**2)
        z=sum1/(2*(x.size))
    return z
def stepGradient(b_current,m_current,n_current,x,y,learningRate):
        b_gradient=0.0
        m_gradient=0.0
        n_gradient=0.0
        n = float(x.size)
        for i in range(x.size):
            n_gradient-=(2/n)*x[i]*x[i]*(y[i]-(n_current*(x[i]**2)+m_current*x[i]+b_current))
            b_gradient-=(2/n)*(y[i]-(n_current*(x[i]**2)+m_current*x[i]+b_current))
            m_gradient-=(2/n)*x[i]*(y[i]-(n_current*(x[i]**2)+m_current*x[i]+b_current))
            
        new_b=b_current-(learningRate*b_gradient)
        new_m=m_current-(learningRate*m_gradient)
        new_n=n_current-(learningRate*n_gradient)
        return(new_b,new_m,new_n)
def main():
    #initializing the data set
    data=pt.read_excel('BreastCancerData.xlsx')
    x=np.array(data[u'Perimeter'],dtype=np.float64)
    y=np.array(data[u'Compactness'],dtype=np.float64)
    #x=np.array([1,2,3,4])
    #y=np.array([2,4,7,8])
    mvalues=np.zeros(x.size+1)
    bvalues=np.zeros(x.size+1)
    nvalues=np.zeros(x.size+1)
    newx=np.zeros(100)
    newy=[2*i*i+1*i+3 for i in newx]
    alpha=0.00000000455
    mvalue=1
    bvalue=1
    nvalue=1
    newz=10000
    z=100000
    i=0	
    while(z>150):
        #print nvalue,mvalue,bvalue,z,i
        z=newz 
        values=stepGradient(bvalue,mvalue,nvalue,x,y,alpha)
        newbvalue=values[0]
        newmvalue=values[1]
        newnvalue=values[2]
        newz=cost_computation(x,y,nvalue,mvalue,bvalue)
        mvalue=newmvalue
        bvalue=newbvalue
        nvalue=newnvalue
        i=i+1
        plt.figure(1)
        plt.scatter(i,newz,color='g')
        plt.title('Cost over Iterations')
        plt.ylabel('Error Function')
        plt.xlabel('No. of iterations')
        print nvalue,mvalue,bvalue,z,i
        
    x1=np.linspace(50,200,num=50)
    newx1=x1
    y1=[nvalue*(i**2)+mvalue*i+bvalue for i in newx1]
    plt.figure(2)
    plt.plot(x1,y1,color='r')
    plt.scatter(x,y,color='g')
    plt.title('Quadratic Regression With One Variable')
    plt.xlabel('Perimeter')
    plt.ylabel('Compactness')
    plt.show()
if __name__=="__main__":
	main()

    