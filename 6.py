import numpy as np
import math
##1
def func(x):
	return((0.5*(np.exp(-x)+np.exp(x)))/(pow((1-x**2),0.5)))

x=np.random.uniform(0,1,10000)
answer=(2*np.mean(func(x)))/(np.pi)
print(answer)
##I used the uniform distribution(0,1).

##2
def fun(u,v):
    x=pow(-2*np.log(u),0.5)*np.cos(2*np.pi*v)
    y=pow(-2*np.log(u),0.5)*np.sin(2*np.pi*v)
    return(u*pow(1+pow(np.sin(x*y),2),0.5)*(2*np.pi)/u)

a=np.random.uniform(0,1,10000)
b=np.random.uniform(0,1,10000)
answer=(np.mean(fun(a,b)))/(2*np.pi)
print(answer)
#Use the Box Muller method to replace the X and Y.