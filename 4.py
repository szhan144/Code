import numpy as np
from scipy.stats import kstest
import math
import time
import matplotlib.pyplot as plt


###1
start = time.clock()
l1=[]
l2=[]
for i in range(10000):
 u1=np.random.uniform(0,1,2)
 x1=pow(-2*np.log(u1[0]),0.5)*(np.cos(2*(np.pi)*u1[1]))
 x2=pow(-2*np.log(u1[0]),0.5)*(np.cos(2*(np.pi)*u1[1]))
 l1.append(x1)
 l2.append(x2)
print(kstest(l1,'norm'))
end=time.clock()-start
print("The estimate of time is:",end)
#Comment: According the kstest, obtained Xi’s conform to the standard Gaussian distribution.


##2
start = time.clock()
def fun(N):
 n=0
 l1=[]
 l2=[]
 while n<=N-1:
    u1=np.random.uniform(-1,1,2)
    r=pow(u1[0],2)+pow(u1[1],2)
    if(r<=1)&(r>0):
      x1=pow((-2*(np.log(r)))/r,0.5)*u1[0]
      x2=pow((-2*(np.log(r)))/r,0.5)*u1[1]
      l1.append(x1)
      l2.append(x2)
      n=n+1
    else:
      continue
 return(l1)
print(kstest(fun(10000),'norm'))
end=time.clock()-start
print("The estimate of time is:",end) 
#Comment: According the kstest, obtained Xi’s conform to the standard Gaussian distribution. And the Marsaglia polar method is faster than Box–Muller method about 6.25%.


##3
u1=np.random.uniform(0,1,10000)
l1=[]
for i in range(10000):
 if u1[i]>0.5:
  s=1
  l1.append(s)
 else:
  s=-1
  l1.append(s)
u2=np.random.uniform(0,1,10000)

x=[]
for i in range(10000):
 a=(l1[i]*(pow((math.pi)/8,0.5))*(math.log((1+u2[i]/1-u2[i]))))
 x.append(a)
print(kstest(x,'norm'))
#Comment: According the kstest, obtained Xi’s do not conform to the Gaussian distribution.
plt.hist(x,bins=10)
plt.show()
#Because from the histgram plot, I think the obtained Xis distribute at three clustering interval.
#Use another approximation function.

