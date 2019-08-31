import numpy as np
import matplotlib.pyplot as plt
##0
def gen_random(N):
 y=[]*N
 for i in range(N):
     x=np.random.uniform(0,1,1)
     if (x[0]<0.35)&(x[0]>0):
         y.append(-2)
     elif (x[0]>=0.35)&(x[0]<=0.9):
         y.append(0)
     elif (x[0]>0.9)&(x[0]<=1):
         y.append(3)
 return(y)
         
print(gen_random(10))
plt.hist(gen_random(10000),bins=3)
plt.show()
#Comment: The plot basically matches my expectation that the frequency of 2 is about 0.35, frequency of 0 is about 0.55 and frequency of 3 is about 0.1.



##1
#(1),(2),(3)
y=[]*10000
for i in range(10000):
      X=np.random.uniform(0,1,1)
      y.append(pow(-1.2,-1)*(np.log(X[0])))
     
m=np.linspace(0,5,10000)
Y=1.2*(np.exp(-1.2*m))
plt.plot(m,Y)   
plt.show() 
plt.hist(y,bins=10)
plt.show()
#Comment: I think the curve agrees with histogram.

##2
def inner(N,p):
  Y=[]*N
  for k in range(N):
     X=np.random.uniform(0,1,1)
     if (X[0]<p)&(X[0]>=0):
       Y.append(1)
     else:
       Y.append(0)
  return(sum(Y))
a=[inner(26,0.77)for i in range(10000)]
a=np.array(a)
print(len(a[a>=14]))
b=np.random.binomial(26,0.77,10000)
print(len(b>=14))
#Comment: I think the python binomial function results are roughly consistent with my empirical estimate.




##3
def second(N,p):
  k=1
  n=0
  l=[]
  while n<=N-1:
     X=np.random.uniform(0,1,1)
     if (X[0]<p)&(X[0]>=0):
       n=n+1
       l.append(k)
       k=1
     else:
       k=k+1
  return(l)
print(second(12,0.12))
a=np.array(second(10000,0.12))
print(len(a[a>=10]))
b=np.random.geometric(0.12,10000)
print(len(b[b>=10]))
#Comment: I think the python geometric function results are roughly consistent with my empirical estimate.



##4
def third(N,lambd):
  s=1
  k=0
  n=0
  l=[]
  while n<=N-1:
    X=np.random.uniform(0,1,1)
    s=s*X[0]
    if s<=np.exp(-lambd):
        n=n+1
        l.append(k)
        k=0
        s=1
    else:
        k=k+1
  return(l)
print(third(200,1.5))
a=np.array(third(10000,1.5))
print(len(a[a>=10]))
b=np.random.poisson(1.5,10000)
print(len(b[b>=10]))
#Comment: I think the python geometric function results are roughly consistent with my empirical estimate.


c=np.array(third(10000,100))
print(len(c[c>=10]))
b=np.random.poisson(100,10000)
print(len(b[b>=10]))
#Comment: After lambda turns to be 100, all the random numbers are bigger than 10.