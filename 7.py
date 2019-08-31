import numpy as np
import matplotlib.pyplot as plt
import matplotlib.mlab as mlab
import scipy.stats as stats
from scipy.stats import kstest

np.random.seed(123)
with open('aa.txt', 'r') as f:
    data = f.readlines()
a=data[0].split(",")
b=list(map(float,a))
def y_k(k):
  y = np.sqrt((k*(1000-k))/(pow(1000,2)))
  e = np.mean(b[:k])-np.mean(b[k:])
  return(y*e)
l_=list()
for i in range(1,len(b)):
  c = abs(y_k(i))
  l_.append(c)


def y_k1(k):
  y = (k*(1000-k))/(pow(1000,2))
  e = np.mean(b[:k])-np.mean(b[k:])
  return(y*e)
l_1=list()
for i in range(1,len(b)):
  c1 = abs(y_k1(i))
  l_1.append(c1)


def y_k0(k):
  e = np.mean(b[:k])-np.mean(b[k:])
  return(e)
l_0=list()
for i in range(1,len(b)):
  c0 = abs(y_k0(i))
  l_0.append(c0)
print('121',l_0)
print('121',l_1)
plt.figure()
plt.plot(np.arange(1,1000),l_)
plt.ylim([0,0.1])
plt.xlabel('k',fontsize=15)
plt.ylabel('The Brodsky–Darkhovsky statistic',fontsize=15)
plt.annotate('max statistic',xy=(501,0.091),xytext=(600,0.078),arrowprops=dict(facecolor='red', shrink=0.009))
plt.show()

plt.figure(12)
plt.title('Different Power of Weight')
plt.subplot(121)
plt.plot(np.arange(1,1000),l_1,'g',label='power 1')
plt.xlabel('k',fontsize=15)
plt.ylabel('The Brodsky–Darkhovsky statistic',fontsize=15)
plt.legend(loc='upleft')
plt.subplot(122)
plt.plot(np.arange(1,1000),l_0,'r',label='power 0')
plt.xlabel('k',fontsize=15)
plt.ylabel('The Brodsky–Darkhovsky statistic',fontsize=15)
plt.legend(loc='upleft')
plt.show()
cn = float(1.36/pow(1000,0.5))
print("Critical number is",cn)
print("y max is:",max(l_))
print("y max1 is:",max(l_1))
print("y max0 is:",max(l_0))
print("Changing point is:",1+l_.index(max(l_)))
print("Changing point is:",1+l_1.index(max(l_1)))
print("Changing point is:",1+l_0.index(max(l_0)))


plt.figure(12)
plt.subplot(121)
font1={'size':15}
n, bins, patches = plt.hist(b[:501], 50,normed=1, facecolor='blue', alpha=0.5)
y = mlab.normpdf(bins,np.mean(b[:501]),np.std(b[:501]))
plt.plot(bins,y,'r--', label='Normal Density')
plt.legend(loc='upleft',prop=font1)
plt.title("Points before the changing point",fontsize=18)
plt.subplot(122)
n, bins, patches = plt.hist(b[502:], 50,normed=1, facecolor='blue', alpha=0.5)
y = mlab.normpdf(bins,np.mean(b[502:]),np.std(b[502:]))
plt.plot(bins,y,'r--', label='Normal Density')
plt.legend(loc='upleft',prop=font1)
plt.title("Points after the changing point",fontsize=18)
plt.show()
#print(kstest(b[:501],'norm'))
#print(kstest(b[502:],'norm'))
cn = float(1.36/pow(1000,0.5))



font1={'size':20}
plt.scatter(np.arange(1,502),b[:501],c='g')
plt.scatter(np.arange(503,1001),b[502:],c='g')
plt.scatter(x=502,y=b[501],marker='^',c='m',label='changing point at 502')
plt.legend(loc='upleft',prop=font1)
plt.show()

plt.figure(12)
plt.subplot(121)
stats.probplot(b[:501], dist="norm", plot=plt)
plt.title("Normal Q-Q plot before the change")
plt.subplot(122)


stats.probplot(b[501:], dist="norm", plot=plt)
plt.title("Normal Q-Q plot after the change")
plt.show()

##changing point is extremely small
np.random.seed(123)
first = list(np.random.normal(0,1,10))
second = list(np.random.normal(.001,1,990))

data = first+second
plt.figure()
plt.scatter(np.arange(1,1001),data)
plt.show()


def y_k(k):
  y = np.sqrt((k*(1000-k))/(pow(1000,2)))
  e = np.mean(data[:k])-np.mean(data[k:])
  return(y*e)
l_=list()
for i in range(1,len(data)):
  c = abs(y_k(i))
  l_.append(c)
print('point',max(l_),1+l_.index(max(l_)))
plt.figure()
plt.plot(np.arange(1,1000),l_)
plt.ylim([0,0.1])
plt.xlabel('k',fontsize=15)
plt.ylabel('The Brodsky–Darkhovsky statistic',fontsize=15)
plt.show()

y = []
for i in range(1,1000):
  y.append(np.std(data[:i]))
print('as',len(y))
print(y.index(max(y)))
plt.figure()
plt.plot(np.arange(1,1000),y)
plt.xlabel('k',fontsize=15)
plt.ylabel('Root of Variance',fontsize=15)
plt.show()
def y_k0(k):
  e = np.mean(data[:k])-np.mean(data[k:])
  return(e)
l_0=list()
for i in range(1,len(b)):
  c0 = abs(y_k0(i))
  l_0.append(c0)
plt.figure(12)
plt.title('Different Power of Weight')
plt.subplot(121)
plt.plot(np.arange(1,1000),l_1,'g',label='power 1')
plt.xlabel('k',fontsize=15)
plt.ylabel('The Brodsky–Darkhovsky statistic',fontsize=15)
plt.legend(loc='upleft')
plt.subplot(122)
plt.plot(np.arange(1,1000),l_0,'r',label='power 0')
plt.xlabel('k',fontsize=15)
plt.ylabel('The Brodsky–Darkhovsky statistic',fontsize=15)
plt.legend(loc='upleft')
plt.show()
##changing point is extremely large

first = list(np.random.normal(0,1,10))
second = list(np.random.normal(0.01,1,990))
data = first+second
def y_k(k):
  y = np.sqrt((k*(1000-k))/(pow(1000,2)))
  e = np.mean(data[:k])-np.mean(data[k:])
  return(y*e)

l_=list()
for i in range(1,len(data)):
  c = abs(y_k(i))
  l_.append(c)


print(max(l_),1+l_.index(max(l_)))




y = []
for i in range(1,1000):
  y.append(np.std(data[:i]))
print('as',len(y))
print(y.index(max(y)))
plt.figure()
plt.plot(np.arange(1,1000),y)
plt.xlabel('k',fontsize=15)
plt.ylabel('Root of Variance',fontsize=15)
plt.show()
