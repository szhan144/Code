import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
import statsmodels.tsa.arima_process
from statsmodels.tsa.ar_model import AR
#################################(1)##################################
##(a)
np.random.seed(123)
ar = [0.8]
ma = [1]
y = statsmodels.tsa.arima_process.arma_generate_sample(ar,ma,nsample=50,sigma=1,distrvs=np.random.randn)
y=y+500
print(np.mean(y))

plt.plot(np.arange(1,31),y[0:30],'bo-')
plt.plot(np.arange(30,51),y[29:],'bo-',mfc='none')
plt.xlabel("k")
plt.ylabel("$X_k$")
#plt.show()

model = AR(y[:30])
model_fit =model.fit()
y_hat = model_fit.predict(30,49)
print(np.mean(y_hat))
print(np.mean(y[30:]))

ll=[]
for i ,l in zip(y_hat,y[30:]):
       ll.append((pow((i-l),2)))
print(np.mean(ll))

plt.figure(21)
plt.subplot(211)
plt.scatter(np.arange(31,51),y_hat)
plt.xlabel("k")
plt.ylabel("$y_hat$")
plt.subplot(212)
plt.scatter(np.arange(31,51),y[30:])
plt.xlabel("k")
plt.ylabel("$y$")
#plt.show()
#I think the predicted value are very close to the actual value.

##(b)
X_1 = y[29]
l2=[]
for i in range(20):
	X = (0.8*X_1)+100
	l2.append(X)
	X_1 = X
plt.figure(11)
plt.plot(np.arange(1,31),y[0:30],'bo-')
plt.plot(np.arange(30,51),y[29:],'bo-',mfc='none')
plt.scatter(x=np.arange(31,51),y=l2,marker='s',c='',edgecolors='b')
plt.xlabel("k")
plt.ylabel("$X_k$")
#plt.show()

##(c)
c_ = []
for i,l in zip(y[29:],l2):
	c_.append(pow(i-l2,2))
root_mse = pow(np.mean(c_),0.5)

up_bond_ninezero = list(map(lambda x: x+root_mse*(-norm.ppf(0.05)),l2))
lower_bond_ninezero = list(map(lambda x: x-root_mse*(-norm.ppf(0.05)),l2))

up_bond_ninefive = list(map(lambda x: x+root_mse*(-norm.ppf(0.025)),l2))
lower_bond_ninefive = list(map(lambda x: x-root_mse*(-norm.ppf(0.025)),l2))

up_bond_ninenine = list(map(lambda x: x+root_mse*(-norm.ppf(1/200)),l2))
lower_bond_ninenine = list(map(lambda x: x-root_mse*(-norm.ppf(1/200)),l2))

plt.figure(11)
plt.plot(np.arange(1,31),y[0:30],'ro-')
plt.plot(np.arange(30,51),y[29:],'bo-',mfc='none')
plt.scatter(x=np.arange(31,51),y=l2,marker='s',c='',edgecolors='m')

plt.plot(np.arange(31,51),lower_bond_ninezero,'y--',label='0.9 confidence interval')
plt.plot(np.arange(31,51),up_bond_ninezero,'y--')

plt.plot(np.arange(31,51),lower_bond_ninefive,'g',label='0.95 confidence interval')
plt.plot(np.arange(31,51),up_bond_ninefive,'g')

plt.plot(np.arange(31,51),lower_bond_ninenine,'b-.',label='0.99 confidence interval')
plt.plot(np.arange(31,51),up_bond_ninenine,'b-.')

plt.xlabel("k")
plt.ylabel("$X_k$")
plt.legend(loc = 'upper left')
plt.show()

## From the plot, I think the actual values drop in the bound of confidence level 99%, but not in the 90% and 95% one.
#################################2############################
##(a)
np.random.seed(123)
ar = [1.5,-0.75]
ma = [1,0]
y = statsmodels.tsa.arima_process.arma_generate_sample(ar,ma,nsample=50,sigma=1,distrvs=np.random.randn)
y = y+400

plt.plot(np.arange(1,31),y[0:30],'bo-')
plt.plot(np.arange(30,51),y[29:],'bo-',mfc='none')
plt.xlabel("k")
plt.ylabel("$X_k$")
plt.show()
model = AR(y[:30])
model_fit =model.fit()
y_hat = model_fit.predict(30,49)
print(np.mean(y_hat))
print(np.mean(y[30:]))

ll=[]
for i ,l in zip(y_hat,y[30:]):
       ll.append((pow((i-l),2)))
print(np.mean(ll))

plt.figure(21)
plt.subplot(211)
plt.scatter(np.arange(31,51),y_hat)
plt.xlabel("k")
plt.ylabel("$y_hat$")
plt.subplot(212)
plt.scatter(np.arange(31,51),y[30:])
plt.xlabel("k")
plt.ylabel("$y$")
plt.show()
#I think the predicted value are very close to the actual value.

##(b)
X_1 = y[29]
X_2 = y[28]
l2=[]
for i in range(20):
	X = (1.5*X_1-0.75*X_2)+100
	l2.append(X)
	X_1, X_2 = X, X_1
plt.figure(11)
plt.plot(np.arange(1,31),y[0:30],'bo-')
plt.plot(np.arange(30,51),y[29:],'bo-',mfc='none')
plt.scatter(x=np.arange(31,51),y=l2,marker='s',c='',edgecolors='b')
plt.xlabel("k")
plt.ylabel("$X_k$")
plt.show()

##(c)
c_ = []
for i,l in zip(y[29:],l2):
	c_.append(pow(i-l2,2))
root_mse = pow(np.mean(c_),0.5)

up_bond_ninezero = list(map(lambda x: x+root_mse*(-norm.ppf(0.05)),l2))
lower_bond_ninezero = list(map(lambda x: x-root_mse*(-norm.ppf(0.05)),l2))

up_bond_ninefive = list(map(lambda x: x+root_mse*(-norm.ppf(0.025)),l2))
lower_bond_ninefive = list(map(lambda x: x-root_mse*(-norm.ppf(0.025)),l2))

up_bond_ninenine = list(map(lambda x: x+root_mse*(-norm.ppf(1/200)),l2))
lower_bond_ninenine = list(map(lambda x: x-root_mse*(-norm.ppf(1/200)),l2))

plt.figure(11)
plt.plot(np.arange(1,31),y[0:30],'ro-')
plt.plot(np.arange(30,51),y[29:],'bo-',mfc='none')
plt.scatter(x=np.arange(31,51),y=l2,marker='s',c='',edgecolors='m')

plt.plot(np.arange(31,51),lower_bond_ninezero,'y--',label='0.9 confidence interval')
plt.plot(np.arange(31,51),up_bond_ninezero,'y--')

plt.plot(np.arange(31,51),lower_bond_ninefive,'g',label='0.95 confidence interval')
plt.plot(np.arange(31,51),up_bond_ninefive,'g')

plt.plot(np.arange(31,51),lower_bond_ninenine,'b-.',label='0.99 confidence interval')
plt.plot(np.arange(31,51),up_bond_ninenine,'b-.')

plt.xlabel("k")
plt.ylabel("$X_k$")
plt.legend(loc = 'upper left')
plt.show()

## From the plot, I think the actual values drop in the bound of confidence level 99% and 95%, but not in the 90% one.