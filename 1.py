import numpy as np
import time



start = time.clock()


def fun(x,y):
    n=1000000
    n_pts_in_circ=0
    d=np.sqrt((np.add(np.multiply(x,x),np.multiply(y,y))))
    for i in range(len(d)):
           if d[i]<=1:
             n_pts_in_circ+=1
           else:
             pass
    pi_est=4*(n_pts_in_circ/n)
    print("The estimator of pie is",pi_est)
x=np.random.uniform(-1,1,1000000)
y=np.random.uniform(-1,1,1000000)
fun(x,y)

end=time.clock()-start
print("The estimate of time is:",end)




    
