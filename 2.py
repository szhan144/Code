import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D


###Professor, please run the whole script, the results and plots will be shown. And comment is written during the script.

###1(a)####
def mid_square_rng(N,x_0):
    
    length_random=[]*N
    
    for i in range(N):
        x_0_square=pow(x_0,2)
        x_0_str_list=list(str(x_0_square).zfill(8))
        new_list=[int(x_0_str_list[2]),int(x_0_str_list[3]),int(x_0_str_list[4]),int(x_0_str_list[5])]
        x_1=1000*new_list[0]+new_list[1]*100+new_list[2]*10+new_list[3]
        length_random.append((x_1)/10000)
        x_0=x_1

    return(length_random)
###1(b)####
print('The mid square rng series are:\n', mid_square_rng(20,1010))
###1(c)####
print('The mid square rng series are:\n', mid_square_rng(20,6100))
###1(d)####
print('The mid square rng series are:\n', mid_square_rng(20,3792))
###Comment: All the random numbers output are the same which is 0.3792.

###1(e)####
###Comment: Because the biggest number 9999^2 is 8 digits. 


###1(f)####
###Comment: According the algorithm, some numbers will generate the loop pattern, which are not random.  


###1(g)####
###Comment: The generator will be stuck, when encountering special numbers.   



###2(a)####
def lehmer_rng(n,m,a,b,x_0):
    length_random=[]*n
    for _ in range(n):
        if (m>=1)&(0<=a)&(a<m)&(0<=b)&(b<m):
           x_1=(a*x_0+b)%m
           length_random.append((x_1)/m)
           x_0=x_1
        else:
            print('Abnormal!')
            break
        
    return(length_random)


###2(b)####
sam = ['r-', 'o-', 'g-', 'b-', 'p-','c-','k-','g:','m-','y-','r:', 'o:', 'g--', 'b:', 'p:','c:'] 
label=np.arange(1,17)   
angles=np.linspace(0, 2*np.pi, len(label), endpoint=False) 
stick=[]
fig = plt.figure(figsize=(7,7)) 
ax1 = fig.add_subplot(111, polar=True) 

ax1.set_theta_zero_location('N')
for i in np.arange(3,16,1):
 data1=lehmer_rng(16,16,i,1,1)
 ax1.set_thetagrids(angles*180/np.pi, label) 
 ax1.plot(angles,data1,sam[i])
 ax1.set_rlim(0,0.91)
 ax1.set_rlabel_position(321)
 ax1.grid(True)
 ax1.set_title("a=3~15")
 stick.append('a='+str(i))
plt.legend(stick,loc='upper left')
plt.show()

##Comment: I think when a=3, series are the largest.

angles=np.linspace(0, 2*np.pi, 15, endpoint=False) 


###2(c)####    
sam = ['r-', 'o-', 'g-', 'b-', 'p-','c-','k-','g:','m-','y-','r:', 'o:', 'g--', 'b:', 'p:','c:'] 
label=np.arange(1,17)   
angles=np.linspace(0, 2*np.pi, len(label), endpoint=False) 
stick=[]
fig = plt.figure(figsize=(7,7)) 
ax1 = fig.add_subplot(111, polar=True) 

ax1.set_theta_zero_location('N')
for i in np.arange(2,8,1):
 data1=lehmer_rng(16,16,5,i,1) 
 ax1.set_thetagrids(angles*180/np.pi, label) 
 ax1.plot(angles,data1,sam[i])
 ax1.set_rlim(0,1)
 ax1.set_rlabel_position(321)
 ax1.grid(True)
 ax1.set_title("b=2~8")
 stick.append('b='+str(i))
plt.legend(stick,loc='upper left')
plt.show()
##Comment: I think when b=6, series are the largest.


###2(d)####
lehmer_rng(20,100,21,1,6)
##Comment: The series are random.


###2(e)####
aa=lehmer_rng(5000,pow(2,11),1229,1,1)
plt.hist(aa,bins=100)
plt.show()
##Comment: From the plot, I think the numbers are uniform distribution.



###2(f)####
x=aa[0:4999]
y=aa[1:5000] 
plt.scatter(x,y)
plt.show()
##Comment: From the scatterplot, we can see that there is linear relation between Rk and Rk+1.


###2(g)####
bb=lehmer_rng(5000, 244944,1597,51749,1)
plt.hist(bb,bins=100)
plt.show()

x=bb[0:4999]
y=bb[1:5000] 
plt.scatter(x,y)
plt.show()
##Comment: From the scatterplot, we can see that the numbers are distributed randomly and pattern likes clouds. Thus, I think it is improved.

###2(h)####
bb=lehmer_rng(5000, pow(2,31),pow(2,16)+3,0,1)
plt.hist(bb,bins=100)
plt.show()

x=bb[0:4999]
y=bb[1:5000] 
plt.scatter(x,y)
plt.show()
##Comment: From the scatterplot, we can see that the numbers are distributed randomly and pattern likes clouds. Thus, I think it is improved.


###2(i)####
bb=lehmer_rng(5000, pow(2,31),pow(2,16)+3,0,1)
x=bb[0:4998:3]
y=bb[1:4999:3]
z=bb[2:5000:3]

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(x, y, z)
 
ax.set_xlabel('Rk-1 Label')
ax.set_ylabel('Rk Label')
ax.set_zlabel('Rk+1 Label')
plt.show()
