import numpy as np
#call option
def call_strategy(s00,K,r,u,d,n):
    p_star = 0.6
    q_star = 1-p_star

    #create the tree, k is the state
    tree={}
    
    for k in range(n+1):
        tree[k] = [0]*(k+1)

    for k in range(n+1):
        for i in range(len(tree[k])):
            tree[k][i] = (pow(u,k-i))*(pow(d,i))*s00
#judge
    for l in range(n+1):
        tree[n][l]=max(0,(tree[n][l]-K))

    for m in range(n,0,-1):
        for s,a in zip(range(1,len(tree[m])),range(len(tree[m-1]))):
            tree[m-1][a]=max((((tree[m][s-1])*p_star)+(q_star*tree[m][s]))/(1+r),max(0,tree[m-1][a]-K))

    return(tree)
a=call_strategy(400,375,0.07,1.25,0.8,100)[0]
print("Thus, I think that when the call option is mature, people should exercise and the fair price is:", a)
##put option


def put_strategy(s00,K,r,u,d,n):
    p_star = 0.6
    q_star = 1-p_star

    #create the tree, k is the state
    tree={}
    
    for k in range(n+1):
        tree[k] = [0]*(k+1)

    for k in range(n+1):
        for i in range(len(tree[k])):
            tree[k][i] = (pow(u,k-i))*(pow(d,i))*s00
#judge
    for l in range(n+1):
        tree[n][l]=max(0,-(tree[n][l]-K))

    for m in range(n,0,-1):
        for s,a in zip(range(1,len(tree[m])),range(len(tree[m-1]))):
            if (((tree[m][s-1])*p_star)+(q_star*tree[m][s]))/(1+r) > (-(tree[m-1][a]-K)):
                return(m-1)

print("Thus, I think that when the put option is n=:",put_strategy(400,375,0.07,1.25,0.8,100),"we should exercise.")
           
def put_strategy1(s00,K,r,u,d,n):
    p_star = 0.6
    q_star = 1-p_star

    #create the tree, k is the state
    tree={}
    
    for k in range(n+1):
        tree[k] = [0]*(k+1)

    for k in range(n+1):
        for i in range(len(tree[k])):
            tree[k][i] = (pow(u,k-i))*(pow(d,i))*s00
#judge
    for l in range(n+1):
        tree[n][l]=max(0,-(tree[n][l]-K))

    for m in range(n,0,-1):
        for s,a in zip(range(1,len(tree[m])),range(len(tree[m-1]))):
           tree[m-1][a]=max((((tree[m][s-1])*p_star)+(q_star*tree[m][s]))/(1+r),max(0,-(tree[m-1][a]-K)))

    return(tree)
print("Thus, when we exercise, the fair price is",put_strategy1(400,375,0.07,1.25,0.8,99)[0])