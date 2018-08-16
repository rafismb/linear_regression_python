import numpy as np
#training function to find the suitable parameters
def lin_reg(x,Y,m):
    x1=np.ones((np.shape(x)))
    for i in range(1,m):
         x1=np.hstack((x1,x**i))
    w=np.dot(np.linalg.inv(np.dot(x1.T,x1)),np.dot(x1.T,Y))
    return w
#testing function to predict the label: square of a number
def pred_label(xt,w,m):
    x1=1;
    for i in range(1,m):
        x1=np.hstack((x1,xt**i))
    pred=np.dot(w,x1)
    return pred
#main
train_data=np.genfromtxt('data.csv',delimiter=',')
Y=train_data[:,-1]
x=train_data[:,:-1]
m=input('enter the model order: ')
wt=lin_reg(x,Y,m)
print 'trained parameters are:\n',wt
xt=input('enter a number: ')
pred=pred_label(xt,wt,m)
print 'predicted square of the given number is:',pred
