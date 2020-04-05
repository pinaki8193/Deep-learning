import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler, MinMaxScaler,OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, mean_squared_error,log_loss
from sklearn.datasets import load_iris
X=load_iris().data
Y=load_iris().target
Y_features=load_iris().target_names
print(Y_features)
print('for branch')
#['setosa' 'versicolor' 'virginica']
X_train,X_test,Y_train,Y_test=train_test_split(X,Y,random_state=0,stratify=Y,test_size=0.2)
print(X_train.shape,X_test.shape)
print(Y_train.shape,Y_test.shape)
x=X_train[0:10,:]
y_cnt=[]
n=50
np.random.seed(0)
out_look=np.random.randint(0,3,n)
temp=np.random.randint(0,3,n).tolist()
hum=np.random.randint(0,3,n).tolist()
wind=np.random.randint(0,3,n).tolist()
out=np.random.randint(0,2,n)
data=np.array([temp,hum,wind])
data=data.T
x_test=data[0:50,:]
y=out[0:50]
y_cnt.append(y.tolist().count(0))
y_cnt.append(y.tolist().count(1))
#print(len(x_test))
#print(y_cnt)
p=[(3,2),(5,6),(8,9)]
for i in p:
    if 2 in i:
        print(i)
seperated=dict()
for rows in range(len(x_test)):
    class_value=y[rows]
    if class_value not in seperated:
        seperated[class_value]=list()
        seperated[class_value].append(x_test[rows,:].tolist())
    else:
     seperated[class_value].append(x_test[rows,:].tolist())
#print(seperated)
prob=dict()
for label in seperated:
 #print("label{}".format(label))
 c=0
 l_col=[]
 for column in zip(*seperated[label]):
    #print("col={}".format(c))
    l=[]
    for val in column:
     #print("count of={}".format(val),list(column).count(val),round(list(column).count(val)/y_cnt[label],3))
     l.append((val,round(list(column).count(val)/y_cnt[label],3)))
    l_col.append(l)
    c+=1
    prob[label]=l_col
#print(prob)
y_pred=[]
for rows in range(x_test.shape[0]):
    #print("x{}".format(rows),x_test[rows,:],y[rows])
    max_prob=[]
    for l in prob:
        #print("label{}".format(l))
        m=[]
        for cols in range(x_test.shape[1]):
            #print(prob[l][cols],x_test[rows][cols])
            for k in prob[l][cols]:
                if x_test[rows][cols] in k:
                    #print(k)
                    m.append(k[1])
                    break
        #print(round(np.prod(m),3),)
        max_prob.append(round(np.prod(m),3))
    #print(np.argmax(max_prob,axis=0))
    y_pred.append(np.argmax(max_prob,axis=0))
print(accuracy_score(y_pred,y))          
