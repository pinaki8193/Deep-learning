import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler, MinMaxScaler,OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, mean_squared_error,log_loss
from sklearn.datasets import load_iris
'''
iris=load_iris()
data=iris.data
labels=iris.target
print(data.shape,labels.shape)
print(iris.target_names)
#print(labels)
'''
r_train=pd.read_csv('C:\\Users\\Pinaki8193\\Downloads\\train.csv')
r_test=pd.read_csv('C:\\Users\\Pinaki8193\\Downloads\\test.csv')
data_test=pd.DataFrame(r_test)
X_test_new=data_test.head(1000).values
data=pd.DataFrame(r_train)
print(type(data))
#data_rd=data[:,:100]
###X_test=pd.DataFrame(r_test)
Y=data.head(1000)['label'].values
X=data.head(1000).drop('label',axis=1).values
X_train,X_test,Y_train,Y_test=train_test_split(X,Y,random_state=0,stratify=Y,test_size=0.2)
print(X_train.shape,X_test.shape)
print(Y_train.shape,Y_test.shape)
l=[]
y_pred=[]
def distance(x_test,K):
    l=[]
    for i in range(X_train.shape[0]):
     s=0
     for j,t in zip(X_train[i],x_test):
      s+=(t-j)**2
     #print(s**0.5,Y_train[i])
     l.append((s**0.5,Y_train[i]))
    l.sort()
    #print(l)
    l_min=[]
    for k in range(K):
        l_min.append(l[k])
    #print(l_min)
    #print()
    l_labels=[]
    for i in l_min:
        #print(i[1])
        l_labels.append(i[1])
    #print("majority",most_frequent(l_labels))
    y_pred.append(most_frequent(l_labels))
def most_frequent(List): 
    return max(set(List), key = List.count) 
for i in range(X_test.shape[0]):
 distance(X_test[i],3)#change here for different k values
#print(y_pred)
print(Y_test)
print(accuracy_score(y_pred,Y_test))

         
