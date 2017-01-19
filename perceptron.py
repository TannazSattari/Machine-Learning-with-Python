# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap


class perceptron(object):
    
    def __init__(self,n_iter=100,eta=0.01):
        self.n_iter=n_iter
        self.eta=eta
    
    def fit(self,X,Y):
        self.w = np.zeros(1+X.shape[1])
        self.errors = [] 
        for n in range(self.n_iter):
            errors=0
            for xi,target in zip(X,Y):
                update = self.eta*(target - self.predict(xi))
                self.w[1:] += update *xi
                self.w[0] += update
                errors += int(update != 0.0)
            self.errors.append(errors)
        return self
    
    def predict(self, x):
        return np.where(self.net_Input(x) >= 0.0, 1,-1)
    
    def net_Input(self, x):
        return np.dot(x,self.w[1:])+self.w[0]

df = pd.read_csv('https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data',header=None)

def plot_decision_region(X,Y,classifier,resolution=0.02):
    
    colors = ['red','blue','lightgreen','gray','cyan']
    markers = ['s','x','o','^','v']

    cmap = ListedColormap(colors[:len(np.unique(Y))])
    
    x1_min,x1_max = X[:,0].min()-1,X[:,0].max()+1
    x2_min,x2_max = X[:,1].min()-1,X[:,1].max()+1

    xx1,xx2=np.meshgrid(np.arange(x1_min,x1_max,resolution),np.arange(x2_min,x2_max,resolution))
    
    Z = classifier.predict(np.array([xx1.ravel(),xx2.ravel()]).T)
    Z = Z.reshape(xx1.shape)
    plt.contourf(xx1,xx2,Z,alpha=0.4,cmap=cmap)
    plt.xlim(xx1.min(),xx1.max())
    plt.ylim(xx2.min(),xx2.max())
    
    for idx,cl in enumerate(np.unique(Y)):
        plt.scatter(x=X[Y == cl,0],y=X[Y==cl,1],alpha=0.8, c=cmap(idx), marker=markers[idx],label = cl)
    
    
    

Y = df.iloc[0:100,4].values
Y = np.where(Y=='Iris-versicolor',-1,1)

x = df.iloc[0:100,[0,2]].values
plt.scatter(x[:50,0],x[:50,1], color='red', marker='o',label='setosa')
plt.scatter(x[50:100,0],x[50:100,1], color='blue', marker='x',label='versicolor')

plt.xlabel('petal length')
plt.ylabel('sepal length')
plt.legend(loc = 'upper left')
plt.show()

ppn = perceptron(n_iter=10,eta=0.01)
ppn.fit(x,Y)

plt.plot(range(1,len(ppn.errors)+1),ppn.errors,marker='o')
plt.show()

plot_decision_region(x,Y,classifier=ppn)
plt.xlabel('SepalLength[cm]')
plt.ylabel('petal length[cm]')
plt.legend(loc='upper left')
plt.show()


