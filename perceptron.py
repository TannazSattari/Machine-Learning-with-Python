# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def perceptron():
    
    def _init(self,n_iter=100,eta=0.01):
        self.n_iter=n_iter
        self.eta=eta
    
    def fit(self,X,Y):
        self.w = np.zeros(1+X.shape[1])
        self.errors = [] 
        for n in range(self.n_iter):
            for xi,target in zip(X,Y):
                update = self.eta*(target - self.predict(xi))
                self.w[1:] = update *xi
                self.w[0] = update
                errors = int(update != 0.0)
        self.errors.append(errors)
        return self
    
    def predict(self, x):
        return np.where(self.net_Input(x) >= 0.0, 1,-1)
    
    def net_Input(self, x):
        return np.dot(x,self.w[1:])+w[0]

df = pd.read_csv('https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data',header=None)

Y = df.iloc[0:100,4].values
Y = np.where(Y=='Iris-versicolor',-1,1)

x = df.iloc[0:100,[0,2]]
plt.scatter(x[:50,0],x[:50,1], color='red', marker='o',label='setosa')
plt.scatter(x[50:100,0],x[50:100,1], color='blue', marker='x',label='versicolor')

plt.xlabel('petal length')
plt.ylabel('sepal length')
plt.legend(loc = 'upper left')
plt.show()