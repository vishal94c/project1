# -*- coding: utf-8 -*-
"""
Created on Tue Apr 17 11:23:51 2018

@author: vishal
"""

import numpy as np
import pandas as pd
import pickle

colname=['Name','Info']
mylist = []

for chunk in  pd.read_csv('shuffled-full-set-hashed.csv', names=colname,chunksize=20000):
    mylist.append(chunk)

dataset = pd.concat(mylist, axis= 0)
del mylist

df=pd.DataFrame(dataset)
df=df.dropna(axis=0,how='any')


array = df.values
X = array[:,1]  #data
y = array[:,0]  #label

x_test = X
y_test = y

##https://machinelearningmastery.com/save-load-machine-learning-models-python-scikit-learn/
filename = 'finalized_model.sav'
loaded_model = pickle.load(open(filename, 'rb'))
predicted_svm = loaded_model.predict(x_test)
np.mean(predicted_svm == y_test)


from sklearn import metrics
print(metrics.classification_report(y_test, predicted_svm,target_names=y_test))

