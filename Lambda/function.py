# -*- coding: utf-8 -*-
"""
Created on Tue Apr 17 21:39:10 2018

@author: vishal
"""

import boto3
import os
import ctypes
import uuid
import sklearn
import pickle
import numpy as np
import pandas as pd
from sklearn import metrics

for d, _, files in os.walk('lib'):
    for f in files:
        if f.endswith('.a'):
            continue
        ctypes.cdll.LoadLibrary(os.path.join(d, f))

s3_client = boto3.client('s3')

def handler(event, context):

    #Info
    #sepal_length = float(event.get('Iris')['sepal_length'])
    #sepal_width = float(event.get('Iris')['sepal_width'])
    #petal_length = float(event.get('Iris')['petal_length'])
    #petal_width = float(event.get('Iris')['petal_width'])

    bucket = 'hevay'
    key = 'shuffled-full-set-hashed.csv'
    fileA = '/tmp/{}{}'.format(uuid.uuid4(), key)
    s3_client.download_file(bucket, key, fileA)

    colname=['Name','Info']
    mylist = []

    for chunk in  pd.read_csv(fileA, names=colname,chunksize=20000):
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
#load model
    bucket = 'hevay'
    key = 'finalized_model.sav'
    download_path = '/tmp/{}{}'.format(uuid.uuid4(), key)
    s3_client.download_file(bucket, key, download_path)
    

#filename = 'finalized_model.sav'
    loaded_model = pickle.load(open(download_path, 'rb'))
    predicted_svm = loaded_model.predict(x_test)
    np.mean(predicted_svm == y_test)
    return metrics.classification_report(y_test, predicted_svm,target_names=y_test)