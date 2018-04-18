# -*- coding: utf-8 -*-
"""
Created on Wed Apr 11 16:47:28 2018

@author: vishal
"""

#import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import model_selection
from sklearn import datasets
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import SGDClassifier
from sklearn.pipeline import Pipeline
import pickle




#https://stackoverflow.com/questions/41303246/error-tokenizing-data-c-error-out-of-memory-pandas-python-large-file-csv
colname=['Name','Info']
mylist = []

for chunk in  pd.read_csv('shuffled-full-set-hashed.csv', names=colname,chunksize=20000):
    mylist.append(chunk)

dataset = pd.concat(mylist, axis= 0)
del mylist

#remove all the nan values
df=pd.DataFrame(dataset)
df=df.dropna(axis=0,how='any')

array = df.values
X = array[:,1]  #data
y = array[:,0]  #label
validation_size = 0.20
seed = 7

##splitting the data into list format
#for i in range(0,len(X)) :
#    temp=X[i]
#    temp=temp.split()
#    X[i]=temp

# Split-out validation dataset
x_train, x_test, y_train, y_test = model_selection.train_test_split(X, y, test_size=validation_size, random_state=seed)

## encoding the data
#
#cv = CountVectorizer()
#X_count = cv.fit_transform(x_train)
#print (X_count.shape)
#
###with tfidf accruacy is 50% which is much less
#
##from sklearn.feature_extraction.text import TfidfTransformer
##tfidf = TfidfTransformer()
##X_train = tfidf.fit_transform(X_count)
##print (X_train.shape)
#
##NB training
#from sklearn.naive_bayes import MultinomialNB
#clf = MultinomialNB()
#model=clf.fit(X_count,y_train)
#
#X_n = cv.transform(x_test)
#predicted = clf.predict(X_n)
#
#print (np.mean(predicted == y_test))

###https://towardsdatascience.com/machine-learning-nlp-text-classification-using-scikit-learn-python-and-nltk-c52b92a7c73a

##svm has higher accuracy 85%
####svm encoding training and testing using pipeline
text_clf_svm = Pipeline([('vect', CountVectorizer()),('clf-svm', SGDClassifier(loss='hinge', penalty='l2',alpha=1e-3, n_iter=5, random_state=42)),])
text_clf_svm=text_clf_svm.fit(x_train, y_train)

filename = 'finalized_model.sav'
pickle.dump(text_clf_svm, open(filename, 'wb'))


loaded_model = pickle.load(open(filename, 'rb'))
predicted_svm = loaded_model.predict(x_test)
np.mean(predicted_svm == y_test)


from sklearn import metrics
print(metrics.classification_report(y_test, predicted_svm,target_names=y_test))



