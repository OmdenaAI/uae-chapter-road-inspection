# -*- coding: utf-8 -*-
"""
Created on Fri Nov 11 23:41:41 2022

@author: User
"""

# -*- coding: utf-8 -*-
"""
Created on Tue May 17 23:24:37 2022
dataset is prepread as 
crack -1 
groove -0
rut -2
subscident-3
@author: User
"""

import pandas as pd 

from sklearn import metrics

df=pd.read_csv("F:/internship/real time project/omden uae project/train_csv.csv")

# SPLIT DF INTO FEATURES AND TARGET
X=df.iloc[:,:-1]
y=df.iloc[:,-1]

# SPLIT DATA INTO TRAIN TEST AS 60% TRAIN AND 40 % TEST
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=1)

#logistic reg
print("Logistic Regration solution\n")
from sklearn.preprocessing import StandardScaler
sc_x = StandardScaler()
xtrain = sc_x.fit_transform(X_train)
xtest = sc_x.transform(X_test)
print (xtrain[0:10, :])
# BUILD A MODEL AND FIT TRAIN DATA 
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
classifier = LogisticRegression(random_state = 0)
classifier.fit(X_train, y_train)
# PREDICT VALUES FOR TEST DATA 
y_pred = classifier.predict(X_test)
print (y_pred)
# CONFUSSION MATRIX 
cnf_matrix = metrics.confusion_matrix(y_test, y_pred)
print (cnf_matrix)
# FIND ACCURACY 
from sklearn.metrics import accuracy_score
print ("Accuracy : ", accuracy_score(y_test, y_pred))
#classification report 
print("classification Report\n",classification_report(y_test, y_pred))


#knn 
print("KNN solution\n")
from sklearn.neighbors import KNeighborsClassifier  
classifier= KNeighborsClassifier(n_neighbors=5, metric='minkowski', p=2 ) .fit(X_train, y_train)
y_pred= classifier.predict(X_test)  
print(y_pred)
from sklearn.metrics import confusion_matrix  
print("Confusion matrix \n" ,confusion_matrix(y_test, y_pred) )
from sklearn.metrics import accuracy_score
print ("Accuracy : ", accuracy_score(y_test, y_pred))
print("classification Report\n",classification_report(y_test, y_pred))


#svm
print("SVM Solution ")
from sklearn import svm 
# SVC is (support vector classifier)
model=svm.SVC().fit(X_train,y_train)
y_pred= classifier.predict(X_test)  
print(y_pred)
from sklearn.metrics import confusion_matrix  
print("Confusion matrix \n" ,confusion_matrix(y_test, y_pred) )
from sklearn.metrics import accuracy_score
print ("Accuracy : ", accuracy_score(y_test, y_pred))
print("classification Report\n",classification_report(y_test, y_pred))


#naive base 
print ("Naive Baise solution \n") 
from sklearn.naive_bayes import GaussianNB
model = GaussianNB().fit(X_train, y_train)
y_pred= classifier.predict(X_test)  
print(y_pred)
from sklearn.metrics import confusion_matrix 
print("Confusion matrix \n" ,confusion_matrix(y_test, y_pred) )
from sklearn.metrics import accuracy_score
print ("Accuracy : ", accuracy_score(y_test, y_pred))
print("classification Report\n",classification_report(y_test, y_pred))



#decision tree
#X_train.head()
print ("Decision Tree slution\n")
from sklearn.tree import DecisionTreeClassifier
model=DecisionTreeClassifier(criterion = "gini",
            random_state = 0,max_depth=3, min_samples_leaf=5)
y_pred= classifier.predict(X_test)  
print(y_pred)
from sklearn.metrics import confusion_matrix 
print("Confusion matrix \n" ,confusion_matrix(y_test, y_pred) )
from sklearn.metrics import accuracy_score
print ("Accuracy : ", accuracy_score(y_test, y_pred))
print("classification Report\n",classification_report(y_test, y_pred))
