import numpy as np
import pandas as pd
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn import metrics
from sklearn.metrics import accuracy_score, classification_report

iris = pd.read_csv("Dataset_Iris/iris.csv")



train, test = train_test_split(iris, test_size = 0.2)           # in this our main data is split into train and test

print(train.shape)
print(test.shape)

train_X = train[['sepal_length','sepal_width','petal_length','petal_width']]             # taking the training data features
train_y=train.species# output of our training data
test_X= test[['sepal_length','sepal_width','petal_length','petal_width']]                 # taking test data features
test_y =test.species   #output value of test data




model = svm.SVC() #select the algorithm
model.fit(train_X,train_y) # we train the algorithm with the training data and the training output
prediction=model.predict(test_X) #now we pass the testing data to the trained algorithm
print('The accuracy of the SVM is:',metrics.accuracy_score(prediction,test_y)) 

