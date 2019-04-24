

# 1.1 - import packages and data

import pandas as pd
import matplotlib.pyplot as plot
import seaborn as sb
import numpy as np
from sklearn.model_selection import KFold, cross_val_score, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.metrics import roc_curve, auc
import warnings

tr=pd.read_csv("https://raw.githubusercontent.com/lzlzh/CS6375_project/master/input/train.csv")
te=pd.read_csv("https://raw.githubusercontent.com/lzlzh/CS6375_project/master/input/test.csv")
ts2=pd.read_csv("https://raw.githubusercontent.com/lzlzh/CS6375_project/master/input/test.csv")
data=pd.concat([tr, te], sort=False)
len_train=tr.shape[0]


# 1.2 - Preprocessing: fix the the values that are missing

data.isnull().sum()[data.isnull().sum()>0]

# Age

tr.Age=tr.Age.fillna(tr.Age.mean())
te.Age=te.Age.fillna(tr.Age.mean())

# Fare

tr.Fare=tr.Fare.fillna(tr.Fare.mean())
te.Fare=te.Fare.fillna(tr.Fare.mean())

# Cabin

tr.Cabin=tr.Cabin.fillna("unknow")
te.Cabin=te.Cabin.fillna("unknow")

# Embarked

tr.Embarked=tr.Embarked.fillna(method='bfill',inplace=True)
te.Embarked=te.Embarked.fillna(method='bfill',inplace=True)



# 1.3 - Prepare the data for models

#drop the columns that whil not be used
tr.drop(['PassengerId','Name','Ticket','SibSp','Parch','Ticket','Cabin'],axis=1,inplace=True)
te.drop(['PassengerId','Name','Ticket','SibSp','Parch','Ticket','Cabin'],axis=1,inplace=True)

data=pd.concat([tr, te], sort=False)

data=pd.get_dummies(data)

tr=data[:len_train]
te=data[len_train:]

tr.Survived=tr.Survived.astype('int')
tr.Survived.dtype

xtr=tr.drop("Survived",axis=1)
ytr=tr['Survived']
xte=te.drop("Survived", axis=1)



# 2 - Nested Cross Validation

cv_score = KFold(n_splits=5, shuffle=True, random_state=1)

# Random Forest

RF=RandomForestClassifier(random_state=1)
PRF=[{'n_estimators':[10,100],'max_depth':[3,6],'criterion':['gini','entropy']}]
GSRF=GridSearchCV(estimator=RF, param_grid=PRF, scoring='accuracy',cv=cv_score)
scores_rf=cross_val_score(GSRF,xtr,ytr,scoring='accuracy',cv=cv_score)

np.mean(scores_rf)

# SVM

svc=make_pipeline(StandardScaler(),SVC(random_state=1))
r=[0.0001,0.001,0.1,1,10,50,100]
PSVM=[{'svc__C':r, 'svc__kernel':['rbf']},
      {'svc__C':r, 'svc__gamma':r, 'svc__kernel':['rbf']}]
GSSVM=GridSearchCV(estimator=svc, param_grid=PSVM, scoring='accuracy', cv=cv_score)
scores_svm=cross_val_score(GSSVM, xtr.astype(float), ytr,scoring='accuracy', cv=cv_score)

np.mean(scores_svm)



# 3 - Submission

# model=GSSVM.fit(xtr, ytr)
# pred=model.predict(xte)

# output=pd.DataFrame({'PassengerId':ts2['PassengerId'],'Survived':pred})
# output.to_csv('submission.csv', index=False)

