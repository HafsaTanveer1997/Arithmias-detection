# -*- coding: utf-8 -*-
"""
Created on Mon Nov  9 12:27:54 2020

@author: Sheeza
"""

# -*- coding: utf-8 -*-
"""
Created on Tue Oct 20 09:44:43 2020

@author: Sheeza
"""

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from sklearn.preprocessing import MinMaxScaler
from sklearn.feature_selection import RFE
from sklearn.feature_selection import RFECV
from matplotlib import pyplot as plt
from sklearn.metrics import r2_score, mean_squared_error
import joblib
import warnings
warnings.filterwarnings('ignore') 

data = pd.read_csv('cardiac_arrhythmia.csv')
output = pd.DataFrame(index=None, columns=['model','train_Rsquare', 'test_Rsquare', 'train_MSE','test_MSE'])
data.describe()
import numpy as np
data['J'] = data['J'].replace('?',np.NaN)
data['HR'] = data['HR'].replace('?',np.NaN)
data['P'] = data['P'].replace('?',np.NaN)
data['T'] = data['T'].replace('?',np.NaN)
data['QRST'] = data['QRST'].replace('?',np.NaN)
Data_Y = data.cardiac_arrhythmia.values.ravel()
Data_X=data.drop('cardiac_arrhythmia', 1)


np.unique(Data_Y, return_counts=True)
Data_X=Data_X.drop(columns=['J'])
from sklearn.impute import SimpleImputer
z=SimpleImputer(missing_values=np.nan, strategy='mean', verbose=1).fit_transform(Data_X)
Data_X = pd.DataFrame(data=z,columns=Data_X.columns.values)
Data_X.isnull().sum()
data_train_x, data_test_x, data_train_y, data_test_y = train_test_split(Data_X, Data_Y, random_state=2)
# Scaling the data (MIN MAX Scaling)
print('Shape of train {}, shape of test {}'.format(data_train_x.shape, data_test_x.shape))


#MinMax
#MinMax = MinMaxScaler(feature_range= (0,1))
#data_train_x= MinMax.fit_transform(data_train_x)
#data_test_x = MinMax.transform(data_test_x)


from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import GridSearchCV



param_grid={'weights':['distance', 'uniform'], 'n_neighbors':range(1,100)}
grid_search = GridSearchCV( KNeighborsClassifier(),param_grid, cv = 10,iid='deprecated',return_train_score=False)
grid_search.fit(data_train_x, data_train_y)


train_Rsquare = grid_search.score(data_train_x, data_train_y)
test_Rsquare = grid_search.score(data_test_x, data_test_y)
train_MSE = mean_squared_error(data_train_y, grid_search.predict(data_train_x))
test_MSE = mean_squared_error(data_test_y, grid_search.predict(data_test_x))

output = output.append(pd.Series({'model':'KNN Classifier','train_Rsquare':train_Rsquare, 'test_Rsquare':test_Rsquare, 'train_MSE':train_MSE,'test_MSE':test_MSE}),ignore_index=True )


pd.DataFrame(grid_search.cv_results_)
print(grid_search.best_estimator_)

from sklearn.metrics import classification_report
knn = KNeighborsClassifier(algorithm='auto', leaf_size=30, metric='minkowski',
           metric_params=None, n_jobs=1, n_neighbors=6, p=2,
           weights='distance')
knn.fit(data_train_x, data_train_y)
pred = knn.predict(data_test_x)
print(classification_report(data_test_y,pred))


# Logistic Regression

from sklearn.linear_model import LogisticRegression
  
param_grid = {'C': [0.001, 0.01, 0.1, 1, 10, 100, 1000] }

grid_search_log = GridSearchCV(LogisticRegression(penalty='l2'), param_grid, cv=5)
grid_search_log.fit(data_train_x, data_train_y)

train_Rsquare = grid_search_log.score(data_train_x, data_train_y)
test_Rsquare = grid_search_log.score(data_test_x, data_test_y)
train_MSE = mean_squared_error(data_train_y, grid_search_log.predict(data_train_x))
test_MSE = mean_squared_error(data_test_y, grid_search_log.predict(data_test_x))
output = output.append(pd.Series({'model':'Logistic Regression','train_Rsquare':train_Rsquare, 'test_Rsquare':test_Rsquare, 'train_MSE':train_MSE,'test_MSE':test_MSE}),ignore_index=True )

pd.DataFrame(grid_search_log.cv_results_)
print(grid_search_log.best_estimator_)


log = LogisticRegression(C=1, class_weight=None, dual=False, fit_intercept=True,
          intercept_scaling=1, max_iter=100, multi_class='ovr', n_jobs=1,
          penalty='l2', random_state=None, solver='liblinear', tol=0.0001,
          verbose=0, warm_start=False)
log.fit(data_train_x, data_train_y)
pred = log.predict(data_test_x)
print(classification_report(data_test_y,pred))

# Linear Supprt Vector Machine
from sklearn.svm import LinearSVC

param_grid = {'C': [0.001, 0.01, 0.1, 0.5, 1, 10, 50, 100, 1000], 'max_iter':[1000,10000] }

grid_search_SVC = GridSearchCV(LinearSVC(random_state=0), param_grid, cv=5)
grid_search_SVC.fit(data_train_x, data_train_y)
train_Rsquare = grid_search_SVC.score(data_train_x, data_train_y)
test_Rsquare = grid_search_SVC.score(data_test_x, data_test_y)
train_MSE = mean_squared_error(data_train_y, grid_search_SVC.predict(data_train_x))
test_MSE = mean_squared_error(data_test_y, grid_search_SVC.predict(data_test_x))
output = output.append(pd.Series({'model':'Linear SVC','train_Rsquare':train_Rsquare, 'test_Rsquare':test_Rsquare, 'train_MSE':train_MSE,'test_MSE':test_MSE}),ignore_index=True )

pd.DataFrame(grid_search_SVC.cv_results_)
print(grid_search_SVC.best_estimator_)

linearsvc = LinearSVC(C=0.1, class_weight=None, dual=True, fit_intercept=True,
     intercept_scaling=1, loss='squared_hinge', max_iter=1000,
     multi_class='ovr', penalty='l2', random_state=0, tol=0.0001,
     verbose=0)
linearsvc.fit(data_train_x, data_train_y)
pred = linearsvc.predict(data_test_x)
print(classification_report(data_test_y,pred))



# Kerenilzed Support Vector Machine


from sklearn.svm import SVC

param_grid = {'C':[0.001, 0.01, 0.1, 0.5, 1, 10, 50, 100, 1000], 'gamma':[0.001, 0.01, 0.1, 0.5, 1, 10]}

grid_search_KSVC = GridSearchCV(SVC(kernel = 'rbf'), param_grid, cv=5)
grid_search_KSVC.fit(data_train_x, data_train_y)

train_Rsquare = grid_search_KSVC.score(data_train_x, data_train_y)
test_Rsquare = grid_search_KSVC.score(data_test_x, data_test_y)
train_MSE = mean_squared_error(data_train_y, grid_search_KSVC.predict(data_train_x))
test_MSE = mean_squared_error(data_test_y, grid_search_KSVC.predict(data_test_x))
output = output.append(pd.Series({'model':'Kernel SVC','train_Rsquare':train_Rsquare, 'test_Rsquare':test_Rsquare, 'train_MSE':train_MSE,'test_MSE':test_MSE}),ignore_index=True )

pd.DataFrame(grid_search_KSVC.cv_results_)
print(grid_search_KSVC.best_estimator_)

svc = SVC(C=50, cache_size=200, class_weight=None, coef0=0.0,
  decision_function_shape='ovr', degree=3, gamma=0.01, kernel='rbf',
  max_iter=-1, probability=False, random_state=None, shrinking=True,
  tol=0.001, verbose=False)
svc.fit(data_train_x, data_train_y)
pred = svc.predict(data_test_x)
print(classification_report(data_test_y,pred))

# Decision Tree

from sklearn.tree import DecisionTreeClassifier

param_grid = {'max_features':[None,'auto', 'log2'], 'max_depth':[5,10,15,20,50]}

grid_search_DT = GridSearchCV(DecisionTreeClassifier(random_state = 10), param_grid, cv=5)
grid_search_DT.fit(data_train_x, data_train_y)


train_Rsquare = grid_search_DT.score(data_train_x, data_train_y)
test_Rsquare = grid_search_DT.score(data_test_x, data_test_y)
train_MSE = mean_squared_error(data_train_y, grid_search_DT.predict(data_train_x))
test_MSE = mean_squared_error(data_test_y, grid_search_DT.predict(data_test_x))
output = output.append(pd.Series({'model':'Decision Tree Classifier','train_Rsquare':train_Rsquare, 'test_Rsquare':test_Rsquare, 'train_MSE':train_MSE,'test_MSE':test_MSE}),ignore_index=True )
output

pd.DataFrame(grid_search_DT.cv_results_)
print(grid_search_DT.best_estimator_)

dt = DecisionTreeClassifier(class_weight=None, criterion='gini', max_depth=10,
            max_features=None, max_leaf_nodes=None,
            min_impurity_decrease=0.0, min_impurity_split=None,
            min_samples_leaf=1, min_samples_split=2,
            min_weight_fraction_leaf=0.0, presort=False, random_state=10,
            splitter='best')
dt.fit(data_train_x, data_train_y)
pred = dt.predict(data_test_x)
print(classification_report(data_test_y,pred))



# Random Forest

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import RandomizedSearchCV
from scipy.stats import randint as sp_randint

#Tuning ridge on new dataset
param_grid = {"max_depth": [3, 5],
              "max_features": sp_randint(1, 40),
              "min_samples_split": sp_randint(2, 30),
              "min_samples_leaf": sp_randint(1, 20),
              "bootstrap": [True, False]}
grid_search_RF = RandomizedSearchCV(RandomForestClassifier(n_estimators=1000), param_distributions=param_grid,
                                   n_iter=30, random_state=0,n_jobs=-1)
grid_search_RF.fit(data_train_x, data_train_y)


train_Rsquare = grid_search_RF.score(data_train_x, data_train_y)
test_Rsquare = grid_search_RF.score(data_test_x, data_test_y)
train_MSE = mean_squared_error(data_train_y, grid_search_RF.predict(data_train_x))
test_MSE = mean_squared_error(data_test_y, grid_search_RF.predict(data_test_x))
output = output.append(pd.Series({'model':'Random Forest Classifier','train_Rsquare':train_Rsquare, 'test_Rsquare':test_Rsquare, 'train_MSE':train_MSE,'test_MSE':test_MSE}),ignore_index=True )
output

pd.DataFrame(grid_search_RF.cv_results_)
print(grid_search_RF.best_estimator_)

rf = RandomForestClassifier(bootstrap=False, class_weight=None, criterion='gini',
            max_depth=5, max_features=36, max_leaf_nodes=None,
            min_impurity_decrease=0.0, min_impurity_split=None,
            min_samples_leaf=7, min_samples_split=25,
            min_weight_fraction_leaf=0.0, n_estimators=1000, n_jobs=1,
            oob_score=False, random_state=None, verbose=0,
            warm_start=False)
rf.fit(data_train_x, data_train_y)
pred = rf.predict(data_test_x)
print(classification_report(data_test_y,pred))

###########bagging#############################################

#   KNN with bagging

from sklearn.ensemble import BaggingClassifier

KNN_bagging = BaggingClassifier(knn, n_estimators = 100, bootstrap = True)
KNN_bagging.fit(data_train_x,data_train_y)
pred = KNN_bagging.predict(data_test_x)
print(classification_report(data_test_y,pred))


# Logistic regression with bagging

log_bagging = BaggingClassifier(log , n_estimators = 100, max_features = 200 ,bootstrap = True, oob_score = True)
log_bagging.fit(data_train_x, data_train_y)
pred = log_bagging.predict(data_test_x)
print(classification_report(data_test_y,pred))


# Linear SVC with bagging

linearsvc_bagging = BaggingClassifier(linearsvc , n_estimators = 100, max_features = 200 ,
                                      bootstrap = True, oob_score = True)
linearsvc_bagging.fit(data_train_x, data_train_y)
pred = linearsvc_bagging.predict(data_test_x)
print(classification_report(data_test_y,pred))


# Kernalized SVC with bagging

svc_bagging = BaggingClassifier(svc , n_estimators = 100, max_features = 200 ,
                                      bootstrap = True, oob_score = True)
svc_bagging.fit(data_train_x, data_train_y)
pred = svc_bagging.predict(data_test_x)
print(classification_report(data_test_y,pred))

# Decision Tree with bagging

dt_bagging = BaggingClassifier(dt , n_estimators = 100, max_features = 200 ,
                                      bootstrap = True, oob_score = True)
dt_bagging.fit(data_train_x, data_train_y)
pred = dt_bagging.predict(data_test_x)
print(classification_report(data_test_y,pred))

# Random forest with bagging

rf_bagging = BaggingClassifier(rf , n_estimators = 100, max_features = 200 ,
                                      bootstrap = True, oob_score = True)
rf_bagging.fit(data_train_x, data_train_y)
pred = rf_bagging.predict(data_test_x)
print(classification_report(data_test_y,pred))


# Boosting

# Ada boosting with Logistic regression

from sklearn.ensemble import AdaBoostClassifier
param_grid = {'learning_rate':[0.0001,0.001,0.01,0.1,1]}
adaboost_log = GridSearchCV(AdaBoostClassifier(base_estimator = log,random_state = 0), param_grid, cv=5,return_train_score=True)
adaboost_log.fit(data_train_x, data_train_y)

pred = adaboost_log.predict(data_test_x)
print(classification_report(data_test_y,pred))


# Ada boosting with Linear SVM


from sklearn.ensemble import GradientBoostingClassifier

from sklearn.ensemble import AdaBoostClassifier
param_grid = {'learning_rate':[0.0001,0.001,0.01,0.1,1]}
adaboost_svc = GridSearchCV(AdaBoostClassifier(base_estimator = linearsvc,random_state = 0, algorithm='SAMME'),
                            param_grid, cv=5,return_train_score=True)
adaboost_svc.fit(data_train_x, data_train_y)

pred = adaboost_svc.predict(data_test_x)
print(classification_report(data_test_y,pred))


# Ada boosting with Kernalized SVM

from sklearn.ensemble import AdaBoostClassifier
param_grid = {'learning_rate':[0.0001,0.001,0.01,0.1,1]}
adaboost_ksvc = GridSearchCV(AdaBoostClassifier(base_estimator = svc,random_state = 0, algorithm='SAMME'),
                            param_grid, cv=5,return_train_score=True)
adaboost_ksvc.fit(data_train_x, data_train_y)

pred = adaboost_ksvc.predict(data_test_x)
print(classification_report(data_test_y,pred))


# Ada Boosting with Decision Tree


from sklearn.ensemble import AdaBoostClassifier
param_grid = {'learning_rate':[0.0001,0.001,0.01,0.1,1]}
adaboost_dt = GridSearchCV(AdaBoostClassifier(base_estimator = dt,random_state = 0),
                            param_grid, cv=5,return_train_score=True)
adaboost_dt.fit(data_train_x, data_train_y)

pred = adaboost_dt.predict(data_test_x)
print(classification_report(data_test_y,pred))


# Ada Boosting with Random Forest
model = GradientBoostingClassifier(random_state=10, n_estimators= 500)

param_grid = {'max_features':['auto', 'log2'], 'learning_rate' : [0.01,0.1], 'max_depth':[5,10,15,30,50]}

grid_search_gb = GridSearchCV(model, param_grid, cv=5)
grid_search_gb.fit(data_train_x, data_train_y)
pred = grid_search_gb.predict(data_test_x)
print(classification_report(data_test_y,pred))



# # 4. PCA

from sklearn.decomposition import PCA

pca = PCA(0.95)
pca.fit(Data_X)
Data_X_PCA = pca.transform(Data_X)

data_train_x_pca, data_test_x_pca, data_train_y_pca, data_test_y_pca = train_test_split(Data_X_PCA, Data_Y, random_state = 100)


#Checking results of 2 scalars

from sklearn.preprocessing import MinMaxScaler

#MinMax
MinMax = MinMaxScaler(feature_range= (0,1))
data_train_x_pca = MinMax.fit_transform(data_train_x_pca)
data_test_x_pca = MinMax.transform(data_test_x_pca)


# # KNN with PCA

from sklearn.neighbors import KNeighborsClassifier

param_grid = {'weights':['distance', 'uniform'], 'n_neighbors':range(3,100)}

grid_search_knn = GridSearchCV(KNeighborsClassifier(), param_grid, cv=5,return_train_score=True)
grid_search_knn.fit(data_train_x_pca, data_train_y_pca)
pd.DataFrame(grid_search_knn.cv_results_)
print(grid_search_knn.best_score_)


print(grid_search_knn.best_estimator_)

knn_pca = KNeighborsClassifier(algorithm='auto', leaf_size=30, metric='minkowski',
           metric_params=None, n_jobs=1, n_neighbors=5, p=2,
           weights='distance')
knn_pca.fit(data_train_x_pca, data_train_y_pca)
pred = knn_pca.predict(data_test_x_pca)
print(classification_report(data_test_y_pca,pred))

# # Logistic with PCA

from sklearn.linear_model import LogisticRegression

param_grid = {'C': range(1,100) }

grid_search_log = GridSearchCV(LogisticRegression(penalty='l1'), param_grid, cv=5,return_train_score=True)
grid_search_log.fit(data_train_x_pca, data_train_y_pca)
pd.DataFrame(grid_search_log.cv_results_)
print(grid_search_log.best_estimator_)

print(grid_search_log.best_estimator_)

logistic_pca = LogisticRegression(C=4, class_weight=None, dual=False, fit_intercept=True,
          intercept_scaling=1, max_iter=100, multi_class='ovr', n_jobs=1,
          penalty='l1', random_state=None, solver='liblinear', tol=0.0001,
          verbose=0, warm_start=False)
logistic_pca.fit(data_train_x_pca, data_train_y_pca)
pred = logistic_pca.predict(data_test_x_pca)
print(classification_report(data_test_y_pca,pred))

# # Linear SVM with PCA

from sklearn.svm import LinearSVC

param_grid = {'C': [0.0001,0.001,0.01,0.1,2,3,4,5,6,7,8,9], 'max_iter':[100,1000,10000] }

grid_search_linearsvc = GridSearchCV(LinearSVC(random_state=0), param_grid, cv=5,return_train_score=True)
grid_search_linearsvc.fit(data_train_x_pca, data_train_y_pca)
pd.DataFrame(grid_search_linearsvc.cv_results_)
print(grid_search_linearsvc.best_estimator_)

linearsvc = LinearSVC(C=2, class_weight=None, dual=True, fit_intercept=True,
     intercept_scaling=1, loss='squared_hinge', max_iter=100,
     multi_class='ovr', penalty='l2', random_state=0, tol=0.0001,
     verbose=0)
linearsvc.fit(data_train_x_pca, data_train_y_pca)
pred = linearsvc.predict(data_test_x_pca)
print(classification_report(data_test_y_pca,pred))


# # Kernalized SVM with PCA

from sklearn.svm import SVC

param_grid = {'C':[0.001, 0.01, 0.1, 0.5, 1, 10, 50, 100, 1000], 'gamma':[0.001, 0.01, 0.1, 0.5, 1, 10]}

grid_search_svc = GridSearchCV(SVC(kernel = 'rbf'), param_grid, cv=5,return_train_score=True)
grid_search_svc.fit(data_train_x_pca, data_train_y_pca)
pd.DataFrame(grid_search_svc.cv_results_)
print(grid_search_svc.best_estimator_)

svc = SVC(C=10, cache_size=200, class_weight=None, coef0=0.0,
  decision_function_shape='ovr', degree=3, gamma=0.1, kernel='rbf',
  max_iter=-1, probability=False, random_state=None, shrinking=True,
  tol=0.001, verbose=False)
svc.fit(data_train_x_pca, data_train_y_pca)
pred = svc.predict(data_test_x_pca)
print(classification_report(data_test_y_pca,pred))


# # Decision tree with PCA

from sklearn.tree import DecisionTreeClassifier

param_grid = {'max_features':['auto', 'log2'], 'max_depth':[3,5,7,9,10,11,13,15,20,50]}

grid_search_dt = GridSearchCV(DecisionTreeClassifier(random_state = 0), param_grid, cv=5,return_train_score=True)
grid_search_dt.fit(data_train_x_pca, data_train_y_pca)
pd.DataFrame(grid_search_dt.cv_results_)
print(grid_search_dt.best_estimator_)


dt_pca = DecisionTreeClassifier(class_weight=None, criterion='gini', max_depth=3,
            max_features='auto', max_leaf_nodes=None,
            min_impurity_decrease=0.0, min_impurity_split=None,
            min_samples_leaf=1, min_samples_split=2,
            min_weight_fraction_leaf=0.0, presort=False, random_state=0,
            splitter='best')
dt_pca.fit(data_train_x_pca, data_train_y_pca)
pred = dt_pca.predict(data_test_x_pca)
print(classification_report(data_test_y_pca,pred))


# # Random Forest with PCA

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import RandomizedSearchCV
from scipy.stats import randint as sp_randint

#Tuning ridge on new dataset
param_grid = {"max_depth": sp_randint(3, 9),
              "max_features": sp_randint(1, 10),
              "min_samples_split": sp_randint(2, 20),
              "min_samples_leaf": sp_randint(1, 20),
              "bootstrap": [True, False]}
random_search_rf = RandomizedSearchCV(RandomForestClassifier(n_estimators=1000), param_distributions=param_grid,
                                   n_iter=30, random_state=0,n_jobs=-1, cv= 5,return_train_score=True)
random_search_rf.fit(data_train_x_pca, data_train_y_pca)
pd.DataFrame(random_search_rf.cv_results_)
print(random_search_rf.best_estimator_)

rf_pca = RandomForestClassifier(bootstrap=False, class_weight=None, criterion='gini',
            max_depth=5, max_features=9, max_leaf_nodes=None,
            min_impurity_decrease=0.0, min_impurity_split=None,
            min_samples_leaf=2, min_samples_split=3,
            min_weight_fraction_leaf=0.0, n_estimators=1000, n_jobs=1,
            oob_score=False, random_state=None, verbose=0,
            warm_start=False)
rf_pca.fit(data_train_x_pca, data_train_y_pca)
pred = rf_pca.predict(data_test_x_pca)
print(classification_report(data_test_y_pca,pred))


# # Gradient boosting with PCA

param_grid = {"max_depth": sp_randint(5,30),
              "max_features": sp_randint(1, 35),
              "min_samples_split": sp_randint(2, 40),
              "min_samples_leaf": sp_randint(1, 30),
             'learning_rate': [0.01,0.001,0.1]}
random_search_gb = RandomizedSearchCV(GradientBoostingClassifier(n_estimators=1000), param_distributions=param_grid,
                                   n_iter=30, random_state=0,n_jobs=-1,cv= 5,return_train_score=True)
random_search_gb.fit(data_train_x_pca, data_train_y_pca)
pd.DataFrame(random_search_gb.cv_results_)
print(random_search_gb.best_estimator_)

gbc_pca = GradientBoostingClassifier(criterion='friedman_mse', init=None,
              learning_rate=0.01, loss='deviance', max_depth=9,
              max_features=6, max_leaf_nodes=None,
              min_impurity_decrease=0.0, min_impurity_split=None,
              min_samples_leaf=7, min_samples_split=19,
              min_weight_fraction_leaf=0.0, n_estimators=1000,
              presort='auto', random_state=None, subsample=1.0, verbose=0,
              warm_start=False)
gbc_pca.fit(data_train_x_pca, data_train_y_pca)
pred = gbc_pca.predict(data_test_x_pca)
print(classification_report(data_test_y_pca,pred))

# ADA boosting with PCA

param_grid = {'learning_rate' :[0.0001,0.001,0.01,0.1,1,1.2,2] ,
              "algorithm":['SAMME', 'SAMME.R']}
random_search_ada = RandomizedSearchCV(AdaBoostClassifier(n_estimators = 1000), param_distributions=param_grid,
                                   n_iter=10, random_state=0,n_jobs=-1,cv= 5,return_train_score=True)
random_search_ada.fit(data_train_x_pca, data_train_y_pca)
pd.DataFrame(random_search_ada.cv_results_)
print(random_search_ada.best_estimator_)

ada_pca = AdaBoostClassifier(algorithm='SAMME', base_estimator=None, learning_rate=1,
          n_estimators=1000, random_state=None)
ada_pca.fit(data_train_x_pca, data_train_y_pca)
pred = ada_pca.predict(data_test_x_pca)
print(classification_report(data_test_y_pca,pred))


 
from sklearn.ensemble import VotingClassifier

estimators=[('svm', svc), ('Gb', gbc_pca), ('lr', logistic_pca), ('dt', dt_pca),('rf', rf_pca),('knn', knn_pca),('svc',linearsvc),('AB',ada_pca)]

majority_voting = VotingClassifier(estimators, voting='hard')


majority_voting.fit(data_train_x_pca, data_train_y_pca)

majority_voting.score(data_test_x_pca, data_test_y_pca)


joblib.dump(majority_voting, "heart_disease.joblib")


model_max_v = joblib.load("heart_disease.joblib" )


y_preds = model_max_v.predict(data_test_x_pca)
print('majority_voting_accuracy: ',majority_voting.score(data_test_x_pca, data_test_y_pca))
print('\n')