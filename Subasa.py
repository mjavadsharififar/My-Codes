#!/usr/bin/env python
# coding: utf-8

# These codes are answer to a challenge designed by https://quera.org
# first challenge is about data analysis and second on wants to build classification model

# In[211]:


import pandas as pd
train = pd.read_csv('C:\\Users\\admin\\Desktop\\sobasa\\train.csv')


# ## 1.New job 

# Link of the question of this challenge is https://quera.org/contest/assignments/28792/problems/95183

# ### Q1: number of players in dataset

# In[215]:


import numpy as np
np.count_nonzero(np.unique(train['playerId']))


# ### Q2: Which player has scored the most goals?

# In[228]:


pd.crosstab(train.playerId,train.outcome).idxmax()


# ### Q3:  which players have the highest and smallest conversion rate?

# In[240]:


a = pd.crosstab(train.playerId,train.outcome,margins = True)
a['rate'] = a['گُل']/a['All']
print(a[['rate']].idxmax())
print(a[['rate']].idxmin())


# ### Q4: What was the Euclidean distance of the farthest shot to the target?

# In[247]:


from math import sqrt
train['distance'] = (train['x']**2 + train['y']**2).apply(lambda x: sqrt(x))
int(train['distance'].max())


# ## 2.Goal Scoring Prabablity 

# In[ ]:


Link of the question of this challenge is https://quera.org/contest/assignments/28792/problems/95184


# In[279]:


#preprocessing training dataset
train2 = train.drop(['matchId','playerId','distance'],axis = 'columns')
dic = {'مهار توسط دروازه بان':0,'موقعیت از دست رفته':0,'برخورد به دفاع':0,'برخورد به تیردروازه':0,'گُل':1,'گُل به خودی':1}
train2['outcome'] = train2['outcome'].map(dic)
train3 = pd.get_dummies(train2[['playType','bodyPart','interferenceOnShooter']])
train3 = train3.join(train2[['x','y','interveningOpponents','interveningTeammates','minute','second','outcome']])


# In[284]:


#train_test_split
y = train3['outcome']
X = train3.drop(['outcome'],axis = 1)
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test = train_test_split(X,y,random_state = 0)


# In[317]:


#normalization
from sklearn.preprocessing import MinMaxScaler
minmax = MinMaxScaler().fit(X_train)
X_train_norm = minmax.transform(X_train)
X_test_norm = minmax.transform(X_test)


# In[329]:


#examin different algorithemes
#1.KNN
from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier()


# In[330]:


#Grid search for parameter tuning
from sklearn.model_selection import GridSearchCV
grid_val = {'n_neighbors':[3,4,5,6,7],'weights': ['uniform','distance'],'metric': ['minkowski','Euclidean']}
grid_knn_auc = GridSearchCV(knn, param_grid = grid_val ,cv = 3,scoring = 'roc_auc') # using auc as metric
grid_knn_auc.fit(X_train_norm,y_train)
print('knn best values based on auc:',grid_knn_auc.best_params_ )
print('knn best score based on auc:',grid_knn_auc.best_score_ )


# In[326]:


#results of best parameters for KNN
knn = KNeighborsClassifier('metric'= 'minkowski', 'n_neighbors'= 7, 'weights'= 'distance')
knn.fit(X_train_norm,y_train)
score_knn = knn.score(X_test_norm,y_test)
y_proba = knn.predict_proba(X_test_norm)
y_score = y_proba[:,1]
from sklearn.metrics import roc_curve, auc
fpr_knn, tpr_knn,thresh = roc_curve(y_test, y_score)
auc_knn = auc(fpr_knn, tpr_knn)
print(auc_knn)


# In[335]:


#2. LogReg
from sklearn.linear_model import LogisticRegression
lr = LogisticRegression(max_iter = 10000)


# In[341]:


#Grid search for parameter tuning
grid_val = {'C':[0.01,0.1,1,10,100]}
grid_lr_auc = GridSearchCV(lr, param_grid = grid_val ,cv = 3,scoring = 'roc_auc') # using auc as metric
grid_lr_auc.fit(X_train_norm,y_train)
print('lr best values based on auc:',grid_lr_auc.best_params_ )
print('lr best score based on auc:',grid_lr_auc.best_score_ )


# In[346]:


#results of best parameters for LogReg
lr = LogisticRegression(C = 1,max_iter = 10000)
lr.fit(X_train_norm,y_train)
score_lr = lr.score(X_test_norm,y_test)
y_proba2 = lr.predict_proba(X_test_norm)
y_score2 = y_proba2[:,1]
fpr_lr, tpr_lr,thresh = roc_curve(y_test, y_score2)
auc_lr = auc(fpr_lr, tpr_lr)
print(auc_lr)


# In[348]:


#3. Perceptron
from sklearn.neural_network import MLPClassifier
nnclf = MLPClassifier(max_iter = 10000,solver = 'lbfgs',random_state = 0)


# In[ ]:


#Grid search for parameter tuning
grid_val = {'hidden_layer_sizes':[[100],[100,100],[100,100,100]],'activation':['relu','logistic','tanh'],'alpha':[0.001,0.01,0.1,1,10]}
grid_nnclf_auc = GridSearchCV(nnclf, param_grid = grid_val ,cv = 3,scoring = 'roc_auc') # using auc as metric
grid_nnclf_auc.fit(X_train_norm,y_train)
print('nnclf best values based on auc:',grid_nnclf_auc.best_params_ )
print('nnclf best score based on auc:',grid_nnclf_auc.best_score_ )


# In[356]:


nnclf = MLPClassifier(max_iter = 10000,solver = 'lbfgs',hidden_layer_sizes = [100,100],random_state = 0,activation ='tanh',alpha = 1 )
nnclf.fit(X_train_norm,y_train)
score_nnclf = nnclf.score(X_test_norm,y_test)
y_proba3 = nnclf.predict_proba(X_test_norm)
y_score3 = y_proba3[:,1]
fpr_nnclf, tpr_nnclf,thresh = roc_curve(y_test, y_score3)
auc_nnclf = auc(fpr_nnclf, tpr_nnclf)
print(auc_nnclf)


# In[327]:


#Comparing algorithemes using plots
import matplotlib.pyplot as plt
plt.figure()
plt.xlim([-0.01, 1.00])
plt.ylim([-0.01, 1.01])
print(f"model: knn, accuracy = {score_knn}   AUC = {auc_knn}")
print(f"model: LogReg, accuracy = {score_lr}   AUC = {auc_lr}")
print(f"model: nnclf, accuracy = {score_nnclf}   AUC = {auc_nnclf}")
plt.plot(fpr_knn, tpr_knn, lw=3, alpha=0.7, label='knn')
plt.plot(fpr_lr, tpr_lr, lw=3, alpha=0.7, label='LogReg')
plt.plot(fpr_nnclf, tpr_nnclf, lw=3, alpha=0.7, label='perceptron')
plt.xlabel('False Positive Rate', fontsize=16)
plt.ylabel('True Positive Rate (Recall)', fontsize=16)
plt.plot([0, 1], [0, 1], color='k', lw=0.5, linestyle='--')
plt.legend(loc="lower right", fontsize=11)
plt.title('ROC curve', fontsize=16)
plt.show()
#according to this, best algorithem is perceptron, so we use it for test data


# In[ ]:


#importing test data and preprocessing
test = pd.read_csv('C:\\Users\\admin\\Desktop\\sobasa\\test.csv')
test2 = pd.get_dummies(test[['playType','bodyPart','interferenceOnShooter']])
test2 = MinMaxScaler().fit_transform(test2)


# In[ ]:


#bulding prediction dataframe
y_proba_prediction = nnclf.predict_proba(test)
prediction = pd.DataFrame(y_proba_prediction[:,1],)

