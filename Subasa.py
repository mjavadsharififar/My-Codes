#!/usr/bin/env python
# coding: utf-8
# In[211]:
import pandas as pd
import sklearn.preprocessing

# In[212]:


train = pd.read_csv('C:\\Users\\admin\\Desktop\\sobasa\\train.csv')


# In[213]:


train.tail(60)


# In[214]:


import numpy as np


# In[215]:


np.count_nonzero(np.unique(train['playerId']))


# In[221]:


pd.DataFrame(train.groupby('outcome').max())


# In[228]:


pd.crosstab(train.playerId,train.outcome).idxmax()


# In[240]:


a = pd.crosstab(train.playerId,train.outcome,margins = True)
a['rate'] = a['گُل']/a['All']
a


# In[244]:


print(a[['rate']].idxmax())
a[['rate']].idxmin()


# In[247]:


from math import sqrt
train['distance'] = (train['x']**2 + train['y']**2).apply(lambda x: sqrt(x))


# In[249]:


int(train['distance'].max())


# ## Goal Prabablity 

# In[10]:


train2 = train.drop(['matchId','playerId'],axis = 'columns')


# In[11]:


train2


# In[12]:


dic = {'مهار توسط دروازه بان':0,'موقعیت از دست رفته':0,'برخورد به دفاع':0,'برخورد به تیردروازه':0,'گُل':1,'گُل به خودی':1}


# In[13]:


train2['outcome'] = train2['outcome'].map(dic)


# In[14]:


train2


# In[15]:


train3 = pd.get_dummies(train2[['playType','bodyPart','interferenceOnShooter']])


# In[16]:


train3 = train3.join(train2[['x','y','interveningOpponents','interveningTeammates','minute','second','outcome']])


# In[17]:


train3


# In[18]:


y = train3['outcome']


# In[19]:


X = train3.drop(['outcome'],axis = 1)


# In[20]:


X


# In[106]:


from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test = train_test_split(X,y,random_state = 0)


# In[107]:


from sklearn.preprocessing import MinMaxScaler
minmax = MinMaxScaler().fit(X_train)
X_train_norm = minmax.transform(X_train)
X_test_norm = minmax.transform(X_test)


# In[108]:


from sklearn.preprocessing import StandardScaler
std = StandardScaler().fit(X_train)
X_train_std = std.transform(X_train)
X_test_std = std.transform(X_test)


# In[90]:


from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier(n_neighbors=5,metric = 'euclidean',weights ='distance' )


# In[91]:


knn.fit(X_train_std,y_train)


# In[92]:


knn.score(X_test_std,y_test)


# In[93]:


y_proba = knn.predict_proba(X_test_std)
y_proba


# In[94]:


pred = pd.DataFrame(data = y_proba[:,1], index = None, columns = ['prediction'])


# In[95]:


pred


# In[96]:


y_score = y_proba[:,1]
from sklearn.metrics import roc_curve, auc
fpr_lr, tpr_lr,thresh = roc_curve(y_test, y_score)
roc_auc_lr = auc(fpr_lr, tpr_lr)


# In[98]:


roc_auc_lr


# In[161]:


from sklearn.linear_model import LogisticRegression
lr = LogisticRegression(C = 10)


# In[162]:


lr.fit(X_train_std,y_train)


# In[163]:


lr.score(X_test_std,y_test)


# In[164]:


y_proba2 = lr.predict_proba(X_test_std)


# In[165]:


y_score2 = y_proba2[:,1]
from sklearn.metrics import roc_curve, auc
fpr_lr, tpr_lr,thresh = roc_curve(y_test, y_score2)
roc_auc_lr2 = auc(fpr_lr, tpr_lr)
roc_auc_lr2


# In[154]:


from sklearn.svm import SVC
svm = SVC(kernel = 'rbf', C = 10)


# In[155]:


svm.fit(X_train_std,y_train)


# In[156]:


lr.score(X_test_std,y_test)


# In[157]:


y_proba3 = svm.decision_function(X_test_std)
y_score3 = y_proba3
from sklearn.metrics import roc_curve, auc
fpr_lr, tpr_lr,thresh = roc_curve(y_test, y_score3)
roc_auc_lr3 = auc(fpr_lr, tpr_lr)
roc_auc_lr3


# In[166]:


from sklearn.naive_bayes import GaussianNB
clf = GaussianNB().fit(X_train_std,y_train)


# In[167]:


clf.score(X_test_std,y_test)


# In[169]:


y_proba4 = clf.predict_proba(X_test_std)
y_score4 = y_proba4[:,1]
from sklearn.metrics import roc_curve, auc
fpr_lr, tpr_lr,thresh = roc_curve(y_test, y_score4)
roc_auc_lr4 = auc(fpr_lr, tpr_lr)
roc_auc_lr4


# In[170]:


from sklearn.ensemble import RandomForestClassifier
rf = RandomForestClassifier().fit(X_train_std,y_train)


# In[171]:


rf.score(X_test_std,y_test)


# In[172]:


y_proba5 = rf.predict_proba(X_test_std)
y_score5 = y_proba5[:,1]
from sklearn.metrics import roc_curve, auc
fpr_lr, tpr_lr,thresh = roc_curve(y_test, y_score5)
roc_auc_lr5 = auc(fpr_lr, tpr_lr)
roc_auc_lr5


# In[175]:


from sklearn.neural_network import MLPClassifier
nnclf = MLPClassifier(max_iter = 10000)


# In[176]:


nnclf.fit(X_train_std,y_train)


# In[177]:


nnclf.score(X_test_std,y_test)


# In[178]:


y_proba6 = nnclf.predict_proba(X_test_std)
y_score6 = y_proba6[:,1]
from sklearn.metrics import roc_curve, auc
fpr_lr, tpr_lr,thresh = roc_curve(y_test, y_score5)
roc_auc_lr6 = auc(fpr_lr, tpr_lr)
roc_auc_lr6


# In[194]:


train10 = pd.get_dummies(train2[['playType','bodyPart']])


# In[195]:


train10 = train10.join(train2[['x','y','interveningOpponents','interveningTeammates','minute','second','interferenceOnShooter','outcome']])


# In[196]:


train10


# In[197]:


dic2 = {'کم':1,'متوسط':2,'زیاد':3}


# In[198]:


train10['interferenceOnShooter'] = train10['interferenceOnShooter'].map(dic2)


# In[199]:


train10


# In[200]:


y2 = train10['outcome']
X2 = train10.drop(['outcome'],axis = 1)


# In[201]:


from sklearn.model_selection import train_test_split
X_train2,X_test2,y_train2,y_test2 = train_test_split(X2,y2,random_state = 0)


# In[202]:


from sklearn.preprocessing import MinMaxScaler
minmax = MinMaxScaler().fit(X_train2)
X_train_norm2 = minmax.transform(X_train2)
X_test_norm2 = minmax.transform(X_test2)


# In[203]:


from sklearn.preprocessing import StandardScaler
std = StandardScaler().fit(X_train2)
X_train_std2 = std.transform(X_train2)
X_test_std2 = std.transform(X_test2)


# In[210]:


svm.fit(X_train_norm2,y_train2)


# In[209]:


X_train2


# In[ ]:




