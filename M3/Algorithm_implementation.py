#!/usr/bin/env python
# coding: utf-8

# In[35]:


import numpy as np
import pandas as pd


# In[36]:


import matplotlib.pyplot as plt
from matplotlib import rcParams
from matplotlib.cm import rainbow
get_ipython().run_line_magic('matplotlib', 'inline')
import warnings
warnings.filterwarnings('ignore')
import seaborn as sns


# In[37]:


from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


# In[38]:


from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier


# In[39]:


dataset = pd.read_csv('./dataset/Preprocessed_Dataset.csv')
dataset


# In[32]:


dataset.head()


# In[33]:


dataset.tail()


# In[34]:


dataset.info()


# In[10]:


dataset.describe()


# In[11]:


rcParams['figure.figsize'] = 20, 14
plt.matshow(dataset.corr())
plt.yticks(np.arange(dataset.shape[1]), dataset.columns)
plt.xticks(np.arange(dataset.shape[1]), dataset.columns)
plt.colorbar()


# In[12]:


rcParams['figure.figsize'] = 8,6
plt.bar(dataset['target'].unique(), dataset['target'].value_counts(), color = ['red', 'green'])
plt.xticks([0, 1])
plt.xlabel('Target Classes')
plt.ylabel('Count')
plt.title('Count of each Target Class')


# In[13]:


dataset = pd.get_dummies(dataset, columns = ['sex', 'cp', 'fbs', 'restecg', 'exang', 'slope', 'ca', 'thal'])


# In[14]:


standardScaler = StandardScaler()
columns_to_scale = ['age', 'trestbps', 'chol', 'thalach', 'oldpeak']
dataset[columns_to_scale] = standardScaler.fit_transform(dataset[columns_to_scale])


# In[15]:


y = dataset['target']
X = dataset.drop(['target'], axis = 1)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.33, random_state = 0)


# In[16]:


knn_scores = []
for k in range(1,21):
    Model = KNeighborsClassifier(n_neighbors = k)
    Model.fit(X_train, y_train)
    knn_scores.append(Model.score(X_test, y_test))


# In[17]:


kn_sc=knn_scores[7]*100
print("The score for K Neighbors Classifier is {}% with {} nieghbors.".format(knn_scores[7]*100, 8))


# In[18]:


svc_scores = []
kernels = ['linear', 'poly', 'rbf', 'sigmoid']
for i in range(len(kernels)):
    svc_classifier = SVC(kernel = kernels[i])
    svc_classifier.fit(X_train, y_train)
    svc_scores.append(svc_classifier.score(X_test, y_test))


# In[19]:


svc_sc=svc_scores[0]*100
print("The score for Support Vector Classifier is {}% with {} kernel.".format(svc_scores[0]*100, 'linear'))


# In[20]:


dt_scores = []
for i in range(1, len(X.columns) + 1):
    dt_classifier = DecisionTreeClassifier(max_features = i, random_state = 0)
    dt_classifier.fit(X_train, y_train)
    dt_scores.append(dt_classifier.score(X_test, y_test))


# In[21]:


dt_sc=dt_scores[17]*100
print("The score for Decision Tree Classifier is {}% with {} maximum features.".format(dt_scores[17]*100, [2,4,18]))


# In[22]:


rf_scores = []
estimators = [10, 100, 200, 500, 1000]
for i in estimators:
    rf_classifier = RandomForestClassifier(n_estimators = i, random_state = 0)
    rf_classifier.fit(X_train, y_train)
    rf_scores.append(rf_classifier.score(X_test, y_test))


# In[23]:


rf_sc=rf_scores[1]*100
print("The score for Random Forest Classifier is {}% with {} estimators.".format(rf_scores[1]*100, [100, 500]))


# In[24]:


model_al=pd.DataFrame({'KNN':kn_sc,'decision':dt_sc,'random':rf_sc,'svc':svc_sc},index=[0])


# In[25]:


model_al


# In[26]:


models=list(model_al.keys())
scores=[kn_sc,dt_sc,rf_sc,svc_sc]


# In[27]:


plt.figure(figsize=(10,5))
sns.barplot(x=models,y=scores)
plt.xlabel('models')
plt.ylabel('scores')
plt.show()


# In[28]:


import pickle


# In[40]:


with open('./Model/heart_model.pickle','wb') as f:
    pickle.dump(Model,f)
    f.close()


# In[41]:


model = pickle.load(open('./Model/heart_model.pickle', 'rb'))
print("Model Successfully created...!")


# In[ ]:




