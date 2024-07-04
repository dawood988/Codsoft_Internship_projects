#!/usr/bin/env python
# coding: utf-8

# In[3]:


get_ipython().system('pip install xgboost')
get_ipython().system('pip install lightgbm')
get_ipython().system('pip install catboost')


# In[4]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import classification_report, accuracy_score, roc_auc_score
import xgboost as xgb
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier
from sklearn.metrics import accuracy_score
import os


# In[5]:


file =r"C:\Users\Dawood MD\OneDrive\Desktop\Codsoft\Task_5_CredIt_Card_Fraud_Detection\creditcard.csv"


# In[7]:


import chardet
with open(file,'rb') as rawdata:
    result = chardet.detect(rawdata.read(10000))
result    


# In[8]:


df = pd.read_csv(file,encoding='ascii')


# In[9]:


df.head()


# In[10]:


df.tail()


# In[11]:


df.shape


# In[12]:


df.size


# In[13]:


df.info()


# In[14]:


df.describe()


# In[15]:


df.columns


# In[16]:


df.duplicated().sum()


# In[17]:


df.isnull().sum()


# In[18]:


df[df.duplicated()]


# In[20]:


class_count = df["Class"].value_counts()
print(class_count)
class_count_df = pd.DataFrame({'Class': class_count.index, 'values': class_count.values})

plt.figure(figsize=(8, 6))
plt.bar(class_count_df['Class'], class_count_df['values'], color='blue')
plt.title('Credit Card Fraud Class - data unbalance (Not fraud = 0, Fraud = 1)')
plt.xlabel('Class')
plt.ylabel('Number of transactions')
plt.xticks(class_count_df['Class'])
for i in range(len(class_count_df)):
    plt.text(class_count_df['Class'][i], class_count_df['values'][i], class_count_df['values'][i], ha='center', va='bottom')
plt.show()


# We can see that the data is highly imbalanced with 28315 for class 0 and 492 for class 1
# 
# Let's split the data to fraud and not_fraud:

# In[23]:


not_fraud = df.loc[df['Class'] == 0]
fraud = df.loc[df['Class'] == 1]


# In[24]:


not_fraud.Amount.describe()


# In[25]:


fraud.Amount.describe()


# We can see there is a remarkable difference between the mean of the amount of fraud and not_fraud

# In[28]:


# Compute the correlation matrix
corr_matrix = df.corr()

# Plot the heatmap
plt.figure(figsize=(12, 10))
sns.heatmap(corr_matrix, cmap='coolwarm', annot=False)
plt.title('Correlation Matrix')
plt.show()


# # EDA Data Preprocessing 

# Approach to solve unbalanced data
# 
# Creating new sample dataset containing same distribution as normal transactions and fraudulent transactions

# In[29]:


not_fraud_sample = not_fraud.sample(n=492)


# In[30]:


new_dataset = pd.concat([not_fraud_sample, fraud], axis=0)


# In[31]:


new_dataset.head()


# Splitting the data into Features & Targets by train test split function

# In[32]:


X = new_dataset.drop(columns='Class', axis=1)
Y = new_dataset['Class']


# In[33]:


X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, stratify=Y, random_state=2)


# Model Training and Hyperparameter Tuning

# Setting Up the Models and Hyperparameter Grids

# In[34]:


#Logistic Regression
log_reg = LogisticRegression(solver='liblinear')  # 'liblinear' is good for small datasets or when you need a simple model
log_reg_param_grid = {
    'C': [0.01, 0.1, 1, 10, 100],  # Regularization strength
    'penalty': ['l1', 'l2']  # Type of regularization
}


# In[35]:


# Random Forest Classifier
rf = RandomForestClassifier()
rf_param_grid = {
    'n_estimators': [100, 200, 300],
    'max_depth': [None, 10, 20, 30],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4],
    'bootstrap': [True, False]
}


# In[36]:


# AdaBoost Classifier
ada = AdaBoostClassifier()
ada_param_grid = {
    'n_estimators': [50, 100, 200],
    'learning_rate': [0.01, 0.1, 1.0, 1.5]
}


# In[37]:


# Categorical Boosting method called cat boost
cat = CatBoostClassifier(verbose=0)
cat_param_grid = {
    'iterations': [100, 200, 300],
    'depth': [4, 6, 10],
    'learning_rate': [0.01, 0.1, 0.2],
    'l2_leaf_reg': [1, 3, 5, 7, 9]
}


# In[38]:


# XGBoost classifier
xgb_model = xgb.XGBClassifier(eval_metric='logloss')
xgb_param_grid = {
    'n_estimators': [100, 200, 300],
    'learning_rate': [0.01, 0.1, 0.2],
    'max_depth': [3, 6, 9],
    'subsample': [0.6, 0.8, 1.0],
    'colsample_bytree': [0.6, 0.8, 1.0]
}


# In[39]:


#LGBoost classifier
lgbm = LGBMClassifier()
lgbm_param_grid = {
    'n_estimators': [100, 200, 300],
    'learning_rate': [0.01, 0.1, 0.2],
    'num_leaves': [31, 40, 50],
    'boosting_type': ['gbdt', 'dart'],
    'max_depth': [-1, 10, 20]
}


# In[40]:


# HyperparameterTuning
def grid_search(model, param_grid):
    grid_search = GridSearchCV(estimator=model, param_grid=param_grid, 
                               cv=3, n_jobs=-1, scoring='roc_auc', verbose=2)
    grid_search.fit(X_train, Y_train)
    print(f"Best parameters for {model.__class__.__name__}: {grid_search.best_params_}")
    print(f"Best ROC AUC for {model.__class__.__name__}: {grid_search.best_score_}")
    return grid_search.best_estimator_

# Random Forest
best_rf = grid_search(rf, rf_param_grid)

# AdaBoost
best_ada = grid_search(ada, ada_param_grid)

# CatBoost
best_cat = grid_search(cat, cat_param_grid)

# XGBoost
best_xgb = grid_search(xgb_model, xgb_param_grid)

# LightGBM
best_lgbm = grid_search(lgbm, lgbm_param_grid)


# Evaluate the Best Models Load Packages and Data

# In[41]:


models = {
    'Random Forest': best_rf,
    'AdaBoost': best_ada,
    'CatBoost': best_cat,
    'XGBoost': best_xgb,
    'LightGBM': best_lgbm
}

for name, model in models.items():
    Y_pred = model.predict(X_test)
    print(f"--- {name} ---")
    print(f"Accuracy: {accuracy_score(Y_test, Y_pred):.4f}")
    print(f"ROC AUC: {roc_auc_score(Y_test, model.predict_proba(X_test)[:,1]):.4f}")
    print(classification_report(Y_test, Y_pred))


# In[ ]:




