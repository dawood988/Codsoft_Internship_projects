#!/usr/bin/env python
# coding: utf-8

# # TITANIC SURVIVAL PREDICTION

# -Use the Titanic dataset to build a model that predicts whether a
#  passenger on the Titanic survived or not. This is a classic beginner
#  project with readily available data.
#  
#  -The dataset typically used for this project contains information
#  about individual passengers, such as their age, gender, ticket
#  class, fare, cabin, and whether or not they survived.

# # Importing libraries

# In[1]:


import pandas as pd
import numpy as np


# In[2]:


data = pd.read_csv(r"C:\Users\Dawood MD\OneDrive\Desktop\Codsoft\Titanic archive (1)\Titanic-Dataset.csv")


# In[3]:


data


# EDA (exploratory Data Analysis)

# In[4]:


data.describe(include='all')


# In[5]:


data.info()


# Drop All the unnecessary data from the dataset

# In[6]:


data.duplicated().sum()


# In[7]:


data.drop("Name",axis=1,inplace=True)
data.drop("Ticket",axis=1,inplace=True)
data.drop("PassengerId",axis=1,inplace=True)
data.drop("Cabin",axis=1,inplace=True)
data.drop("Embarked",axis=1,inplace=True)


# In[8]:


data


# In[9]:


from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
data['Sex']=le.fit_transform(data['Sex'])
df = data 


# In[10]:


data


# In[11]:


# Splitting the data in X and y Columns
y = data['Survived']
X = data.iloc[:,1:]


# In[12]:


X.info()


# Deleting the column with missing data

# In[13]:


df_1 = X.dropna(axis=1)


# In[14]:


df_1.info()


# Model Building with removing the NaN value column i,e 'Age'

# In[15]:


from sklearn.linear_model import LogisticRegression
from sklearn import metrics
from sklearn.model_selection import train_test_split

#spliting the data in training and testing 
x_train,x_test,y_train,y_test = train_test_split(df_1,y,test_size=0.3,random_state=42)
#buliding a model from a dataframe without Age column
lr = LogisticRegression()
model_1 = lr.fit(x_train,y_train)
pred = model_1.predict(x_test)
print("Accuracy:",metrics.accuracy_score(y_test,pred))


# In[16]:


from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

# Calculate confusion matrix for model
cm_model = confusion_matrix(y_test, pred)

# Plot confusion matrix for model
plt.figure(figsize=(8, 6))
sns.heatmap(cm_model, annot=True, fmt='d', cmap='Blues')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix - Model')
plt.show()


# Similarly Deleting the column with missing data

# In[17]:


df_2 = df.dropna(axis=0)


# In[18]:


df_2.info()


# In[19]:


y_1 = df_2['Survived']
X_1 = df_2.iloc[:,1:]


# In[20]:


from sklearn.linear_model import LogisticRegression
from sklearn import metrics
from sklearn.model_selection import train_test_split

#spliting the data in training and testing 
x_train,x_test,y_train,y_test = train_test_split(X_1,y_1,test_size=0.3,random_state=42)
#buliding a model from a dataframe without Age column
lr = LogisticRegression()
model_2 = lr.fit(x_train,y_train)
pred = model_2.predict(x_test)
print("Accuracy:",metrics.accuracy_score(y_test,pred))


# In[21]:


from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

# Calculate confusion matrix for model
cm_model = confusion_matrix(y_test, pred)

# Plot confusion matrix for model
plt.figure(figsize=(8, 6))
sns.heatmap(cm_model, annot=True, fmt='d', cmap='Blues')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix - Model')
plt.show()


# Similarly for the Imputation of NaN values |Filling the Missing Values – Imputation

# In[22]:


df_3 = df.fillna(df["Age"].mean())
df_3.info()


# In[23]:


y_3 = df_3['Survived']
X_3 = df_3.iloc[:,1:]


# In[24]:


y_3.info()


# In[25]:


X_3.info()


# In[26]:


from sklearn.linear_model import LogisticRegression
from sklearn import metrics
from sklearn.model_selection import train_test_split

#spliting the data in training and testing 
x_train,x_test,y_train,y_test = train_test_split(X_3,y_3,test_size=0.3,random_state=42)
#buliding a model from a dataframe without Age column
lr = LogisticRegression()
model_2 = lr.fit(x_train,y_train)
pred = model_2.predict(x_test)
print("Accuracy:",metrics.accuracy_score(y_test,pred))


# In[27]:


from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

# Calculate confusion matrix for model
cm_model = confusion_matrix(y_test, pred)

# Plot confusion matrix for model
plt.figure(figsize=(8, 6))
sns.heatmap(cm_model, annot=True, fmt='d', cmap='Blues')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix - Model')
plt.show()


# Simple Imputer Method 

# In[28]:


df_4 = df


# In[29]:


df_4['Missing age values'] = df_4['Age'].isnull()


# In[30]:


df_4.info()


# In[31]:


from sklearn.impute import SimpleImputer
my_imputer = SimpleImputer(strategy = 'median')
data_new = my_imputer.fit_transform(df_4)
df_5 = pd.DataFrame(data_new,columns=['Survived','Pclass','Sex','Age','SibSp','Parch','Fare','Missing age values'])
df_5.info()


# In[49]:


y_4 = df_5['Survived']
X_4 = df_5.iloc[:,1:]
from sklearn.linear_model import LogisticRegression
from sklearn import metrics
from sklearn.model_selection import train_test_split

#spliting the data in training and testing 
x_train,x_test,y_train,y_test = train_test_split(X_4,y_4,test_size=0.3,random_state=42)
#buliding a model from a dataframe without Age column
lr = LogisticRegression()
pred = model_2.predict(x_test)
print("Accuracy:",metrics.accuracy_score(y_test,pred))


# In[50]:


from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

# Calculate confusion matrix for model
cm_model = confusion_matrix(y_test, pred)

# Plot confusion matrix for model
plt.figure(figsize=(8, 6))
sns.heatmap(cm_model, annot=True, fmt='d', cmap='Blues')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix - Model')
plt.show()


# Filling with a Regression Model
# 
# In this case, the null values in one column are filled by fitting a regression model using other columns in the dataset.
# 
# I.e. in this case the regression model will contain all the columns except Age in X and Age in Y.
# 
# Then after filling the values in the Age column, then we will use logistic regression to calculate accuracy.

# In[42]:


import warnings
warnings.filterwarnings("ignore")


# In[43]:


from sklearn.linear_model import LinearRegression
import warnings
lr = LinearRegression()
df.head()
testdf = df[df['Age'].isnull()==True]
traindf = df[df['Age'].isnull()==False]
y = traindf['Age']
traindf.drop("Age",axis=1,inplace=True)
lr.fit(traindf,y)
testdf.drop("Age",axis=1,inplace=True)
pred = lr.predict(testdf)
testdf['Age']= pred


# In[44]:


traindf['Age']=y


# In[45]:


y = traindf['Survived']
traindf.drop("Survived",axis=1,inplace=True)
from sklearn.linear_model import LogisticRegression
lr = LogisticRegression()
lr.fit(traindf,y)


# In[46]:


LogisticRegression(C=1.0, class_weight=None, dual=False, fit_intercept=True,
                   intercept_scaling=1, l1_ratio=None, max_iter=100,
                   multi_class='auto', n_jobs=None, penalty='l2',
                   random_state=None, solver='lbfgs', tol=0.0001, verbose=0,
                   warm_start=False)


# In[47]:


y_test = testdf['Survived']
testdf.drop("Survived",axis=1,inplace=True)
pred = lr.predict(testdf)


# In[48]:


print(metrics.accuracy_score(pred,y_test))


# In[41]:


from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

# Calculate confusion matrix for model
cm_model = confusion_matrix(y_test, pred)

# Plot confusion matrix for model
plt.figure(figsize=(8, 6))
sns.heatmap(cm_model, annot=True, fmt='d', cmap='Blues')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix - Model')
plt.show()


# In[ ]:




