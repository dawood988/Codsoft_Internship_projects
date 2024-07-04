#!/usr/bin/env python
# coding: utf-8

# # SALES PREDICTION USING PYTHON
#  Sales prediction involves forecasting the amount of a product that
#  customers will purchase, taking into account various factors such as
#  advertising expenditure, target audience segmentation, and
#  advertising platform selection.
#  
#  In businesses that offer products or services, the role of a Data
#  Scientist is crucial for predicting future sales. They utilize machine
#  learning techniques in Python to analyze and interpret data, allowing
#  them to make informed decisions regarding advertising costs. By
#  leveraging these predictions, businesses can optimize their
#  advertising strategies and maximize sales potential. Let's embark on
#  the journey of sales prediction using machine learning in Python

# In[1]:


# Supress Warnings
import warnings
warnings.filterwarnings('ignore')

# Import the numpy and pandas package
import numpy as np
import pandas as pd

# Data Visualisation
import matplotlib.pyplot as plt 
import seaborn as sns


# In[2]:


file =r"C:\Users\Dawood MD\OneDrive\Desktop\Codsoft\Task_4_ sales_prediction_dataset\advertising.csv"


# In[3]:


import chardet
with open(file,'rb')as rawdata:
    result = chardet.detect(rawdata.read(10000))
result    


# In[4]:


df = pd.read_csv(file,encoding='ascii')


# In[5]:


df


# In[6]:


df.head()


# In[7]:


df.tail()


# In[8]:


df.shape


# In[10]:


df.dtypes


# In[11]:


df.describe()


# In[12]:


df.info()


# In[13]:


df.duplicated().sum()


# In[15]:


df.isna().sum()


# In[20]:


# Outlier Analysis
fig, axs = plt.subplots(3, figsize = (5,10))
plt1 = sns.boxplot(df['TV'], ax = axs[0])
plt2 = sns.boxplot(df['Newspaper'], ax = axs[1])
plt3 = sns.boxplot(df['Radio'], ax = axs[2])
plt.tight_layout()


#  There are no considerable outliers present in the data.

# # Exploratory Data Analysis
# Univariate Analysis

# In[22]:


# Sales as Traget variables
sns.boxplot(df['Sales'])
plt.show()


#  To undersatnd that how Sales are related with other variables using scatter plot.

# In[23]:


sns.pairplot(df, x_vars=['TV', 'Newspaper', 'Radio'], y_vars='Sales', height=4, aspect=1, kind='scatter')
plt.show()


# To Undersatnd how the correlation between different variables and how they are related.

# In[25]:


sns.heatmap(df.corr(), cmap="YlGnBu", annot = True)
plt.show()


# As it can be seen from the pairplot and the heatmap, the variable TV seems to be most correlated with Sales. 
# 
# So let's go ahead and perform simple linear regression using TV as our feature variable.

# # Model Buidling
# Steps in model building

# In[27]:


X = df['TV']
y = df['Sales']


# In[28]:


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state = 100)


# In[29]:


X_train.head()


# In[30]:


y_train.head()


# Building a Linear Model

# In[31]:


import statsmodels.api as sm


# In[32]:


# Add a constant to get an intercept
X_train_sm = sm.add_constant(X_train)

# Fit the resgression line using 'OLS'
lr = sm.OLS(y_train, X_train_sm).fit()


# In[33]:


# Print the parameters, i.e. the intercept and the slope of the regression line fitted
lr.params


# In[35]:


lr.summary()


# In[36]:


plt.scatter(X_train, y_train)
plt.plot(X_train, 6.948 + 0.054*X_train, 'r')
plt.show()


# In[37]:


y_train_pred = lr.predict(X_train_sm)
res = (y_train - y_train_pred)


# In[38]:


fig = plt.figure()
sns.distplot(res, bins = 15)
fig.suptitle('Error Terms', fontsize = 15)                  # Plot heading 
plt.xlabel('y_train - y_train_pred', fontsize = 15)         # X-label
plt.show()


# In[39]:


plt.scatter(X_train,res)
plt.show()


# In[40]:


# Add a constant to X_test
X_test_sm = sm.add_constant(X_test)

# Predict the y values corresponding to X_test_sm
y_pred = lr.predict(X_test_sm)


# In[41]:


y_pred.head()


# In[42]:


from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score


# In[43]:


# Returns the mean squared error
# we'll take a square root
np.sqrt(mean_squared_error(y_test, y_pred))


# In[45]:


accuracy = r2_score(y_test, y_pred)
accuracy


# Visualization of the test values

# In[46]:


plt.scatter(X_test, y_test)
plt.plot(X_test, 6.948 + 0.054 * X_test, 'r')
plt.show()


# In[ ]:




