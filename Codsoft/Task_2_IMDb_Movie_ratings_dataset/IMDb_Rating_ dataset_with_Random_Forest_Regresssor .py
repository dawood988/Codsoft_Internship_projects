#!/usr/bin/env python
# coding: utf-8

# In[2]:


import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.metrics import mean_squared_error,r2_score
from sklearn.impute import SimpleImputer


# In[3]:


file=r'C:\Users\Dawood MD\OneDrive\Desktop\Codsoft\IMDb Movie ratings\IMDb Movies India.csv'


# In[4]:


# To get to know the encoding of the csv file
import chardet

with open(file,'rb') as rawdata:
    result = chardet.detect(rawdata.read(100000))
result    


# In[5]:


df = pd.read_csv(file,encoding='ISO-8859-1')


# In[6]:


# Shows the first 5 rows of dataframe
df.head()


# In[7]:


# Shows the last % rows of dataframe
df.tail()


# In[8]:


#information about the dataframe
df.info()


# In[9]:


# checking for the duplicates
df.duplicated().sum()


# In[10]:


# To check what are the false duplicates
df[df.duplicated()]


# In[11]:


df.describe(include='all')


# In[12]:


df.isna().sum()


# In[13]:


# Removing Nan from target column
df = df.dropna(subset=['Rating'])


# In[14]:


df1 = df.copy()


# In[15]:


df.info()


# In[16]:


# Defining the X and Y Varibles i.e, Independent and Trahet variables
X = df[['Genre','Director','Actor 1','Actor 2','Actor 3']]
Y = df['Rating'] 


# In[17]:


# Train and test the function
x_train, x_test ,y_train ,y_test = train_test_split(X,Y,test_size=0.3,random_state=42)  


# In[18]:


# Defining the necessary columns required of Regression
categories = ['Genre','Director','Actor 1','Actor 2','Actor 3']


# In[19]:


preprocessor = ColumnTransformer(
    transformers =[
        ('cat',OneHotEncoder(handle_unknown='ignore'),categories)
                            ])


# In[20]:


#Defining Pipelines
model_pipeline = Pipeline(steps=[
    ('preprocessor',preprocessor),
    ('model',RandomForestRegressor(n_estimators=100,random_state=42))])


# In[21]:


# Train the model
model_pipeline.fit(x_train, y_train)


# In[22]:


# Predict on the test set
y_pred = model_pipeline.predict(x_test)


# In[23]:


# Evaluate the model
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
print(f'Mean Squared Error: {mse}')
print(f'R-squared: {r2}')


# In[26]:


from sklearn.ensemble import RandomForestClassifier
#Defining Pipelines
model_pipeline_2 = Pipeline(steps=[
    ('preprocessor',preprocessor),
    ('imputer', SimpleImputer(strategy='mean')), # if we have any missing values)
    ('model_2',RandomForestRegressor(n_estimators=100,random_state=42))])


# In[27]:


# Train the model
model_pipeline_2.fit(x_train, y_train)


# In[28]:


# Predict on the test set
y_pred_2 = model_pipeline_2.predict(x_test)


# In[29]:


# Evaluate the model
mse = mean_squared_error(y_test, y_pred_2)
r2 = r2_score(y_test, y_pred_2)
print(f'Mean Squared Error: {mse}')
print(f'R-squared: {r2}')


# In[31]:


# considering all the features  in x and rating in target varible
x = df.dropna(subset='Rating')
y = df['Rating']


# In[36]:


x.drop(columns="Name")


# In[35]:


# Convert 'Year' and 'Votes' to numeric, and 'Duration' to minutes if it's in the format 'h m'
x['Year']     = pd.to_numeric(x['Year'],errors='coerce')
x['Votes']    = x['Votes'].str.replace(',','').astype(float)
x['Duration'] = x['Duration'].str.extract('(\d+)').astype(float)


# In[37]:


# Fill missing or empty values in categorical features
categorical_features = ['Genre','Director','Actor 1','Actor 2','Actor 3']
for feature in categorical_features:
    x[feature] = x[feature].replace(' ','unknown').fillna('unknown')


# In[50]:


# Fill missing values in numerical features with the median
numerical_features = ['Year','Duration','Votes']
for feature in numerical_features:
    x[feature] = x[feature].fillna(x[feature].median()).replace('NaN',x[feature].median())


# In[51]:


# Split data into training and testing sets
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)


# In[52]:


# Create a column transformer for both categorical and numerical features
from sklearn.preprocessing import StandardScaler
preprocessor = ColumnTransformer(
    transformers=[
        ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features),
        ('num', StandardScaler(), numerical_features)
    ])


# In[57]:


# Create a preprocessing and training pipeline
model_pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('imputer', SimpleImputer(strategy='mean')), # if we have any missing values)
    ('model', RandomForestRegressor(random_state=42))
])


# In[58]:


# Hyperparameter tuning using Grid Search
param_grid = {
    'model__n_estimators': [100, 200, 300],
    'model__max_depth': [None, 10, 20, 30],
    'model__min_samples_split': [2, 5, 10],
    'model__min_samples_leaf': [1, 2, 4]
}


# In[55]:


x_train.isna().sum()


# In[59]:


from sklearn.model_selection import GridSearchCV
grid_search = GridSearchCV(model_pipeline, param_grid, cv=5, scoring='r2', n_jobs=-1)
grid_search.fit(x_train, y_train)


# In[60]:


# Best parameters
best_params = grid_search.best_params_
print(f'Best parameters: {best_params}')


# In[62]:


# Predict on the test set using the best estimator
best_model = grid_search.best_estimator_
y_pred = best_model.predict(x_test)


# In[63]:


# Evaluate the model
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
print(f'Mean Squared Error: {mse}')
print(f'R-squared: {r2}')


# In[ ]:




