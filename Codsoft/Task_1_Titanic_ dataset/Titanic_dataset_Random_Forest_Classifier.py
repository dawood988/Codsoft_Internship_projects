#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from sklearn.metrics import mean_absolute_error
from sklearn.ensemble import RandomForestClassifier


# In[2]:


file = r"C:\Users\Dawood MD\OneDrive\Desktop\Codsoft\Task_1_Titanic_ dataset\Titanic-Dataset.csv"


# In[3]:


import chardet
with open(file,'rb') as rawdata:
    result = chardet.detect(rawdata.read(100000))
result    


# In[4]:


df = pd.read_csv(r"C:\Users\Dawood MD\OneDrive\Desktop\Codsoft\Task_1_Titanic_ dataset\Titanic-Dataset.csv",encoding='ascii')


# In[5]:


df


# In[6]:


df.shape


# In[7]:


df.head()


# In[8]:


df.tail()


# In[9]:


df.describe()


# In[10]:


df.duplicated().sum()


# In[11]:


Survived = df['Survived'].value_counts().reset_index()


# In[12]:


Survived


# In[13]:


df_1 =df.copy()


# In[14]:


df_1 = pd.get_dummies(df[['Survived','Sex']])


# In[15]:


df_1


# In[16]:


df_1.value_counts(subset=['Survived','Sex_female'])


# In[17]:


df_1.value_counts(subset=['Survived','Sex_male'])


# In[18]:


data = {'Survived': ['Male - No', 'Male - Yes', 'Female - No', 'Female - Yes'],
        'Counts': [468, 109, 81, 233]}  # replace with actual counts
Survived = pd.DataFrame(data)

plt.figure(figsize=(8, 6))
plt.bar(Survived['Survived'], Survived['Counts'],color=["Black","blue","red","pink"])
plt.xticks(Survived['Survived'])
plt.title('Comparison of Survival')
plt.xlabel('Gender and Survival Status')
plt.ylabel('Number of People')
plt.show()


# In[19]:


df.drop(['PassengerId','Name','SibSp','Parch','Ticket','Cabin','Embarked'],axis='columns',inplace=True)
df.head()


# In[20]:


inputs = df.drop('Survived',axis='columns')
target = df['Survived']


# In[21]:


target


# In[22]:


sex=pd.get_dummies(inputs.Sex)
sex.head()


# In[23]:


inputs=pd.concat([inputs,sex],axis="columns")
inputs.head()


# In[24]:


inputs.drop(["Sex"],axis="columns",inplace=True)


# In[25]:


inputs.head()


# In[26]:


inputs.isna().sum()


# In[27]:


inputs.Age = inputs.Age.fillna(inputs.Age.mean())
inputs.head()


# In[28]:


inputs.info()


# In[29]:


inputs.isna().sum()


# In[30]:


counts = df.groupby(['Survived', 'Sex']).size().unstack().fillna(0)

# Define the bar width
bar_width = 0.40
index = counts.index

# Plotting
fig, ax = plt.subplots()

# Plot bars for each Sex
bar1 = ax.bar(index - bar_width/2, counts['male'], bar_width, label='male')
bar2 = ax.bar(index + bar_width/2, counts['female'], bar_width, label='female')

# Setting labels and title
ax.set_xlabel('Survived')
ax.set_ylabel('Count')
ax.set_title('Survival Counts by Gender')
ax.set_xticks(index)
ax.set_xticklabels(['Not Survived', 'Survived'])
ax.legend()

# Display the plot
plt.show()


# In[31]:


X_train, X_test, y_train, y_test=train_test_split(inputs,target,test_size=0.2)


# In[32]:


X_train


# In[33]:


X_test


# In[34]:


y_train


# In[35]:


y_test


# In[36]:


inputs.corr()


# In[37]:


import seaborn as sns


# In[38]:


sns.heatmap(inputs.corr(), annot=True, cmap='coolwarm', fmt=".2f")


# In[39]:


model=RandomForestClassifier()


# In[40]:


model.fit(X_train,y_train)


# In[41]:


model.score(X_test,y_test)


# In[42]:


pred=model.predict(X_test)


# In[43]:


matrices=r2_score(pred,y_test)
matrices


# In[ ]:




