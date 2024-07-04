#!/usr/bin/env python
# coding: utf-8

# In[12]:


# Necessary Libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import plotly.express as px
import seaborn as sns


# In[13]:


# Clasiifiers and Regressors from sci-kit Learn
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier


# In[15]:


# Preprosessing ,Model Building and Accuracy from sci-kit Learn
from sklearn.preprocessing import MinMaxScaler, LabelEncoder 
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.model_selection  import cross_val_score
from sklearn.metrics import precision_score, recall_score, f1_score, mean_squared_error
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix


# In[16]:


# Clustering from the sci-kit Learn
from sklearn.cluster import KMeans


# In[17]:


# Tensorflow and ANN Libraries 
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense,Dropout


# In[18]:


#Warning Libraries
import warnings
warnings.filterwarnings("ignore")


# # Exploratrory Data Analysis

# In[28]:


# Reading the dataset
data= pd.read_csv(r"C:\Users\Dawood MD\OneDrive\Desktop\Codsoft\Task_3_iris dataset\IRIS.csv")
data


# In[29]:


# data Description
data.describe()


# In[30]:


# Checking For the null values
data.isna().sum()


# In[31]:


# Checking For the null values
data.isna().sum().sum()  


# In[40]:


# Counting the traget Variable 
Species = data['species'].value_counts().reset_index()
Species 


# In[41]:


# Plotting the p[ir chart
plt.figure(figsize=(8,8))
plt.pie(Species.value_counts(),labels=['Iris-setosa','Iris-versicolor','Iris-virginica'],autopct='%1.3f%%',explode=[0,0,0])
plt.legend(loc='upper left')
plt.show()


# In[42]:


#Scatter plot of different species
sns.FacetGrid(data, hue ='species', height = 4).map(plt.scatter,"petal_length","sepal_width").add_legend()
plt.show()


# In[43]:


#Scatter plot of different species
sns.FacetGrid(data, hue ='species', height = 4).map(plt.scatter,"sepal_length","petal_width").add_legend()
plt.show()


# In[44]:


#Scatter plot of different species
sns.FacetGrid(data, hue ='species', height = 4).map(plt.scatter,"sepal_length","sepal_width").add_legend()
plt.show()


# In[45]:


#Scatter plot of different species
sns.FacetGrid(data, hue ='species', height = 4).map(plt.scatter,"petal_length","petal_width").add_legend()
plt.show()


# In[46]:


fig = px.scatter_3d(data, x='sepal_length', y='petal_width', z='petal_length', color='species')
fig.show()


# In[47]:


# Differentiating the Independent and atrget varibales as X And y
X = data.drop('species', axis =1)
y = data['species']


# In[48]:


# Normalization of the Independent Variables
scaler = MinMaxScaler()
X = scaler.fit_transform(X)
X


# In[50]:


# Transformimng the Target variable
encoder = LabelEncoder()
y = encoder.fit_transform(y)
y


# In[51]:


# Splitting the data in Train and test
X_train, X_test, y_train, y_test = train_test_split(X,y,random_state=42,test_size=0.2)


# In[71]:


#Function for performing Linear regression and Heat map Parallelly
def evaluate(model):
    model.fit(X_train,y_train)
    pre = model.predict(X_test) 
    accuracy = accuracy_score(pre,y_test)
    sns.heatmap(confusion_matrix(pre,y_test),annot=True)
    print(model)
    print('Accuracy : ',accuracy)


# In[53]:


# Linear Model
model_LR = LogisticRegression()


# In[54]:


# Performing Linear Model
lsr_best = LogisticRegression(penalty='l2',C=1000.0,random_state = 42)
lsr_clf = lsr_best.fit(X_train,y_train)
evaluate(lsr_clf)


# In[55]:


# Kneighbors or KNN Model 
model_KNN = KNeighborsClassifier()


# In[72]:


# Performing Kneighbors or KNN Model
k_range = np.arange(1, 20, 2)
scores = [] #to store cross val score for each k
for k in k_range:
    model_KNN  = KNeighborsClassifier(n_neighbors=k)
    model_KNN .fit(X_train,y_train)
    score = cross_val_score(model_KNN , X_train, y_train, cv=3, n_jobs = -1)
    scores.append(score.mean())

#Storing the mean squared error to decide optimum k
mse = [1-x for x in scores]


# In[73]:


# Plot For K vs. MSE and K vs. Cross validation Accuracy
plt.figure(figsize=(20,8))
plt.subplot(121)
sns.lineplot(x=k_range,y=mse,markers=True,dashes=False)
plt.xlabel("Value of K")
plt.ylabel("Mean Squared Error")
plt.subplot(122)
sns.lineplot(x=k_range,y=scores,markers=True,dashes=False)
plt.xlabel("Value of K")
plt.ylabel("Cross Validation Accuracy")

plt.show()


# In[58]:


#Performing Kneighbors or KNN Model
knn = KNeighborsClassifier(n_neighbors=7)
knn_clf = knn.fit(X_train,y_train)
evaluate(knn_clf)


# In[59]:


# Random Forest Classifier
model_RFC = RandomForestClassifier()


# In[60]:


# Performing Random Forest Classifier
rf = RandomForestClassifier(max_depth=9, n_estimators=50)
RFC_clf = rf.fit(X_train,y_train)
evaluate(RFC_clf)


# In[61]:


# Support Vector Machine = Support Vecto Classifier
model_SVM = SVC()


# In[62]:


# Performing Support Vecto Classifier
svc = SVC()
svc_clf = svc.fit(X_train, y_train)
evaluate(svc_clf)


# In[63]:


# Decision Tree Classifier
model_DT = DecisionTreeClassifier()


# In[64]:


# Performing Decision Tree Classifier
DT = model_DT.fit(X_train, y_train)
evaluate(svc_clf)


# In[65]:


# ANN Model
model_ANN = Sequential([
    Dense(128, activation='relu', input_dim=X_train.shape[1]),

    Dense(64, activation='relu'),
    
    Dense(32, activation='relu'),
    
    Dense(16, activation='relu'),
    
    Dense(8, activation='relu'),
    
    Dense(3, activation='softmax')
])

model_ANN.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
model_ANN.summary()


# In[66]:


# Performong Ann Model
history = model_ANN.fit(X_train, y_train, epochs=50, validation_split=0.2)


# In[67]:


#Finding the optimum number of clusters for k-means classification
wcss = []

for i in range(1, 11):
    kmeans = KMeans(n_clusters = i, init = 'k-means++', max_iter = 300, n_init = 10, random_state = 42)
    kmeans.fit(X)
    wcss.append(kmeans.inertia_)


# In[68]:


plt.plot(range(1, 11), wcss)
plt.title('The elbow method')
plt.xlabel('Number of clusters')
plt.ylabel('WCSS') #within cluster sum of squares
plt.show()


# In[69]:


kmeans = KMeans(n_clusters = 3, init = 'k-means++', max_iter = 300, n_init = 10, random_state = 42)
y_kmeans = kmeans.fit_predict(X)


# In[70]:


plt.scatter(X[y_kmeans == 0, 0], X[y_kmeans == 0, 1], 
            s = 100, c = 'red', label = 'Iris-setosa')
plt.scatter(X[y_kmeans == 1, 0], X[y_kmeans == 1, 1], 
            s = 100, c = 'blue', label = 'Iris-versicolour')
plt.scatter(X[y_kmeans == 2, 0], X[y_kmeans == 2, 1],
            s = 100, c = 'green', label = 'Iris-virginica')

# Plotting the centroids of the clusters
plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:,1], 
            s = 100, c = 'yellow', label = 'Centroids')

plt.legend()


# In[ ]:




