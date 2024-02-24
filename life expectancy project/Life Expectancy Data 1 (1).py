#!/usr/bin/env python
# coding: utf-8

# In[5]:


#Importing Libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
plt.style.use('dark_background')


# ### Data Analysis :

# In[6]:


df = pd.read_csv('Life Expectancy Data.csv')
df.head()


# In[7]:


#Checking dimensions of DataFrame
df.shape


# In[8]:


df.columns


# In[9]:


df.info()


# In[10]:


#Removing rows with missing values
df = df.dropna()


# In[11]:


#Count of missing values
df.isna().sum()


# In[12]:


#Totalcount of missing values
df.isnull().sum().sum()


# ### Data Preprocessing :

# In[13]:


#Dropping unnecessary columns
df = df.drop(['Status','percentage expenditure','Hepatitis B','Measles ','under-five deaths ','Total expenditure','infant deaths',' HIV/AIDS',' thinness  1-19 years',' thinness 5-9 years','Income composition of resources','Schooling'], axis = 1)
df.head()


# In[14]:


df['Country'].value_counts()


# In[15]:


# Selecting Features and Target Varaibles :
x = df[['Year', 'Adult Mortality', 'Alcohol', ' BMI ', 'Polio', 'Diphtheria ' , 'GDP', 'Population']]
y = df['Life expectancy ']


# In[16]:


#Splitting Data into Training & Testing
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2)


# In[17]:


#Training the Model
from sklearn.linear_model import LinearRegression
lr = LinearRegression()
lr.fit(x_train, y_train)
predictions = lr.predict(x_test)


# In[18]:


diff = y_test - predictions


# ### Visualization :

# In[19]:


import seaborn as sns
sns.distplot(diff)


# In[22]:


#Saving the File
import pickle
pickle.dump(lr, open('./LifeExpectancy_Model.sav', 'wb'))


# ### Prediction :

# In[27]:


x_test.iloc[1:2,:]


# In[20]:


#Opening the Model file In binary read mode
file = open('./LifeExpectancy_Model.sav','rb')
model = pickle.load(file)


# In[21]:


model.predict(x_test.iloc[1:2,:])


# In[33]:


# Replace with the correct absolute path
file = open('D:/Project/LifeExpectancy_Model.sav', 'rb')
model = pickle.load(file)


# In[34]:


# Prepare the input data for prediction
data = pd.DataFrame({
    'Year': [2016],
    'Adult Mortality': [200],
    'Alcohol': [4.5],
    'BMI': [25],
    'Polio': [90],
    'Diphtheria': [95],
    'GDP': [5000],
    'Population': [5000000]
})


# In[35]:


# Predict life expectancy
prediction = model.predict(data)


# In[36]:


print(prediction)


# ### Conclusion :

# Based on the predicted life expectancy value of approximately 67.70 years, it can be concluded that the input data provided is associated with an estimated life expectancy of around 67.70 years.
