#!/usr/bin/env python
# coding: utf-8

# ___
# 
# <a href='http://www.pieriandata.com'> <img src='../Pierian_Data_Logo.png' /></a>
# ___
# # Logistic Regression Project 
# 
# In this project we will be working with a fake advertising data set, indicating whether or not a particular internet user clicked on an Advertisement. We will try to create a model that will predict whether or not they will click on an ad based off the features of that user.
# 
# This data set contains the following features:
# 
# * 'Daily Time Spent on Site': consumer time on site in minutes
# * 'Age': cutomer age in years
# * 'Area Income': Avg. Income of geographical area of consumer
# * 'Daily Internet Usage': Avg. minutes a day consumer is on the internet
# * 'Ad Topic Line': Headline of the advertisement
# * 'City': City of consumer
# * 'Male': Whether or not consumer was male
# * 'Country': Country of consumer
# * 'Timestamp': Time at which consumer clicked on Ad or closed window
# * 'Clicked on Ad': 0 or 1 indicated clicking on Ad
# 
# ## Import Libraries
# 
# **Import a few libraries you think you'll need (Or just import them as you go along!)**

# In[1]:


import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')


# ## Get the Data
# **Read in the advertising.csv file and set it to a data frame called ad_data.**

# In[2]:


ad_data = pd.read_csv('C:/Users/003ZCV744/Documents/Jupyter Notebooks/Course materials/Refactored_Py_DS_ML_Bootcamp-master/13-Logistic-Regression/advertising.csv')


# **Check the head of ad_data**

# In[3]:


ad_data.head()


# ** Use info and describe() on ad_data**

# In[4]:


ad_data.info()


# In[5]:


ad_data.describe()


# ## Exploratory Data Analysis
# 
# Let's use seaborn to explore the data!
# 
# Try recreating the plots shown below!
# 
# ** Create a histogram of the Age**

# In[15]:


plt.hist('Age', data=ad_data, bins=30)


# **Create a jointplot showing Area Income versus Age.**

# In[18]:


sns.jointplot(x='Age',y='Area Income',data=ad_data)


# **Create a jointplot showing the kde distributions of Daily Time spent on site vs. Age.**

# In[23]:


sns.jointplot(x='Age',y='Daily Time Spent on Site',data=ad_data,kind='kde')


# In[66]:





# ** Create a jointplot of 'Daily Time Spent on Site' vs. 'Daily Internet Usage'**

# In[25]:


sns.jointplot(x='Daily Internet Usage',y='Daily Time Spent on Site',data=ad_data)


# In[72]:





# ** Finally, create a pairplot with the hue defined by the 'Clicked on Ad' column feature.**

# In[26]:


sns.pairplot(ad_data, hue='Clicked on Ad')


# In[84]:





# # Logistic Regression
# 
# Now it's time to do a train test split, and train our model!
# 
# You'll have the freedom here to choose columns that you want to train on!

# ** Split the data into training set and testing set using train_test_split**

# In[27]:


from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression


# In[29]:


ad_data.columns


# In[30]:


ad_data.info()


# In[31]:


X = ad_data[['Daily Time Spent on Site', 'Age', 'Area Income', 'Daily Internet Usage', 'Male']]
y = ad_data['Clicked on Ad']


# In[34]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=100)


# ** Train and fit a logistic regression model on the training set.**

# In[35]:


lrmodel = LogisticRegression()


# In[36]:


lrmodel.fit(X_train, y_train)


# ## Predictions and Evaluations
# ** Now predict values for the testing data.**

# In[37]:


predictions = lrmodel.predict(X_test)


# ** Create a classification report for the model.**

# In[39]:


from sklearn.metrics import classification_report


# In[41]:


print(classification_report(y_test, predictions))


# In[96]:





# ## Great Job!
