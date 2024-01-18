#!/usr/bin/env python
# coding: utf-8

# # GRIP : THE SPARKS FOUNDATION
# 
# Data Science and Business Analytics 
# 
# Task 1 : prediction using supervised ML
# 
# In this task i will predict the percentage score of a student based on the number of hours studied. In this task, i have used two variables where the feature is the no. of hours studied and target value is the percentage score.
# This can be achieved with the help of Linear Regression.
# 

# #### importing required libraries

# In[46]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import seaborn as sns


# In[47]:


# reading the data
data_link = "stud.csv"
data = pd.read_csv(data_link)
data.head(27)


# In[48]:


data.info()


# In[49]:


data.describe()


# In[12]:


import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')


# In[50]:



data.plot(kind='scatter', x="Hours", y="Scores");
plt.show()


# In[15]:


data.corr(method= "pearson")


# In[51]:


hours =data["Hours"]
scores = data["Scores"]


# In[52]:


sns.displot(hours)


# In[53]:


sns.displot(scores)


# #### Linear Regression

# In[58]:


a= data.iloc[:,:-1].values
b= data.iloc[:,1].values


# In[76]:


from sklearn.model_selection import train_test_split
a_train,a_test,b_train,b_test= train_test_split(a,b, test_size=0.2, random_state=50)


# In[60]:


from sklearn.linear_model import LinearRegression
reg= LinearRegression()
reg.fit(a_train,b_train)


# In[67]:


m= reg.coef_          #coef_ and intercept_ are attributes of LinearRegression
c= reg.intercept_
line= m*a+c

plt.scatter (a,b)
plt.plot(a,line)
plt.show()


# In[68]:


b_pred = reg.predict(a_test)


# In[84]:


actual_pred = pd.DataFrame({"target": b_test, "predicted":b_pred})
actual_pred


# In[85]:


sns.set_style("whitegrid")
sns.displot(np.array(b_test - b_pred))


# ## what would be the predicted score of a student if he/she studies for 9.25 hours/day?

# In[90]:


h = 9.25
s = reg.predict([[h]])
print("if a student studies {} hours a day then he/she will score {} % in exam".format(h,s))

