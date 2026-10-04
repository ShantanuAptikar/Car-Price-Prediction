#!/usr/bin/env python
# coding: utf-8

# In[60]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import LabelEncoder,MinMaxScaler,Binarizer
from sklearn.metrics import (mean_absolute_error,mean_squared_error,root_mean_squared_error)
import joblib


# In[61]:


cars=pd.read_csv("car_price_regression_data (1).csv")


# In[62]:


cars


# In[63]:


cars.info()


# In[64]:


cars.isnull().sum()


# In[65]:


cars["Brand"].unique()


# In[66]:


cars["Year"].unique()


# In[67]:


cars["Age_Years"].unique()


# In[68]:


cars["Fuel_Type"].unique()


# In[69]:


cars["Transmission"].unique()


# In[70]:


cars["Engine_CC"].unique()


# In[71]:


cars["Owner_Type"].unique()


# In[72]:


cars["Seats"].unique()


# In[73]:


cars["Price_Lakh"].unique()


# In[74]:


cars["Brand"].value_counts().plot(kind="bar")


# In[75]:


cars["Year"].value_counts().plot(kind="pie")


# In[76]:


a=cars.groupby(["Brand", "Model"])["Transmission"].value_counts()


# In[77]:


a.plot(kind="bar")


# In[78]:


a=cars.groupby(["Brand"])["Transmission"].value_counts()


# In[79]:


a.plot(kind="bar")


# In[80]:


b=cars.groupby(["Model"])["Fuel_Type"].value_counts()


# In[81]:


b.plot(kind="bar")


# In[82]:


cars


# In[83]:


cars.drop("Year",axis=1,inplace=True)


# In[84]:


cars


# In[85]:


cars.drop("Age_Years",axis=1,inplace=True)


# In[86]:


cars


# In[87]:


cars.drop("Mileage_kmpl",axis=1,inplace=True)


# In[88]:


cars


# In[89]:


cars.drop("Seats",axis=1,inplace=True)


# In[90]:


cars


# In[91]:


model=LabelEncoder()


# In[92]:


cars["Fuel_Type"]=model.fit_transform(cars["Fuel_Type"])


# In[93]:


cars


# In[94]:


cars.drop("Kilometers_Driven",axis=1,inplace=True)
cars


# In[95]:


cars["Owner_Type"]=model.fit_transform(cars["Owner_Type"])
cars


# In[96]:


cars.drop("Model",axis=1,inplace=True)
cars


# In[97]:


cars["Brand"]=model.fit_transform(cars["Brand"])
cars


# In[98]:


a = Binarizer()


# In[99]:


cars["Transmission"]=model.fit_transform(cars["Transmission"])
cars


# In[100]:


X = cars[['Brand','Fuel_Type','Transmission','Engine_CC','Owner_Type','Safety_Rating']]


# In[101]:


X


# In[102]:


y = cars['Price_Lakh']


# In[103]:


y


# In[104]:


X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)


# In[105]:


print(X_train.shape)


# In[106]:


print(X_test.shape)


# In[107]:


print(y_train.shape)


# In[108]:


print(y_test.shape)


# In[109]:


c=LinearRegression()


# In[110]:


c.fit(X,y)


# In[111]:


joblib.dump(c, 'cars_model.pkl')


# In[112]:


y_pred=c.predict(X_test)


# In[113]:


d=mean_absolute_error(y_test,y_pred)


# In[114]:


d


# In[115]:


e=mean_squared_error(y_test,y_pred)


# In[116]:


e


# In[117]:


f=root_mean_squared_error(y_test,y_pred)
f


# In[118]:


new_data = pd.DataFrame({
    'Brand': [7],
    'Fuel_Type': [3],
    'Transmission': [1],
    'Engine_CC': [1500],
    'Owner_Type': [0],
    'Safety_Rating': [5.0],
})

prediction = c.predict(new_data)

print("Predicted price:", prediction[0])


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




