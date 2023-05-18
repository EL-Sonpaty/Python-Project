#!/usr/bin/env python
# coding: utf-8

# In[1]:


#Importing Python libararies
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


# In[2]:


#Reading the data
df=pd.read_csv('C:/Suicide.csv')


# In[3]:


##Checking the data
df.info()
df.head()
df.describe()


# In[4]:


#Drop the unnecessary data
df.drop(['country-year', 'HDI for year'], axis = 1, inplace = True)


# In[5]:


df.columns


# In[6]:


#Rename the columns and change GDP golun to float
df.columns = ['country', 'year', 'sex', 'age', 'suicides', 'population', 'suicides/100k', 'gdp_year', 'gdp_capita', 'generation']
df['gdp_year'] = df.gdp_year.str.replace(',','').astype('float')


# In[7]:


df.info
df.describe()


# In[8]:


#Drop the rows which has no suicides
df = df[df['suicides/100k'] != 0]


# In[9]:


#Reordering the ages by categories exist
df.age = df.age.astype('category').cat.set_categories(['5-14 years', '15-24 years', '25-34 years', '35-54 years', '55-74 years', '75+ years'], ordered = True)


# In[10]:


#Checking if there any differences between ages groups,gender, and suicide number
sns.barplot(x = 'age', y = 'suicides/100k', hue = 'sex', data = df)


# In[11]:


#Let's check the suicide rates differences per gdp_Capita
sns.scatterplot(x = 'gdp_capita', y ='suicides/100k' , data = df)


# In[12]:


#Let's look on the suicide rates in countries individually
plt.figure(figsize = (10,20))
sns.barplot(x = 'suicides/100k', y = 'country', data = df)


# In[13]:


#Suicide rates over the years and how they affect the gender
plt.figure(figsize = (10,5))
ax = sns.lineplot(x = 'year', y = 'suicides/100k', hue = 'sex',  data = df)


# In[14]:


#Let's see the relation between gdpand suicide rates per year relation
fig, axes = plt.subplots(1,2 ,figsize=(10, 5))

sns.lineplot(x = 'year', y = 'gdp_capita', data = df, ax = axes[0])
sns.lineplot(x = 'year', y = 'suicides/100k', data = df, ax = axes[1])


# In[15]:


#Now we will build a machin learning model to predict the number of Suicides/100k.
#We will use random forest regressor model
#Random forest regressor is a technique capable of performing both regression and classification.
#with the use of multiple decision trees and a technique called Bootstrap and Aggregation.

#So now,we will need to transform the data so it will be ready for analysis.
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OrdinalEncoder
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import MinMaxScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.compose import make_column_selector as selector


# In[16]:



df.age = df.age.astype('object')
X = df.drop('suicides/100k', axis = 1)
y = df['suicides/100k']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state = 1)


# In[17]:



column_trans = ColumnTransformer(transformers=
        [('num', MinMaxScaler(), selector(dtype_exclude="object")),
        ('cat', OrdinalEncoder(), selector(dtype_include="object"))],
        remainder='drop')


# In[18]:


results = {}

for i in range(1,20):
    
    clf = RandomForestRegressor(random_state=42, max_depth = i)

    pipeline = Pipeline([('prep',column_trans),
                         ('clf', clf)])
    pipeline.fit(X_train, y_train)
    score = pipeline.score(X_test, y_test)
    results[i] = score


# In[19]:


results


# In[20]:



plt.plot(*zip(*sorted(results.items())))


# In[21]:


clf = RandomForestRegressor(random_state=42, max_depth = 10)
pipeline = Pipeline([('prep',column_trans),
                         ('clf', clf)])
pipeline.fit(X_train, y_train)


# In[22]:


pipeline['clf'].feature_importances_


# In[23]:


feature_list = []
targets = X.columns

#Print the name and gini importance of each feature

for feature in zip(targets, pipeline['clf'].feature_importances_):
    feature_list.append(feature)
 

df_imp = pd.DataFrame(feature_list, columns =['FEATURE', 'IMPORTANCE']).sort_values(by='IMPORTANCE', ascending=False)
df_imp['CUMSUM'] = df_imp['IMPORTANCE'].cumsum()

sns.barplot(x = 'IMPORTANCE', y = 'FEATURE', data = df_imp)


# In[ ]:




