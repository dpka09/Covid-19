#!/usr/bin/env python
# coding: utf-8

# In[1]:


import sys
print(sys.version)


# In[2]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# In[3]:


df= pd.read_csv("covid19.csv")


# In[4]:


#set(df['positive_rate'].values)


# In[5]:


df.dtypes


# In[6]:


df["new_cases"].isna().sum()


# In[7]:


df["new_cases"].sum()


# In[8]:


c=df["new_cases"]==0


# In[9]:


c.sum()


# In[10]:


1643+7682


# In[11]:


#!pip install pandas-profilling


# In[12]:


#import pandas_profiling


# In[13]:


#df.profile_report(progress_bar=False)


# In[ ]:





# In[14]:


bucket=["iso_code","new_cases","new_deaths","median_age","aged_70_older","new_tests","new_cases_per_million","new_deaths_per_million"]


# In[15]:


bucket_drop=df.drop(bucket,axis=1,inplace=True)


# In[16]:


data=df.iloc[0:49512]


# In[17]:


#data


# In[18]:


data.isna().sum()


# In[19]:


extra=["total_cases_per_million","new_cases_smoothed_per_million","total_deaths_per_million","new_deaths_smoothed_per_million","total_tests_per_thousand","new_tests_per_thousand","new_tests_smoothed_per_thousand"]


# In[20]:


extraa=["tests_per_case","tests_units","population_density"]


# In[21]:


data.drop(extraa, axis=1, inplace=True)


# In[22]:


misc= ["positive_rate","stringency_index","cardiovasc_death_rate","diabetes_prevalence","female_smokers","male_smokers","handwashing_facilities","hospital_beds_per_thousand"]


# In[23]:


data[misc]=data[misc].fillna(0)


# In[24]:


data["aged_65_older"]=data["aged_65_older"].fillna(0)


# In[25]:


data["extreme_poverty"]= data["extreme_poverty"].fillna(0)


# In[26]:


data["aged_65_older"].isna().sum()


# In[27]:


filll=["gdp_per_capita","life_expectancy","human_development_index"]


# In[28]:


data[filll]= data[filll].fillna(method='ffill').fillna(0)


# In[29]:


#data[misc]


# In[30]:


new_columns = data.columns[list(map(lambda x:"new" in x,data.columns))]


# In[31]:


data[new_columns]=data[new_columns].fillna(0)


# In[32]:


#data[new_columns]


# In[33]:


#data["new_cases_smoothed"] = data["new_cases_smoothed"].fillna(0)


# In[34]:


"total" in data.columns[3]


# In[35]:


total_columns = data.columns[list(map(lambda x:"total" in x,data.columns))]


# In[36]:


total_columns


# In[37]:


data[total_columns] = data[total_columns].fillna(method='ffill').fillna(0)


# In[38]:


sns.lineplot(data=data, x="continent", y="total_cases")
plt.show()


# In[39]:


sns.lineplot(data=data, x="date", y="total_cases")
plt.show()


# In[40]:


#dataa=["total_cases","new_cases_smoothed","total_deaths","new_deaths_smoothed","total_tests""new_tests_smoothed"]


# In[41]:


#data[dataa].max()


# In[42]:


data["total_deaths"].max()


# In[43]:


#fig = plt.figure(figsize =(5, 7)) 
#plt.pie(data[dataa].mean(), labels = dataa) 


# In[44]:


locate=data.groupby('location').max().sort_values('total_deaths',ascending=False)[:10]


# In[45]:


locate


# In[46]:


locate.reset_index(inplace=True)


# In[47]:


sns.barplot(x="location", y="total_deaths", data=locate[:10])
plt.xticks(rotation="45")
plt.show()


# In[48]:


sns.barplot(x="location", y="total_cases", data=locate[:10])
plt.xticks(rotation="45")
plt.show()


# In[49]:


#data[data['location']=='Nepal'].plot(x='date',y='new_cases_smoothed')


# In[50]:


place=["continent", "location","date"]


# In[51]:


a=data.drop(place, axis=1)


# In[52]:





# In[60]:


from sklearn.preprocessing import StandardScaler


# In[61]:


scaler= StandardScaler()


# In[81]:


scaled_a=scaler.fit_transform(a)


# In[90]:


data.drop(["total_cases_per_million","new_cases_smoothed_per_million","total_deaths_per_million"],axis=1, inplace=True)


# In[58]:


#sns.lineplot(x="place", y="total_deaths", data=data)
#sns.lineplot(x="place",y="total_cases", data=data)
#plt.legend(total_deaths, total_cases)
#plt.show()


# In[71]:


data


# In[72]:


data.drop(["new_deaths_smoothed_per_million","total_tests_per_thousand"], axis=1, inplace=True)


# In[73]:


data


# In[74]:


data.drop("new_tests_per_thousand", axis=1, inplace=True)


# In[75]:


a.drop(["new_tests_per_thousand","new_deaths_smoothed_per_million","total_tests_per_thousand","total_cases_per_million","new_cases_smoothed_per_million","total_deaths_per_million"], axis=1, inplace=True)


# In[77]:


a.drop("new_tests_smoothed_per_thousand",axis=1, inplace=True)


# In[79]:


a.dtypes


# In[83]:


data[a.columns]=scaled_a


# In[91]:


data.head()


# In[85]:


data.drop("new_tests_smoothed_per_thousand", axis=1, inplace=True)


# In[96]:


sns.lineplot(x="date", y="gdp_per_capita", data=data)
sns.lineplot(x="date", y="extreme_poverty", data=data)
plt.legend(['GDP per capita', 'extreme poverty'])
plt.show()


# In[100]:


sns.lineplot(x="date", y="cardiovasc_death_rate", data=data)
sns.lineplot(x="date", y="diabetes_prevalence", data=data)
plt.legend(['cardiovasc_death_rate','diabetes_prevalence'])
plt.show()


# In[97]:


sns.lineplot(x="date", y="total_cases", data=data)
sns.lineplot(x="date", y="total_deaths", data=data)
sns.lineplot(x="date", y="positive_rate", data=data)
plt.legend(['total_cases', 'total_deaths','positive rate'])
plt.show()


# In[105]:


sns.barplot(x = 'continent', y = 'positive_rate', data = data, palette = 'magma')
plt.title('Infected Rate')
plt.show()


# In[ ]:


sns.barplot(x = 'total_deaths', y = 'positive_rate', data = data, palette = 'magma')
plt.title('Infected Rate')
plt.show()


# In[ ]:




