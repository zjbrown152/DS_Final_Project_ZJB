#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd

moviedata17=pd.read_csv('finaldata')
moviedata17


# In[2]:


moviedata18=pd.read_csv('finaldata')
moviedata18


# In[3]:


moviedata19=pd.read_csv('finaldata')
moviedata19


# In[4]:


moviedata19['Profits'] = (moviedata19['Revenue'] - moviedata19['Budget'])
moviedata19['Profits'] = moviedata19['Profits'].astype(int)


# In[5]:


moviedata19


# In[6]:


moviedata19.median() #8730891


# In[7]:


#allows us to use logistic regression
moviedata18['Profit'] = (moviedata18['Revenue'] - moviedata18['Budget']) > 8730891
moviedata18['Profit'] = moviedata18['Profit'].astype(int)


# In[8]:


moviedata18


# In[9]:


moviedata18['release_date'] = pd.to_datetime(moviedata18['release_date'])


# In[10]:


moviedata18.info()


# In[11]:


print(moviedata18.value_counts(moviedata18['Profit']))


# In[12]:


import statistics as stats
#Mean
print(stats.mean(moviedata17['popularity']))
print(stats.mean(moviedata17['vote_average']))
print(stats.mean(moviedata17['vote_count']))
print(stats.mean(moviedata17['Budget']))
print(stats.mean(moviedata17['Revenue']))
print(stats.mean(moviedata17['Runtime']))


# In[13]:


#Standard Deviation
print(stats.stdev(moviedata17['popularity']))
print(stats.stdev(moviedata17['vote_average']))
print(stats.stdev(moviedata17['vote_count']))
print(stats.stdev(moviedata17['Budget']))
print(stats.stdev(moviedata17['Revenue']))
print(stats.stdev(moviedata17['Runtime']))


# In[14]:


print(moviedata17['popularity'].describe())


# In[15]:


print(moviedata17['vote_average'].describe())
print(moviedata17['vote_count'].describe())


# In[16]:


print(moviedata17['Budget'].describe())
print(moviedata17['Revenue'].describe())


# In[17]:


print(moviedata17['Runtime'].describe())


# In[18]:


print(moviedata17.value_counts(moviedata17['name'])/len(moviedata17))


# In[19]:


moviedata17['release_date'] = pd.to_datetime(moviedata17['release_date'], format= '%Y/%m/%d')


# In[20]:


month_counts = moviedata17['release_date'].dt.month.value_counts()


# In[21]:


print(month_counts/len(moviedata17))


# In[22]:


moviedata17.info()


# In[23]:


corr_matrix = moviedata17[['popularity', 'vote_average', 'vote_count','Budget','Revenue','Runtime']].corr()

print(corr_matrix)


# In[24]:


import seaborn as sns
heat_map=moviedata17[['popularity','vote_average','vote_count','Budget','Revenue','Runtime']].corr(method='pearson')
cols=['Budget','Revenue','Runtime']
axis=sns.heatmap(heat_map, annot=True)


# In[25]:


scatter_plot=sns.scatterplot(x='Runtime',
                             y='Revenue', 
                             data=moviedata17,
                            color='purple')


# In[26]:


moviedata17['month'] = moviedata17['release_date'].dt.month_name()
month_order = ['January', 'February', 'March', 'April', 'May', 'June',
               'July', 'August', 'September', 'October', 'November', 'December']
bar_plot=sns.barplot(x='month', y='Revenue', data=moviedata17, order=month_order, ci=False)
plt.xticks(rotation=40, fontsize=8)
plt.title('Revenue by Month')
plt.figure(figsize=(30,5))


# In[ ]:


moviedata17['month'] = moviedata17['release_date'].dt.month_name()
month_order = ['January', 'February', 'March', 'April', 'May', 'June',
               'July', 'August', 'September', 'October', 'November', 'December']
# Create a count plot of the number of movies in each month
sns.countplot(x='month', data=moviedata17, order=month_order)

# Add labels and title
plt.xticks(fontsize=5)
plt.xlabel('Month')
plt.ylabel('Number of Movies')
plt.title('Number of Movies by Month')


# In[ ]:


import matplotlib.pyplot as plt
bar_plot=sns.barplot(x='name', y='Revenue', data=moviedata17, ci=False)
plt.xticks(rotation=40, fontsize=8)
plt.title('Revenue by Genre')
plt.figure(figsize=(30,5))


# In[ ]:


scatter_plot=sns.scatterplot(x='vote_count',
                             y='Revenue', 
                             data=moviedata17,
                            color='red')
plt.title('Vote Count vs Revenue')


# In[ ]:


scatter_plot1=sns.scatterplot(x='Budget',
                             y='Revenue', 
                             data=moviedata17,
                            color='blue')
plt.title('Budget vs Revenue')


# In[ ]:


scatter_plot3=sns.scatterplot(x='Runtime',
                             y='Revenue', 
                             data=moviedata17,
                            color='orange')
plt.title('Runtime vs Revenue')


# In[ ]:


lm_plot=sns.lmplot(x='Budget',
                   y='Revenue',
                   data=moviedata17)


# In[ ]:


import seaborn as sns
plt.figure(figsize=(15,5))
box_plot=sns.boxplot(moviedata17['Budget'].values)
plt.show()


# In[ ]:


import seaborn as sns
plt.figure(figsize=(15,5))
box_plot=sns.boxplot(moviedata17['Revenue'].values)
plt.title('Box Plot of Revenue Values')
plt.show()


# In[ ]:


plt.figure(figsize=(15,5))
box_plot=sns.boxplot(moviedata19['Profits'].values)


# In[ ]:


plt.figure(figsize=(15,5))
sns.distplot(moviedata19.Profits)


# In[ ]:


sns.distplot(moviedata17.vote_count)
plt.title('Distribution of Vote Counts')


# In[ ]:


sns.distplot(moviedata17.Budget)


# In[ ]:


sns.distplot(moviedata17.Revenue)
plt.title("Distribution of Revenues")


# In[ ]:


sns.distplot(moviedata17.vote_average)
plt.title('Distribution of Vote Average')


# In[ ]:




