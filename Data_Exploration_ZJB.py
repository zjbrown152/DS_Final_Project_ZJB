#!/usr/bin/env python
# coding: utf-8

# In[134]:


import pandas as pd

moviedata17=pd.read_csv('finaldata')
moviedata17


# In[104]:


moviedata18=pd.read_csv('finaldata')
moviedata18


# In[105]:


#allows us to use logistic regression
moviedata18['Profit'] = (moviedata18['Revenue'] - moviedata18['Budget']) > 0
moviedata18['Profit'] = moviedata18['Profit'].astype(int)


# In[106]:


moviedata18


# In[107]:


print(moviedata18.value_counts(moviedata17['Profit']))


# In[2]:


moviedata17.isnull().sum()


# In[3]:


import statistics as stats
#Mean
print(stats.mean(moviedata17['popularity']))
print(stats.mean(moviedata17['vote_average']))
print(stats.mean(moviedata17['vote_count']))
print(stats.mean(moviedata17['Budget']))
print(stats.mean(moviedata17['Revenue']))
print(stats.mean(moviedata17['Runtime']))


# In[4]:


#Standard Deviation
print(stats.stdev(moviedata17['popularity']))
print(stats.stdev(moviedata17['vote_average']))
print(stats.stdev(moviedata17['vote_count']))
print(stats.stdev(moviedata17['Budget']))
print(stats.stdev(moviedata17['Revenue']))
print(stats.stdev(moviedata17['Runtime']))


# In[5]:


print(moviedata17['popularity'].describe())


# In[6]:


print(moviedata17['vote_average'].describe())
print(moviedata17['vote_count'].describe())


# In[7]:


print(moviedata17['Budget'].describe())
print(moviedata17['Revenue'].describe())


# In[8]:


print(moviedata17['Runtime'].describe())


# In[5]:


print(moviedata17.value_counts(moviedata17['name'])/len(moviedata17))


# In[6]:


moviedata17['release_date'] = pd.to_datetime(moviedata17['release_date'], format= '%Y/%m/%d')


# In[7]:


month_counts = moviedata17['release_date'].dt.month.value_counts()


# In[11]:


print(month_counts/len(moviedata17))


# In[12]:


moviedata17.info()


# In[13]:


corr_matrix = moviedata17[['popularity', 'vote_average', 'vote_count','Budget','Revenue','Runtime']].corr()

print(corr_matrix)


# In[14]:


import seaborn as sns
heat_map=moviedata17[['popularity','vote_average','vote_count','Budget','Revenue','Runtime']].corr(method='pearson')
cols=['Budget','Revenue','Runtime']
axis=sns.heatmap(heat_map, annot=True)


# In[46]:


scatter_plot=sns.scatterplot(x='Runtime',
                             y='Revenue', 
                             data=moviedata17,
                            color='purple')


# In[84]:


moviedata17['month'] = moviedata17['release_date'].dt.month_name()
month_order = ['January', 'February', 'March', 'April', 'May', 'June',
               'July', 'August', 'September', 'October', 'November', 'December']
bar_plot=sns.barplot(x='month', y='Revenue', data=moviedata17, order=month_order, ci=False)
plt.xticks(rotation=40, fontsize=8)
plt.title('Revenue by Month')
plt.figure(figsize=(30,5))


# In[73]:


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


# In[80]:


import matplotlib.pyplot as plt
bar_plot=sns.barplot(x='name', y='Revenue', data=moviedata17, ci=False)
plt.xticks(rotation=40, fontsize=8)
plt.title('Revenue by Genre')
plt.figure(figsize=(30,5))


# In[86]:


scatter_plot=sns.scatterplot(x='vote_count',
                             y='Revenue', 
                             data=moviedata17,
                            color='red')
plt.title('Vote Count vs Revenue')


# In[85]:


scatter_plot1=sns.scatterplot(x='Budget',
                             y='Revenue', 
                             data=moviedata17,
                            color='blue')
plt.title('Budget vs Revenue')


# In[17]:


lm_plot=sns.lmplot(x='Budget',
                   y='Revenue',
                   data=moviedata17)


# In[50]:


box_plot=sns.boxplot(moviedata17['Runtime'].values)


# In[74]:


sns.distplot(moviedata17.vote_count)


# In[20]:


sns.distplot(moviedata17.Budget)


# In[21]:


sns.distplot(moviedata17.Revenue)


# In[87]:


sns.distplot(moviedata17.vote_average)
plt.title('Distribution of Vote Average')


# In[18]:


sns.distplot(moviedata17.Runtime)


# In[135]:


pair_plot=sns.pairplot(moviedata17)


# # Predictive Model Logistic and Multiple Linear Regression

# In[ ]:


X=moviedata17[['release_date','popularity', 'vote_average', 'vote_count', 'Budget', 'Runtime','Revenue']]
y=moviedata17[['Revenue']]


# In[108]:


#for logistic regression
X=moviedata18[['release_date','popularity', 'vote_average', 'vote_count', 'Budget', 'Runtime','Revenue']]
y=moviedata18[['Profit']]


# In[109]:


X


# In[ ]:





# In[110]:


month_dummies = pd.get_dummies(moviedata17['release_date'].dt.month, prefix='month')

# concatenate the original DataFrame with the month dummy variables
X1 = pd.concat([X, month_dummies], axis=1)


# print the resulting DataFrame
X1


# In[111]:


X1.pop('release_date')


# In[112]:


X1


# In[120]:


#Logistic Regression for Profit
from sklearn.model_selection import train_test_split
X_train,X_test, y_train,y_test=train_test_split(X1,y, test_size=.3, random_state=0)


# In[121]:


from sklearn.preprocessing import StandardScaler
sc=StandardScaler()
X_train=sc.fit_transform(X_train)
X_test=sc.fit_transform(X_test)


# In[122]:


from sklearn.linear_model import LogisticRegression
classifier=LogisticRegression(random_state=0)
classifier.fit(X_train,y_train)


# In[123]:


y_pred=classifier.predict(X_test)


# In[124]:


from sklearn.metrics import confusion_matrix, accuracy_score, classification_report,ConfusionMatrixDisplay
cm=confusion_matrix(y_test,y_pred)
print(cm)


# In[125]:


disp=ConfusionMatrixDisplay(confusion_matrix=cm,
                           display_labels=classifier.classes_)
disp.plot()
plt.show()
#177 is True negative
#535 is true positive
#86 is false positive=didnt make profit, predicted to
#2 is false negative=made profit, predicted not to


# In[126]:


accuracy_score(y_test,y_pred)


# In[39]:



#Multiple Linear
from sklearn.model_selection import train_test_split
X_train,X_test, y_train,y_test=train_test_split(X1,y, test_size=.3, random_state=0)


# In[40]:


from sklearn.linear_model import LinearRegression #run from the top if it doesnt work.
regressor=LinearRegression()
regressor.fit(X_train,y_train)


# In[41]:


print('Coefficients', regressor.coef_)
print('Intercept', regressor.intercept_)


# In[42]:


y_pred=regressor.predict(X_test)


# In[44]:


from sklearn.metrics import r2_score
print('r2-score', r2_score(y_test, y_pred))


# In[13]:


import sys
import platform
import imp

print("Python EXE     : " + sys.executable)


# In[ ]:




