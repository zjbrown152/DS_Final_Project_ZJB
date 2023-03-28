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


# In[ ]:


moviedata17.isnull().sum()


# In[ ]:


import statistics as stats
#Mean
print(stats.mean(moviedata17['popularity']))
print(stats.mean(moviedata17['vote_average']))
print(stats.mean(moviedata17['vote_count']))
print(stats.mean(moviedata17['Budget']))
print(stats.mean(moviedata17['Revenue']))
print(stats.mean(moviedata17['Runtime']))


# In[ ]:


#Standard Deviation
print(stats.stdev(moviedata17['popularity']))
print(stats.stdev(moviedata17['vote_average']))
print(stats.stdev(moviedata17['vote_count']))
print(stats.stdev(moviedata17['Budget']))
print(stats.stdev(moviedata17['Revenue']))
print(stats.stdev(moviedata17['Runtime']))


# In[ ]:


print(moviedata17['popularity'].describe())


# In[ ]:


print(moviedata17['vote_average'].describe())
print(moviedata17['vote_count'].describe())


# In[ ]:


print(moviedata17['Budget'].describe())
print(moviedata17['Revenue'].describe())


# In[ ]:


print(moviedata17['Runtime'].describe())


# In[ ]:


print(moviedata17.value_counts(moviedata17['name'])/len(moviedata17))


# In[ ]:


moviedata17['release_date'] = pd.to_datetime(moviedata17['release_date'], format= '%Y/%m/%d')


# In[ ]:


month_counts = moviedata17['release_date'].dt.month.value_counts()


# In[ ]:


print(month_counts/len(moviedata17))


# In[ ]:


moviedata17.info()


# In[ ]:


corr_matrix = moviedata17[['popularity', 'vote_average', 'vote_count','Budget','Revenue','Runtime']].corr()

print(corr_matrix)


# In[ ]:


import seaborn as sns
heat_map=moviedata17[['popularity','vote_average','vote_count','Budget','Revenue','Runtime']].corr(method='pearson')
cols=['Budget','Revenue','Runtime']
axis=sns.heatmap(heat_map, annot=True)


# In[ ]:


scatter_plot=sns.scatterplot(x='Runtime',
                             y='Revenue', 
                             data=moviedata17,
                            color='purple')


# In[ ]:


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


sns.distplot(moviedata17.)


# In[ ]:


sns.distplot(moviedata17.popularity)


# In[ ]:


pair_plot=sns.pairplot(moviedata17)


# # Predictive Model Logistic and Multiple Linear Regression

# In[39]:


#for ML regression
X=moviedata19[['release_date','popularity', 'vote_average', 'vote_count', 'Budget', 'Runtime','name']]
y=moviedata19[['Revenue']]


# In[40]:


genre_dummies=pd.get_dummies(moviedata19['name'])
X1=pd.concat([X,genre_dummies], axis=1)
X1.pop('name')
X1


# In[41]:


moviedata19['release_date'] = pd.to_datetime(moviedata19['release_date'])


# In[42]:


import pandas as pd
month_dummies = pd.get_dummies(moviedata19['release_date'].dt.month, prefix='month')

# concatenate the original DataFrame with the month dummy variables
X2 = pd.concat([X1, month_dummies], axis=1)

X2.pop('release_date')
# print the resulting DataFrame
X2


# In[125]:


#for logistic regression
X99=moviedata18[['release_date','popularity', 'vote_average', 'vote_count', 'Budget', 'Runtime','name']]
y99=moviedata18[['Profit']]


# In[126]:


X99


# In[127]:


genre_dummies=pd.get_dummies(moviedata18['name'])
X100=pd.concat([X99,genre_dummies], axis=1)
X100.pop('name')
X100


# In[128]:


import pandas as pd
month_dummies = pd.get_dummies(moviedata18['release_date'].dt.month, prefix='month')

# concatenate the original DataFrame with the month dummy variables
X200 = pd.concat([X100, month_dummies], axis=1)


# print the resulting DataFrame
X200


# In[129]:


X200.pop('release_date')


# In[130]:


X200


# In[250]:


#Logistic Regression for Profit
from sklearn.model_selection import train_test_split
X_train,X_test, y_train,y_test=train_test_split(X200,y99, test_size=.2, random_state=0)


# In[251]:


len(X_test)


# In[252]:


X_train,x_val,y_train,y_val=train_test_split(X_test,y_test, test_size=.5, random_state=0)


# In[253]:


len(x_val)


# In[255]:


len(y_val)


# In[259]:


from sklearn.preprocessing import StandardScaler
sc=StandardScaler()
X_train=sc.fit_transform(X_train)
X_test=sc.fit_transform(X_test)


# In[265]:


from sklearn.linear_model import LogisticRegression
classifier=LogisticRegression(random_state=0)
classifier.fit(x_val,y_val)


# In[272]:


y_pred=classifier.predict(x_val)


# In[273]:


y_pred


# In[274]:


len(y_pred)


# In[275]:


from sklearn.metrics import confusion_matrix, accuracy_score, classification_report,ConfusionMatrixDisplay
cm=confusion_matrix(y_val,y_pred)
print(cm)


# In[276]:


import matplotlib.pyplot as plt
disp=ConfusionMatrixDisplay(confusion_matrix=cm,
                           display_labels=classifier.classes_)
disp.plot()
plt.show()
#177 is True negative
#535 is true positive
#86 is false positive=didnt make profit, predicted to
#2 is false negative=made profit, predicted not to


# In[270]:


accuracy_score(y_test,y_pred)


# In[49]:



#Multiple Linear
from sklearn.model_selection import train_test_split
X_train,X_test, y_train,y_test=train_test_split(X2,y, test_size=.3, random_state=0)


# In[50]:


from sklearn.linear_model import LinearRegression #run from the top if it doesnt work.
regressor=LinearRegression()
regressor.fit(X_train,y_train)


# In[51]:


print('Coefficients', regressor.coef_)
print('Intercept', regressor.intercept_)


# In[52]:


y_pred1=regressor.predict(X_test)


# In[53]:


from sklearn.metrics import r2_score
print('r2-score', r2_score(y_test, y_pred1))


# In[54]:


y_test


# In[55]:


y_pred1


# In[79]:


#decision tree
X4=moviedata18[['release_date','popularity', 'vote_average', 'vote_count', 'Budget', 'Runtime','name']]
y4=moviedata18[['Profit']]


# In[80]:


y4


# In[ ]:





# In[81]:


genre_dummies=pd.get_dummies(moviedata18['name'])
X5=pd.concat([X4,genre_dummies], axis=1)
X5.pop('name')
X5


# In[82]:


import pandas as pd
month_dummies = pd.get_dummies(moviedata18['release_date'].dt.month, prefix='month')

# concatenate the original DataFrame with the month dummy variables
X6 = pd.concat([X5, month_dummies], axis=1)


# print the resulting DataFrame
X6


# In[83]:


X6.pop('release_date')


# In[84]:


X6


# In[ ]:





# In[93]:


from sklearn.tree import DecisionTreeRegressor
from sklearn.tree import plot_tree #tree diagram
regressor=DecisionTreeRegressor(random_state=0)
regressor.fit(X6,y4)


# In[94]:


y_pred2=regressor.predict(X6)


# In[ ]:





# In[95]:


ypred2=pd.DataFrame(y_pred2)
ypred2.astype('int')


# In[88]:


y4


# In[89]:


regressor = DecisionTreeRegressor(max_depth=3)
regressor.fit(X6, y4)  # Assuming you have X (input features) and y (target variable)

# Visualize the decision tree
fig, ax = plt.subplots(figsize=(10, 10))
plot_tree(regressor, feature_names=X.columns, ax=ax, filled=True, rounded=True)
plt.show()

fig, ax = plt.subplots(figsize=(10, 10))
plot_tree(regressor, feature_names=X.columns, ax=ax, filled=True, rounded=True)
plt.show()


# In[90]:


from sklearn.metrics import mean_squared_error as msc
from math import sqrt
rms=sqrt(msc(y4,y_pred2))
print('RMSE= ',rms)


# In[91]:


accuracy_score(y4,y_pred2)


# In[92]:


y_pred2


# In[96]:


X9=moviedata18[['release_date','popularity', 'vote_average', 'vote_count', 'Budget', 'Runtime','name']]
y9=moviedata18[['Profit']]


# In[97]:


genre_dummies=pd.get_dummies(moviedata18['name'])
X10=pd.concat([X9,genre_dummies], axis=1)
X10.pop('name')
X10


# In[98]:


import pandas as pd
month_dummies = pd.get_dummies(moviedata18['release_date'].dt.month, prefix='month')

# concatenate the original DataFrame with the month dummy variables
X11 = pd.concat([X10, month_dummies], axis=1)

X11.pop('release_date')
# print the resulting DataFrame
X11


# In[209]:


from sklearn.ensemble import RandomForestRegressor
regressor=RandomForestRegressor(n_estimators=20, random_state=0)
regressor.fit(X11,y9)


# In[224]:


y_pred3=regressor.predict(X11)


# In[225]:


from sklearn.metrics import r2_score
print('r2-score', r2_score(y9, y_pred3))


# In[226]:


from sklearn.metrics import mean_squared_error as msc
from math import sqrt
rms=sqrt(msc(y9,y_pred3))
print('RMSE= ',rms) 


# In[227]:


y_pred3


# In[223]:


y9


# In[ ]:


type(X6)


# In[ ]:


import sys
import platform
import imp

print("Python EXE     : " + sys.executable)


# In[ ]:




