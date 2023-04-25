#!/usr/bin/env python
# coding: utf-8

# In[26]:


import pandas as pd

moviedata17=pd.read_csv('finaldata')
moviedata17


# In[27]:


moviedata18=pd.read_csv('finaldata')
moviedata18


# In[28]:


moviedata19=pd.read_csv('finaldata')
moviedata19


# In[29]:


moviedata19['Profits'] = (moviedata19['Revenue'] - moviedata19['Budget'])
moviedata19['Profits'] = moviedata19['Profits'].astype(int)


# In[30]:


moviedata19


# In[31]:


moviedata19.median() #8730891


# In[32]:


#allows us to use logistic regression
moviedata18['Profit'] = (moviedata18['Revenue'] - moviedata18['Budget']) > 8730891
moviedata18['Profit'] = moviedata18['Profit'].astype(int)


# In[33]:


moviedata18


# In[34]:


moviedata18['release_date'] = pd.to_datetime(moviedata18['release_date'])


# In[35]:


moviedata18.info()


# In[36]:


print(moviedata18.value_counts(moviedata18['Profit']))


# # Predictive Model Logistic and Multiple Linear Regression Etc..

# In[37]:


#for ML regression
X=moviedata19[['release_date','popularity', 'vote_average', 'vote_count', 'Budget', 'Runtime','name']]
y=moviedata19[['Revenue']]


# In[38]:


genre_dummies=pd.get_dummies(moviedata19['name'])
X1=pd.concat([X,genre_dummies], axis=1)
X1.pop('name')
X1


# In[39]:


moviedata19['release_date'] = pd.to_datetime(moviedata19['release_date'])


# In[40]:


import pandas as pd
month_dummies = pd.get_dummies(moviedata19['release_date'].dt.month, prefix='month')

# concatenate the original DataFrame with the month dummy variables
X2 = pd.concat([X1, month_dummies], axis=1)

X2.pop('release_date')
# print the resulting DataFrame
X2


# In[41]:


#for logistic regression
X99=moviedata18[['release_date','popularity', 'vote_average', 'vote_count', 'Budget', 'Runtime','name']]
y99=moviedata18[['Profit']]


# In[42]:


X99


# In[43]:


genre_dummies=pd.get_dummies(moviedata18['name'])
X100=pd.concat([X99,genre_dummies], axis=1)
X100.pop('name')
X100


# In[44]:


import pandas as pd
month_dummies = pd.get_dummies(moviedata18['release_date'].dt.month, prefix='month')

# concatenate the original DataFrame with the month dummy variables
X200 = pd.concat([X100, month_dummies], axis=1)


# print the resulting DataFrame
X200


# In[45]:


X200.pop('release_date')


# In[46]:


X200


# In[47]:


#logistic
from sklearn.model_selection import train_test_split
X_train,X_test, y_train,y_test=train_test_split(X200,y99, test_size=.35, random_state=0)
X_test,X_val,y_test,y_val=train_test_split(X_test,y_test, test_size=.5, random_state=0)


# In[48]:


len(X_test)


# In[49]:


len(X_val)


# In[50]:


len(y_val)


# In[51]:


from sklearn.preprocessing import StandardScaler
sc=StandardScaler()
X_train=sc.fit_transform(X_train)
X_test=sc.fit_transform(X_test)
X_val=sc.fit_transform(X_val)


# In[52]:


from sklearn.linear_model import LogisticRegression
classifier=LogisticRegression(random_state=0)
classifier.fit(X_train,y_train)


# In[53]:


y_predict=classifier.predict(X_train)


# In[54]:


y_pred=classifier.predict(X_val)


# In[55]:


y_pred


# In[56]:


y_val


# In[57]:


len(y_pred)


# In[58]:


from sklearn.metrics import confusion_matrix, accuracy_score, classification_report,ConfusionMatrixDisplay
cm3=confusion_matrix(y_train,y_predict)
print(cm3)


# In[59]:


from sklearn.metrics import confusion_matrix, accuracy_score, classification_report,ConfusionMatrixDisplay
cm=confusion_matrix(y_val,y_pred)
print(cm)


# In[60]:


import matplotlib.pyplot as plt
disp2=ConfusionMatrixDisplay(confusion_matrix=cm3,
                           display_labels=classifier.classes_)
disp2.plot()
plt.show()
#177 is True negative
#535 is true positive
#86 is false positive=didnt make profit, predicted to
#2 is false negative=made profit, predicted not to


# In[61]:


import matplotlib.pyplot as plt
disp=ConfusionMatrixDisplay(confusion_matrix=cm,
                           display_labels=classifier.classes_)
disp.plot()
plt.show()
#287 is True negative
#231 is true positive
#52 is false positive=didnt make profit, predicted to
#130 is false negative=made profit, predicted not to


# In[62]:


accuracy_score(y_train, y_predict)


# In[63]:


accuracy_score(y_val,y_pred)
#correctly predicted 74% of the validation points (new data)


# In[ ]:



#Multiple Linear
from sklearn.model_selection import train_test_split
X_train,X_test, y_train,y_test=train_test_split(X2,y, test_size=.3, random_state=0)


# In[37]:


from sklearn.model_selection import train_test_split
X_train1,X_test1, y_train1,y_test1=train_test_split(X2,y, test_size=.3, random_state=0)
X_test1,X_val1,y_test1,y_val1=train_test_split(X_test1,y_test1, test_size=.5, random_state=0)


# In[38]:


y_train1


# In[39]:


from sklearn.linear_model import LinearRegression #run from the top if it doesnt work.
regressor=LinearRegression()
regressor.fit(X_train1,y_train1)


# In[40]:


print('Coefficients', regressor.coef_)
print('Intercept', regressor.intercept_)


# In[41]:


y_pred1=regressor.predict(X_val1)


# In[42]:


from sklearn.metrics import r2_score
print('r2-score', r2_score(y_val1, y_pred1))
#69% of the data can be explained by the model


# In[62]:


from sklearn.metrics import mean_squared_error as msc
from math import sqrt
rms=sqrt(msc(y_val1,y_pred1))
print('RMSE= ',rms) 


# In[43]:


y_test


# In[44]:


y_pred1


# In[45]:


#decision tree
X4=moviedata18[['release_date','popularity', 'vote_average', 'vote_count', 'Budget', 'Runtime','name']]
y4=moviedata18[['Profit']]


# In[46]:


y4


# In[ ]:





# In[47]:


genre_dummies=pd.get_dummies(moviedata18['name'])
X5=pd.concat([X4,genre_dummies], axis=1)
X5.pop('name')
X5


# In[48]:


import pandas as pd
month_dummies = pd.get_dummies(moviedata18['release_date'].dt.month, prefix='month')

# concatenate the original DataFrame with the month dummy variables
X6 = pd.concat([X5, month_dummies], axis=1)


# print the resulting DataFrame
X6


# In[49]:


X6.pop('release_date')


# In[50]:


X6


# In[51]:


from sklearn.model_selection import train_test_split
X_train2,X_test2, y_train2,y_test2=train_test_split(X6,y4, test_size=.2, random_state=0)
X_test2,X_val2,y_test2,y_val2=train_test_split(X_test2,y_test2, test_size=.5, random_state=0)


# In[52]:


from sklearn.preprocessing import StandardScaler
sc=StandardScaler()
X_train2=sc.fit_transform(X_train2)
X_test2=sc.fit_transform(X_test2)
X_val2=sc.fit_transform(X_val2)


# In[ ]:





# In[53]:


from sklearn.tree import DecisionTreeRegressor
from sklearn.tree import plot_tree #tree diagram
regressor=DecisionTreeRegressor(random_state=0)
regressor.fit(X_train2,y_train2)


# In[54]:


y_pred2=regressor.predict(X_val2)


# In[55]:


y_pred2


# In[56]:


ypred2=pd.DataFrame(y_pred2)
ypred2.astype('int')


# In[57]:


y_val2


# In[58]:


from sklearn.metrics import mean_squared_error as msc
from math import sqrt
rms=sqrt(msc(y_val2,y_pred2))
print('RMSE= ',rms)


# In[59]:


accuracy_score(y_val2,y_pred2)


# In[577]:


#moviedata19 predicts revenue or profits value
#moviedata18 predicts if profit is greater than median classifier regressionD
#Random forrest
X9=moviedata18[['release_date','popularity', 'vote_average', 'vote_count', 'Budget', 'Runtime','name']]
y9=moviedata18[['Profit']]


# In[578]:


genre_dummies=pd.get_dummies(moviedata18['name'])
X10=pd.concat([X9,genre_dummies], axis=1)
X10.pop('name')
X10


# In[579]:


import pandas as pd
month_dummies = pd.get_dummies(moviedata18['release_date'].dt.month, prefix='month')

# concatenate the original DataFrame with the month dummy variables
X11 = pd.concat([X10, month_dummies], axis=1)

X11.pop('release_date')
# print the resulting DataFrame
X11


# In[580]:


from sklearn.model_selection import train_test_split
X_train3,X_test3, y_train3,y_test3=train_test_split(X11,y9, test_size=.2, random_state=0)
X_test3,X_val3,y_test3,y_val3=train_test_split(X_test3,y_test3, test_size=.5, random_state=0)


# In[581]:


from sklearn.preprocessing import StandardScaler
sc=StandardScaler()
X_train3=sc.fit_transform(X_train3)
X_test3=sc.fit_transform(X_test3)
X_val3=sc.fit_transform(X_val3)


# In[568]:


from sklearn.ensemble import RandomForestRegressor
regressor4=RandomForestRegressor(n_estimators=20, random_state=0) #try more estimators, check rmse
regressor4.fit(X_train3,y_train3) 


# In[589]:


from sklearn.ensemble import RandomForestClassifier
regressor6=RandomForestClassifier(n_estimators=100, random_state=0)
regressor6.fit(X_train3,y_train3) 


# In[586]:


ypred984=regressor4.predict(X_train3)


# In[590]:


ypredict8=regressor6.predict(X_val3)


# In[570]:


y_pred5=regressor4.predict(X_val3)


# In[571]:


y_pred5


# In[591]:


accuracy_score(y_val3,ypredict8)


# In[572]:


from sklearn.metrics import r2_score
print('r2-score', r2_score(y_val3, y_pred5))


# In[573]:


from sklearn.metrics import r2_score
print('r2-score', r2_score(y_train3, ypred984))


# In[574]:


from sklearn.metrics import mean_squared_error as msc
from math import sqrt
rms=sqrt(msc(y_val3,y_pred5))
print('RMSE= ',rms) 


# In[417]:


from sklearn.metrics import mean_squared_error as msc
from math import sqrt
rms=sqrt(msc(y_val3,y))
print('RMSE= ',rms) 


# In[395]:


len(y_pred5)


# In[541]:


y_pred5


# In[543]:


y_val3


# In[ ]:


type(X6)


# In[ ]:


import sys
import platform
import imp

print("Python EXE     : " + sys.executable)


# In[ ]:




