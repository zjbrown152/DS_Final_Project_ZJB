#!/usr/bin/env python
# coding: utf-8

# In[1]:


import requests
import pandas as pd
import json
import glob


# In[5]:


df2=pd.read_json('https://api.themoviedb.org/3/genre/movie/list?api_key=c680984d1f261c766c61129ac1b932fa')
df2


# In[238]:


r1= requests.get('https://api.themoviedb.org/3/movie/latest?api_key=c680984d1f261c766c61129ac1b932fa')


# In[239]:


print(r1.status_code)


# In[240]:


x=r1.json()
x


# In[338]:


#Pulling data from TMDb, "Popular" section, and putting it into dataframe.
import requests
import pandas as pd

#Gets the data and puts it into a dictionary.
url = 'https://api.themoviedb.org/3/movie/popular?api_key=c680984d1f261c766c61129ac1b932fa'
response = requests.get(url)
json_data = response.json()

#The problem with json_data is it is a dictionary with one row - the value for that row is another
#dictionary. So we have to put that into separate dictionary of its own.
sub_dict = json_data["results"]

#Now we have the dictionary we want. We can convert into a dataframe.
df = pd.DataFrame.from_dict(sub_dict)


# In[339]:


#Selecting what info I want to include in a new dataframe.
newdf=df[['id','release_date','popularity','title','vote_average','vote_count','genre_ids']]
newdf=newdf.rename(columns={'id':'ID','genre_ids':'Genre'})


# In[340]:


newdf 


# In[341]:


##Pulling data from TMDb, "Top Rated" section, and putting it into dataframe.

import requests
import pandas as pd

#Gets the data and puts it into a dictionary.
url1 = 'https://api.themoviedb.org/3/movie/top_rated?api_key=c680984d1f261c766c61129ac1b932fa'
response1 = requests.get(url1)
json_data1 = response1.json()

#The problem with json_data is it is a dictionary with one row - the value for that row is another
#dictionary. So we have to put that into separate dictionary of its own.
sub_dict1 = json_data1["results"]

#Now we have the dictionary we want. We can convert into a dataframe.
df2 = pd.DataFrame.from_dict(sub_dict1)


# In[342]:


#Again selecting the data that I want from the df
newdf1=df2[['id','release_date','popularity','title','vote_average','vote_count','genre_ids']]
newdf1=newdf1.rename(columns={'id':'ID','genre_ids':'Genre'})
newdf1


# In[343]:


#Create a df for Budget and Revenue.
#Using ID from "Popular" df to retrieve the Budget and Revenue for the movie in TMDb.
import pandas as pd
import numpy as np
df3 = pd.DataFrame({'Budget': [],
                   'Revenue': [],})
for ID in newdf['ID']:  ##<-- you didn't have the ['ID] part.
    print(ID)
    url2 = 'https://api.themoviedb.org/3/movie/'+str(ID)+'?api_key=c680984d1f261c766c61129ac1b932fa'
    response2 = requests.get(url2)
    json_data2 = response2.json()
    list1=json_data2['budget']
    list2=json_data2['revenue']
    print('Budget '+str(list1)+'\nRevenue '+str(list2))
    df3.loc[len(df3)] = list1, list2


# In[344]:


#Join the df containing the "Popular" movies and the df with its corresponding budget/revenue.
final1=pd.concat([newdf, df3], axis=1, join='inner')
final1


# In[345]:


#Create a df for Budget and Revenue.
#Using ID from "Top Rated" df to retrieve the Budget and Revenue for the movie in TMDb.
import pandas as pd
import numpy as np
df4 = pd.DataFrame({'Budget': [],
                   'Revenue': [],})
for ID in newdf1['ID']: 
    print(ID)
    url3 = 'https://api.themoviedb.org/3/movie/'+str(ID)+'?api_key=c680984d1f261c766c61129ac1b932fa'
    response3 = requests.get(url3)
    json_data3 = response3.json()
    list3=json_data3['budget']
    list4=json_data3['revenue']
    print('Budget '+str(list3)+'\nRevenue '+str(list4))
    df4.loc[len(df4)] = list3, list4


# In[346]:


#Join the df containing the "Top Rated" movies and the df with its corresponding budget/revenue.

final2=pd.concat([newdf1, df4], axis=1, join='inner')
final2


# In[347]:


# Merging the dataframes for Popular and Top Rated movies into one df
final3 = final1.append(final2, ignore_index=True)
final3


# In[348]:


##Pulling data from TMDb, "Now Playing" section, and putting it into dataframe.

import requests
import pandas as pd

#Gets the data and puts it into a dictionary.
url4 = 'https://api.themoviedb.org/3/movie/now_playing?api_key=c680984d1f261c766c61129ac1b932fa'
response5 = requests.get(url4)
json_data5 = response5.json()

#The problem with json_data is it is a dictionary with one row - the value for that row is another
#dictionary. So we have to put that into separate dictionary of its own.
sub_dict3 = json_data5["results"]

#Now we have the dictionary we want. We can convert into a dataframe.
df5 = pd.DataFrame.from_dict(sub_dict3)


# In[349]:


newdf2=df5[['id','release_date','popularity','title','vote_average','vote_count','genre_ids']]
newdf2=newdf2.rename(columns={'id':'ID','genre_ids':'Genre'})
newdf2


# In[350]:


#Create a df for Budget and Revenue.
#Using ID from "Now Playing" df to retrieve the Budget and Revenue for the movie in TMDb.

import pandas as pd
import numpy as np
df6 = pd.DataFrame({'Budget': [],
                   'Revenue': [],})
for ID in newdf2['ID']: 
    print(ID)
    url6 = 'https://api.themoviedb.org/3/movie/'+str(ID)+'?api_key=c680984d1f261c766c61129ac1b932fa'
    response6 = requests.get(url6)
    json_data6 = response6.json()
    list5=json_data6['budget']
    list6=json_data6['revenue']
    print('Budget '+str(list5)+'\nRevenue '+str(list6))
    df6.loc[len(df6)] = list5, list6


# In[351]:


#Join the df containing the "Now Playing" movies and the df with its corresponding budget/revenue.

final5=pd.concat([newdf2, df6], axis=1, join='inner')
final5


# In[352]:


#Adding the Now Playing df to the df we created that already has the Popular and Top Rated movies.
final6 = final3.append(final5, ignore_index=True)
final6


# In[353]:


#Getting rid of any dupiclate movies from combining the list.
final7=final6.drop_duplicates(subset=['ID'])
final7


# In[354]:


final7.info()


# In[355]:


#Saving df as a csv and opening it.
final7.to_csv('finaldata', index=False)
lastdata=pd.read_csv('finaldata')
lastdata


# In[357]:


import pandas as pd
import csv
import sqlite3


# In[360]:


moviedata=pd.read_csv('finaldata')
moviedata


# In[367]:


#Connecting to sqlite database
connection= sqlite3.connect('/Users/15024/Desktop/movies.db')


# In[368]:


print('\nInserting Data to Database')
moviedata.to_sql(
    name= 'Movies',
    con=connection,
    if_exists= 'replace',
    index= False,
    dtype = {'ID':'integer',
            'release_date': 'text',
            'popularity': 'real',
            'title': 'text',
            'vote_average': 'real',
            'vote_count': 'integer',
            'Genre': 'text',
            'Budget': 'real',
            'Revenue': 'real'}
        )


# In[369]:


connection.commit()
connection.close()


# In[ ]:





# In[ ]:





# In[ ]:


final7['release_date'] = pd.to_datetime(final7['release_date'])

