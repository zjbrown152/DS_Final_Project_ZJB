#!/usr/bin/env python
# coding: utf-8

# In[84]:


import requests
import pandas as pd
import json
import glob


# In[85]:


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


# In[86]:


#Selecting what info I want to include in a new dataframe.
newdf=df[['id','release_date','popularity','title','vote_average','vote_count','genre_ids']]
newdf=newdf.rename(columns={'id':'ID','genre_ids':'Genre'})


# In[87]:


newdf 


# In[88]:


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


# In[89]:


#Again selecting the data that I want from the df
newdf1=df2[['id','release_date','popularity','title','vote_average','vote_count','genre_ids']]
newdf1=newdf1.rename(columns={'id':'ID','genre_ids':'Genre'})
newdf1


# In[90]:


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


# In[91]:


#Join the df containing the "Popular" movies and the df with its corresponding budget/revenue.
final1=pd.concat([newdf, df3], axis=1, join='inner')
final1


# In[92]:


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


# In[93]:


#Join the df containing the "Top Rated" movies and the df with its corresponding budget/revenue.

final2=pd.concat([newdf1, df4], axis=1, join='inner')
final2


# In[94]:


# Merging the dataframes for Popular and Top Rated movies into one df
final3 = final1.append(final2, ignore_index=True)
final3


# In[95]:


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


# In[96]:


newdf2=df5[['id','release_date','popularity','title','vote_average','vote_count','genre_ids']]
newdf2=newdf2.rename(columns={'id':'ID','genre_ids':'Genre'})
newdf2


# In[97]:


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


# In[98]:


#Join the df containing the "Now Playing" movies and the df with its corresponding budget/revenue.

final5=pd.concat([newdf2, df6], axis=1, join='inner')
final5


# In[99]:


#Adding the Now Playing df to the df we created that already has the Popular and Top Rated movies.
final6 = final3.append(final5, ignore_index=True)
final6


# In[100]:


#use this to try to find more movies to add.
import requests
import pandas as pd

#Gets the data and puts it into a dictionary for trending movies in the past 7 days.
url8 = 'https://api.themoviedb.org/3/trending/movie/week?api_key=c680984d1f261c766c61129ac1b932fa'
response8 = requests.get(url8)
json_data8 = response8.json()

#The problem with json_data is it is a dictionary with one row - the value for that row is another
#dictionary. So we have to put that into separate dictionary of its own.
sub_dict6 = json_data8["results"]

#Now we have the dictionary we want. We can convert into a dataframe.
df8 = pd.DataFrame.from_dict(sub_dict6)


# In[101]:


newdf3=df8[['id','release_date','popularity','title','vote_average','vote_count','genre_ids']]
newdf3=newdf3.rename(columns={'id':'ID','genre_ids':'Genre'})
newdf3


# In[102]:


import pandas as pd
import numpy as np
df9 = pd.DataFrame({'Budget': [],
                   'Revenue': [],})
for ID in newdf3['ID']: 
    print(ID)
    url9 = 'https://api.themoviedb.org/3/movie/'+str(ID)+'?api_key=c680984d1f261c766c61129ac1b932fa'
    response9 = requests.get(url9)
    json_data9 = response9.json()
    list8=json_data9['budget']
    list9=json_data9['revenue']
    print('Budget '+str(list8)+'\nRevenue '+str(list9))
    df9.loc[len(df9)] = list8, list9


# In[103]:


final7=pd.concat([newdf3, df9], axis=1, join='inner')
final7


# In[104]:


final8 = final6.append(final7, ignore_index=True)
final8


# In[105]:


df966 = pd.DataFrame({'Genres': [],})
for i in range(len(final8)):
    list906=final8['Genre'].str.get(0)[i]
    df966.loc[len(df966)] = list906
df966['Genres'] = df966['Genres'].astype(int)    


# In[107]:


df966


# In[108]:


final8=pd.concat([final8, df966], axis=1, join='inner')
final8


# In[109]:


final8.pop('Genre')
final8


# In[110]:


final8.info()


# In[111]:


dataf=pd.read_csv('/Users/15024/Downloads/Spring 23/Capstone/dataimdb.txt', sep='\t')
dataf


# In[112]:


dataf1=dataf[dataf['averageRating']>3]
dataf2= dataf1[dataf1['numVotes'] > 2500]  
dataf3=dataf2.head(n=2000)   #works if i use a lower sample but has an error when i try to do llarge sample.
dataf3


# In[113]:


dataf99=dataf[dataf['averageRating']>3]
dataf98=dataf99[dataf99['numVotes']>2500]
dataf97=dataf98.tail(n=2000)
dataf97


# In[114]:


dataf97.info()


# In[115]:


dataf3.info()


# In[116]:


import time


# In[117]:


import pandas as pd
import json

#pulls data i need for each imdb id in the sample of 500 i took from the large imdb text file. ]
#this code throws an error for some reason if i try to do a larger sample, i tried to do 1000 but it wouldnt run.
df99 = pd.DataFrame({'ID': [],
                   'release_date': [],
                    'popularity':[],'title':[], 'vote_average':[], 'vote_count':[],'Genre':[], 'Budget':[], 'Revenue':[]})

for tconst in dataf3['tconst']:
    try:
        url99 = 'https://api.themoviedb.org/3/movie/'+str(tconst)+'?api_key=c680984d1f261c766c61129ac1b932fa'
        response99 = requests.get(url99)
        json_data99 = response99.json()
        list91=json_data99['id']
        list92=json_data99['release_date']
        list93=json_data99['popularity']
        list94=json_data99['title']
        list95=json_data99['vote_average']
        list96=json_data99['vote_count']
        list97=json_data99['genres']
        list98=json_data99['budget']
        list99=json_data99['revenue']
    
        df99.loc[len(df99)] = list91, list92, list93,list94,list95,list96,list97,list98,list99
        time.sleep(.25)
    except:
        pass
    #some ids throw errors. The pass function will skip to the next iteration if this occurs.


# In[134]:


#dont use rn
import pandas as pd
import json

#pulls data i need for each imdb id in the sample of 500 i took from the large imdb text file. ]
#this code throws an error for some reason if i try to do a larger sample, i tried to do 1000 but it wouldnt run.
df152 = pd.DataFrame({'ID': [],
                   'release_date': [],
                    'popularity':[],'title':[], 'vote_average':[], 'vote_count':[],'Genre':[], 'Budget':[], 'Revenue':[]})

for tconst in dataf97['tconst']:
    try:
        url152 = 'https://api.themoviedb.org/3/movie/'+str(tconst)+'?api_key=c680984d1f261c766c61129ac1b932fa'
        response152 = requests.get(url152)
        json_data152 = response152.json()
        list300=json_data152['id']
        list301=json_data152['release_date']
        list302=json_data152['popularity']
        list303=json_data152['title']
        list304=json_data152['vote_average']
        list305=json_data152['vote_count']
        list306=json_data152['genres']
        list307=json_data152['budget']
        list308=json_data152['revenue']
    
        df152.loc[len(df152)] = list300, list301, list302,list303,list304,list305,list306,list307,list308
        time.sleep(.25)
    except:
        pass


# In[68]:


df99.info() #only 618 out of 675 were put into df.


# In[69]:


df152.info()


# In[136]:


df90=df99[df99['Revenue']>0]  #getting rid of any that have no value for budget or revenue reported. 
df91=df90[df90['Budget']>0]
df91['ID'] = df91['ID'].astype(int)
df91['vote_count'] = df91['vote_count'].astype(int)
df91


# In[120]:


df781= (df91.reset_index(drop=True))


# In[121]:


df781


# In[135]:


# dont use rn
df400=df152[df152['Revenue']>0]  #getting rid of any that have no value for budget or revenue reported. 
df401=df400[df400['Budget']>0]
df401['ID'] = df401['ID'].astype(int)
df401['vote_count'] = df401['vote_count'].astype(int)
df401


# In[145]:


df782= (df401.reset_index(drop=True))


# In[143]:


df781.info()


# In[138]:


df959 = pd.DataFrame({'Genres': [],})
for i in range(len(df781)):
    list909=(df781['Genre'].str.get(0)[i]['id'])
    df959.loc[len(df959)] = list909
df959['Genres'] = df959['Genres'].astype(int)  


# In[139]:


df959


# In[146]:


df976 = pd.DataFrame({'Genres': [],})
for i in range(len(df782)):
    list913=(df782['Genre'].str.get(0)[i]['id'])
    df976.loc[len(df976)] = list913
df976['Genres'] = df976['Genres'].astype(int)  


# In[147]:


df976


# In[148]:


resultdf = pd.concat([df781.reset_index(drop=True), df959.reset_index(drop=True)], axis=1)


# In[150]:


resultdf2 = pd.concat([df782.reset_index(drop=True), df976.reset_index(drop=True)], axis=1)


# In[152]:


resultdf3 = resultdf.append(resultdf2, ignore_index=True)
resultdf3


# In[153]:


resultdf3.info()


# In[154]:


resultdf3.pop('Genre')


# In[155]:


resultdf3


# In[156]:


final9 = final8.append(resultdf3, ignore_index=True)
final9


# In[ ]:





# In[157]:


final9.info()


# In[158]:


final9['release_date'] = pd.to_datetime(final9['release_date'], format= '%Y/%m/%d')


# In[159]:


#Getting rid of any dupiclate movies from combining the list.
final10=final9.drop_duplicates(subset=['ID']) #going to keep null for newer movies so I can see them whenever values are updated.
final10


# In[166]:


##Trying to change 
url100 = 'https://api.themoviedb.org/3/genre/movie/list?api_key=c680984d1f261c766c61129ac1b932fa'
response100 = requests.get(url100)
json_data100 = response100.json()
subdict456=json_data100['genres']
df89 = pd.DataFrame.from_dict(subdict456)


# In[167]:


df89


# In[168]:


final111=pd.merge(final9, df89, on = "Genres", how = "inner")


# In[204]:


final10.info()


# In[205]:


#Saving df as a csv and opening it.
final10.to_csv('finaldata', index=False)
lastdata=pd.read_csv('finaldata')
lastdata


# In[206]:


import pandas as pd
import csv
import sqlite3


# In[207]:


moviedata=pd.read_csv('finaldata')
moviedata


# In[208]:


#Connecting to sqlite database
connection= sqlite3.connect('/Users/15024/Desktop/movies.db')


# In[209]:


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
            'Budget': 'real',
            'Revenue': 'real',
            'Genres':'integer'}
        )


# In[210]:


#Save and Close database.
connection.commit()
connection.close()


# Playing around to find more movies if I have time below this.

# In[ ]:





# In[ ]:




