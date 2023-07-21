# DS_Final_Project_ZJB

Data Collection:
For my data collection I used The Movie Database (TMDb), to extract data on movies that are under 3 sections. Popular, Top Rated, and Now Playing.
Once I was able to pull the data, I converted it into a panda dataframe so I could manipulate it however I like.

For each section I used the movie id, which is included in the data pulled from TMDb to make another dataframe that contains the budget and revenue of each movie.
Some movies don't have data for budget or data but this is likely due to the movie being released very recently so the data may not be in the database.
Also, movies that are not shown in the US tend to not always have a value present for budget/revenue as well.

I then merged all of the dataframes for each section, along with their corresponding budget/revenue dataframes into a single dataframe that can be uploaded into a data store. (In this case I am using Sqlite3.)
Some movies are presnt in more than one section, to deal with this I dropped any rows that had duplicate movie ids from the merged dataframe. Lastly, I saved the merged dataframe as a csv.
I then connected to the sqlite program and created a database in a desktop folder. If the database doesn't already exist, it will create it for you.
Finally, I create a table called "Movies" that has the same column names as my merged movie dataframe and set them to the same datatype as well. I can then push my data from python pandas dataframe to the movies.db on Sqlite3 where it will store my data for future data exploration and to build a predictive model that will predict the revenue of a movie.

Project Presentation:
I created a poster for this project that inlcudes an abstract, objectives of the project, an overview of methods and materials used for the project, as well as results from the project so far. 
