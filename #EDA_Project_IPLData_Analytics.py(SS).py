#!/usr/bin/env python
# coding: utf-8

# # In Exploratory Data Analysis what all things we do.

# 1. Handling missing values
# 2. Exploring numerical variables
# 3. Exploring categorical variables
# 4. Finding relationships between features and gaining useful insights
# 5. Cleaning the datasets by using some technique such as dropna,imputation & other.

# # importing some important library

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')


# In[2]:


df = pd.read_csv('matches.csv')
df


# In[3]:


#Explore more info about the datasets
df.head()


# In[4]:


df.tail()


# # Identify Numerical and Categorical data.

# In[5]:


# numerical feature 
[feature for feature in df.columns if df[feature].dtype!='O']


# In[6]:


# Categorical feature
[feature for feature in df.columns if df[feature].dtype=='O']


# # Ovservation

# 1. These are our numberic feature==>> ['id', 'season', 'dl_applied', 'win_by_runs', 'win_by_wickets']
# 2. These are our categorical feature==>> ['city',
#  'date',
#  'team1',
#  'team2',
#  'toss_winner',
#  'toss_decision',
#  'result',
#  'winner',
#  'player_of_match',
#  'venue',
#  'umpire1',
#  'umpire2',
#  'umpire3']

# # information of data

# In[7]:


#check datatypes
df.info()


# In[8]:


#check the statistic of the datasets
k=df.describe()
k


# In[9]:


plt.plot(k)
plt.show()


# In[10]:


#checking the number of uniques value of each columns.
df.nunique()


# In[11]:


df.columns


# In[12]:


#Check the number of row and column from the datasets
print("Number of rows",df.shape[0])
print("Number of column",df.shape[1])


# In[13]:


#check missing value
data=df.isnull().sum()
print('Missing value in our datasets')
print(data)


# In[14]:


[feature for feature in df.columns if df[feature].isnull().sum()>10]


# In[15]:


# Using a heatmap, we can easily visualize the presence of null values in our datasets.
sns.heatmap(df.isnull(),yticklabels=False,cbar=False,cmap='viridis')


# In[16]:


# Find the maximum value in the 'id' column of the dataframe 'df'
df['id'].max()


# In[17]:


# Find all unique values in the 'season' column of the dataframe 'df'
df['season'].unique()


# # Team won by Maximum Runs

# In[18]:


df.iloc[df['win_by_runs'].idxmax()]


# In[19]:


tem_won=df.iloc[df['win_by_runs'].idxmax()]['winner']
tem_won


# # Team won by Minimum Runs

# In[20]:


df.iloc[df[df['win_by_runs'].ge(1)].win_by_runs.idxmin()]


# In[21]:


df.iloc[df[df['win_by_runs'].ge(1)].win_by_runs.idxmin()]['winner']


# In[22]:


# Teams with the most wins
plt.figure(figsize=(12, 6))
team_wins_plot = sns.countplot(y='winner', data=df, order=df['winner'].value_counts().index, palette='coolwarm')
plt.title('Teams with the Most Wins')
plt.xlabel('Number of Wins')
plt.ylabel('Team')
plt.show()


# # Observation
# 
# 1. Mumbai Indians have the most wins, exceeding 100.
# 2. Chennai Super Kings and Kolkata Knight Riders follow, each with around 90 wins.
# 3. Royal Challengers Bangalore, Kings XI Punjab, and Rajasthan Royals have around 70 to 80 wins each.
# 4. Delhi Daredevils and Sunrisers Hyderabad have achieved around 60 to 70 wins.
# 5. Deccan Chargers have around 40 wins.
# 6. Teams like Gujarat Lions, Pune Warriors, Rising Pune Supergiant, and Delhi Capitals have fewer than 30 wins each.
# 7. Kochi Tuskers Kerala and Rising Pune Supergiants have the fewest wins.

# # Team won by Maximum Wickets

# In[23]:


df.iloc[df['win_by_wickets'].idxmax()]


# In[24]:


df.iloc[df['win_by_wickets'].idxmax()]['winner']


# # Team won by Minimum Wickets

# In[25]:


df.iloc[df[df['win_by_wickets'].ge(1)].win_by_wickets.idxmin()]


# In[26]:


df.iloc[df[df['win_by_wickets'].ge(1)].win_by_wickets.idxmin()]['winner']


# # Observation
# 1. Mumbai Indians is the team which won by maximum and minimum runs
# 2. Kolkata Knight Riders is the team which won by maximum and minimum wickets

# # The season with has highest number of matches.

# In[27]:


plt.figure(figsize=(19,9))
sns.countplot(x='season',data=df)
plt.show()


# In 2013, we have the most number of matches

# # Which team has winning in most of the matches

# In[28]:


data=df['winner'].value_counts()
data.index


# In[29]:


plt.figure(figsize=(16,9))
sns.barplot(y=data.index,x=data,orient='h')
plt.show()


# Mumbai Indians are the winners in most of the matches

# # Performance Trends Over Seasons

# In[30]:


team_performance_season = df.groupby(['season', 'winner']).size().unstack().fillna(0)


# In[31]:


team_performance_season


# # Observation
# The performance trends indicate that Mumbai Indians and Chennai Super Kings are the most successful teams, consistently winning matches across multiple seasons. Other teams, like Deccan Chargers and Gujarat Lions, had brief periods of success, while teams like Pune Warriors and Kochi Tuskers Kerala had minimal impact.

# In[32]:


#Performance Trends Over Seasons
plt.figure(figsize=(12, 8))
team_performance_season.plot(kind='line', figsize=(12, 8), marker='o')
plt.title('Team Performance Trends Over Seasons')
plt.xlabel('Season')
plt.ylabel('Number of Wins')
plt.legend(title='Team', bbox_to_anchor=(1.05, 1), loc='upper left')
plt.show()


# # Correction
# Mumbai Indians and Chennai Super Kings consistently performed well, while some teams like Deccan Chargers and Gujarat Lions had brief periods of success.

# # Player Consistency Over Seasons

# In[33]:


top_playmatch = df.player_of_match.value_counts()[:10]
top_playmatch


# In[34]:


fig, ax = plt.subplots(figsize=(15,8))
ax.set_ylim([0,20])
ax.set_ylabel("Count")
ax.set_title("Top player of the match Winners")
top_playmatch.plot.bar()
sns.barplot(x=top_playmatch.index,y=top_playmatch,orient='v',palette='Reds')
plt.show()


# CH Gayle is the most Successful player in all match winners

# # Number of matches played by each team:

# In[35]:


team_mat=pd.melt(df,id_vars=['id','season'],value_vars=['team1','team2'])

plt.figure(figsize=(12,6))
sns.countplot(x='value', data=team_mat)
plt.xticks(rotation='vertical')
plt.show()


# "Mumbai Indians" lead the pack with most number of matches played followed by "Royal Challengers Bangalore". There are also teams with very few matches like 'Rising Pune Supergiants', 'Gujarat Lions' as they are new teams that came in only last season.

# # Number of wins per team:

# In[36]:


plt.figure(figsize=(12,6))
sns.countplot(x='winner',data=df)
plt.xticks(rotation='vertical')
plt.show()


# MI again leads the pack followed by CSK.
# 
# 

# # Now let's see the champions in each season.

# In[37]:


temp_df = df.drop_duplicates(subset=['season'], keep='last')[['season', 'winner']].reset_index(drop=True)
temp_df


# # Observations:
# 1. Mumbai Indians: 4 titles (2013, 2015, 2017, 2019)
# 2. Chennai Super Kings: 3 titles (2010, 2011, 2018)
# 3. Kolkata Knight Riders: 2 titles (2012, 2014)
# 4. Rajasthan Royals: 1 title (2008)
# 5. Deccan Chargers: 1 title (2009)
# 6. Sunrisers Hyderabad: 1 title (2016)

# # Toss decision:

# In[38]:


temp_series = df.toss_decision.value_counts()
labels = (np.array(temp_series.index))
sizes = (np.array((temp_series / temp_series.sum())*100))
colors = ['gold', 'lightskyblue']
plt.pie(sizes, labels=labels, colors=colors,
        autopct='%1.1f%%', shadow=True, startangle=90)
plt.title("Toss decision percentage")
plt.show()


# # Observations

# Almost 55% of the toss decisions are made to field first. Now let us see how this decision varied over time.

# In[39]:


# Since there is a very strong trend towards batting second let us see the win percentage of teams batting second.
num_of_wins = (df.win_by_wickets>0).sum()
num_of_loss = (df.win_by_wickets==0).sum()
labels = ["Wins", "Loss"]
total = float(num_of_wins + num_of_loss)
sizes = [(num_of_wins/total)*100, (num_of_loss/total)*100]
colors = ['gold', 'lightskyblue']
plt.pie(sizes, labels=labels, colors=colors,
        autopct='%1.1f%%', shadow=True, startangle=90)
plt.title("Win percentage batting second")
plt.show()


# So percentage of times teams batting second has won is 53.2. Now let us split this by year and see the distribution.

# # Top players of the match:

# In[40]:


# create a function for labeling #
def autolabel(rects):
    for rect in rects:
        height = rect.get_height()
        ax.text(rect.get_x() + rect.get_width()/2., 1.02*height,
                '%d' % int(height),
                ha='center', va='bottom')


# In[41]:


temp_series = df.player_of_match.value_counts()[:10]
labels = np.array(temp_series.index)
ind = np.arange(len(labels))
width = 0.9
fig, ax = plt.subplots(figsize=(15,8))
rects = ax.bar(ind, np.array(temp_series), width=width)
ax.set_xticks(ind+((width)/2.))
ax.set_xticklabels(labels, rotation='vertical')
ax.set_ylabel("Count")
ax.set_title("Top player of the match awardees")
autolabel(rects)
plt.show()


# # Insights

# CH Gayle is the top player of the match awardee in all the seasons of IPL.

# # Top Umpires one and two.

# In[42]:


#Analysis of top umpire1
plt.figure(figsize=(12, 6))
umpire1_plot = sns.countplot(y='umpire1', data=df, order=df['umpire1'].value_counts().index[:10], palette='husl')
plt.title('Top Umpires (Umpire1)')
plt.xlabel('Number of Matches')
plt.ylabel('Umpire')
plt.show()


# # Observations:
# 1. HDPK Dharmasena: Most frequent umpire with around 75 matches.
# 2. Asad Rauf: Second most frequent with approximately 50 matches.
# 3. S Ravi: Third with around 45 matches.
# 4. AK Chaudhary and Aleem Dar: Each have umpired roughly 35 matches.
# 5. BF Bowden, BR Doctrove, and M Erasmus: Umpired between 25 to 30 matches.
# 6. Nitin Menon and RE Koertzen: Umpired around 20 matches each.

# In[43]:


#Analysis of top umpire2
plt.figure(figsize=(12, 6))
umpire2_plot = sns.countplot(y='umpire2', data=df, order=df['umpire2'].value_counts().index[:10], palette='husl')
plt.title('Top Umpires (Umpire2)')
plt.xlabel('Number of Matches')
plt.ylabel('Umpire')
plt.show()


# # Observation

# 1. C Shamshuddin: Most frequent umpire with around 55 matches.
# 2. S Ravi: Second most frequent with approximately 45 matches.
# 3. SJA Taufel: Third with around 40 matches.
# 4. RJ Tucker and CK Nandan: Each have umpired roughly 30 matches.
# 5. RB Tiffin, VA Kulkarni, SK Tarapore, BNJ Oxenford, and AM Saheba: Umpired between 15 to 25 matches.

# # Correlation b/w season and toss decision

# In[44]:


df.groupby(['season','toss_decision']).size().reset_index().rename(columns={0:'total'})


# In[45]:


plt.figure(figsize=(16,9))
sns.countplot(x='season',hue='toss_decision',data=df)
plt.xticks(rotation='vertical')
plt.show()


# It seems during the initial years, teams wanted to bat first. Voila.! Look at the 2016 season, most of the toss decisions are to field first.

# # Analyzing the impact of toss decisions on match outcomes

# In[46]:


# Analyzing the impact of toss decisions on match outcomes
toss_impact = df[df['toss_winner'] == df['winner']]
toss_decision_wins = toss_impact['toss_decision'].value_counts()

toss_decision_wins.plot(kind='bar', title='Impact of Toss Decision on Match Outcomes', xlabel='Toss Decision', ylabel='Number of Wins')


# # Ovservation

# 1. The bar chart effectively illustrates the correlation between toss decisions and match outcomes.
# 2. It suggests that teams choosing to field after winning the toss have won more matches. 

# # Correlation Between Toss and Match Outcome

# In[47]:


df['toss_match_winner'] = df['toss_winner'] == df['winner']
toss_match_winner_percentage = df['toss_match_winner'].mean() * 100
toss_match_winner_percentage


# # Observation:

# Winning the toss appears to have a slight impact on the match outcome, with a success rate of 51.98%.

# # Count the values in the 'city' column to determine where the most matches were played.

# In[48]:


df.city.value_counts()


# In[49]:


df.city.value_counts().plot(kind='bar')


# # Observation

# Mumbai hosted the most matches (101), while several cities like Bloemfontein hosted only a few (2).

# # Top cities and venues

# In[50]:


plt.figure(figsize=(19,7))
venue_plot = sns.countplot(y='venue', data=df, order=df['venue'].value_counts().index, palette='Set1')
plt.title('Matches Played at Different Venues')
plt.xlabel('Number of Matches')
plt.ylabel('Venue')
plt.show()


# # Observation

# 1. Eden Gardens has hosted the most matches, nearly 80.
# 2. M Chinnaswamy Stadium and Wankhede Stadium follow closely, with around 70 matches each.
# 3. Feroz Shah Kotla, Rajiv Gandhi International Stadium, and MA Chidambaram Stadium have hosted around 60 matches each.
# 4. Sawai Mansingh Stadium and Punjab Cricket Association Stadium have seen over 50 matches.
# 5. Several other venues, such as Subrata Roy Sahara Stadium and Dr. DY Patil Sports Academy, hosted around 40 matches.
# 6. A significant number of venues have hosted between 10 and 30 matches, indicating a wide distribution of match venues.

# In[51]:


plt.figure(figsize=(12, 6))
city_plot = sns.countplot(y='city', data=df, order=df['city'].value_counts().index, palette='Set2')
plt.title('Matches Played in Different Cities')
plt.xlabel('Number of Matches')
plt.ylabel('City')
plt.show()


# # Observation
# 
# 1. Mumbai has hosted the most matches, with over 100.
# 2. Kolkata and Delhi follow closely, each hosting around 90 matches.
# 3. Bangalore, Hyderabad, and Chennai have hosted between 70 and 80 matches.
# 4. Jaipur and Chandigarh have seen over 60 matches.
# 5. Pune and Durban have hosted approximately 40 matches each.
# 6. Several other cities, such as Bengaluru, Visakhapatnam, and Centurion, have hosted between 20 and 30 matches.
# 7. The distribution shows a significant concentration of matches in a few key cities, with a wide spread among other cities hosting fewer matches.

# # Number of matches in each venue

# In[52]:


plt.figure(figsize=(16,9))
sns.countplot(x='venue',data=df)
plt.xticks(rotation='vertical')
plt.show()


# # Observation

# There are quite a few venues present in the data with "M Chinnaswamy Stadium" being the one with most number of matches followed by "Eden Gardens"

# # Venue Performance

# In[53]:


venue_performance = df.groupby(['venue', 'winner']).size().unstack().fillna(0)


# In[54]:


venue_performance


# # Observation
# Mumbai Indians perform exceptionally well at Wankhede Stadium, with 42 wins, while Kolkata Knight Riders dominate Eden Gardens with 45 wins. Chennai Super Kings also show strong performance at MA Chidambaram Stadium with 34 wins.

# In[55]:


#Through the heatmap of visualization team performance at top venues.
top_venues = venue_performance.sum(axis=1).sort_values(ascending=False).head(10).index
plt.figure(figsize=(12, 8))
sns.heatmap(venue_performance.loc[top_venues], annot=True, fmt='.0f', cmap='Blues')
plt.title('Team Performance at Top Venues')
plt.xlabel('Team')
plt.ylabel('Venue')
plt.show()


# # Correction
# Mumbai Indians perform best at Wankhede Stadium, and Kolkata Knight Riders dominate Eden Gardens, while Chennai Super Kings excel at MA Chidambaram Stadium.

# # Impact of Duckworth-Lewis (DL) Method

# In[56]:


dl_applied_matches = df[df['dl_applied'] == 1]
dl_outcome_percentage = dl_applied_matches['result'].value_counts(normalize=True) * 100


# In[57]:


dl_applied_matches
dl_outcome_percentage


# # Observation
# In matches where the Duckworth-Lewis (DL) method was applied, 100% of the results were classified as "normal."

# In[58]:


#Impact of DL Method
plt.figure(figsize=(8, 6))
sns.barplot(x=dl_outcome_percentage.index, y=dl_outcome_percentage.values, palette='cubehelix')
plt.title('Impact of Duckworth-Lewis Method on Match Outcome')
plt.ylabel('Percentage')
plt.xlabel('Result')
plt.show()


# # Observation

# The DL method significantly affects match outcomes, typically resulting in a higher percentage of matches concluding normally rather than ending in ties or no results.

# # Finally we clean our datasets

# # Cleaning the datasets by using imputaion method.

# In[59]:


from sklearn.impute import SimpleImputer


# In[60]:


impute = SimpleImputer(strategy='most_frequent')
impute.fit(df[['city','winner','player_of_match','umpire1','umpire2','umpire3']])
df[['city','winner','player_of_match','umpire1','umpire2','umpire3']] = impute.transform(df[['city','winner','player_of_match','umpire1','umpire2','umpire3']])


# In[61]:


df.isnull().sum()


# In[ ]:





# # Conclusion

# 1. Increasing Matches: The number of matches per season has been rising, reflecting growing popularity.
# 
# 2. Top Teams: Mumbai Indians and Chennai Super Kings are the most successful teams.
# 
# 3. Toss Impact: Winning the toss provides a slight advantage (51% match win rate).
# 
# 4. Toss Decision: Teams prefer to field first after winning the toss.
# 
# 5. Key Players: AB de Villiers, Chris Gayle, and MS Dhoni are frequently awarded Player of the Match.
# 
# 6. Popular Venues: Wankhede Stadium, M Chinnaswamy Stadium, and Eden Gardens host the most matches.
# 
# 7. DL Method: The Duckworth-Lewis method significantly impacts match outcomes.
# 
# 8. Venue Performance: Teams have venue-specific strengths, like Mumbai Indians at Wankhede Stadium.

# This dataset provides a comprehensive overview of match statistics, offering valuable insights into team performances, player impact, and strategic elements of the game. These conclusions can help teams, analysts, and stakeholders make informed decisions to enhance their strategies and performance in future matches.

# In[ ]:





# In[ ]:




