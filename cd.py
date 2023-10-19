#%%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# %%
df_tracks=pd.read_csv('Datasets/tracks.csv')
# %%
df_tracks.info()
# %%
df_tracks.shape
# %%
df_tracks.isnull().sum()
# %%
df_tracks.head()
# %% least popular song in spotify
sorted_df=df_tracks.sort_values('popularity',ascending=True)
sorted_df.head(3)
# %%
df_tracks.describe()

# %% top 10 popular songs > 90
above_90_pop=df_tracks.sort_values('popularity',ascending=False)
#(sorted_df[sorted_df.popularity > 90,]).head()
above_90_pop.head(10)

# %% changing the release date format into date and time
df_tracks.release_date=pd.to_datetime(df_tracks.release_date)
# %%
df_tracks.info()
# %%setting the index as release date
df_tracks.set_index("release_date",inplace=True)

# %%  
df_tracks.head()
# %% checkng the artist in the 18th row
df_tracks['artists'].iloc[18]

# %%
df_tracks.head()
# %% converting duration in mn(duration_ms) into duration in seconds(duration)
df_tracks['duration']=df_tracks['duration_ms'].apply(lambda x: round(x/1000))
df_tracks.drop('duration_ms',axis=1,inplace=True)
# %%
df_tracks.head()
# %% Correlation Heatmap between Variable using Pearson correlation method
cm = df_tracks.drop(['key','mode','explicit'], axis=1).corr(method = 'pearson')
plt.figure(figsize=(14,6))
map = sns.heatmap(cm, annot = True, fmt = '.1g', vmin=-1, vmax=1, center=0, cmap='inferno', linewidths=1, linecolor='Black')
map.set_title('Correlation Heatmap between Variable')
map.set_xticklabels(map.get_xticklabels(), rotation=90)
# %% Considering 0.4% of the total dataset to create Regression plots
sam = df_tracks.sample(int(0.004 * len(df_tracks)))
len(sam)

# %% Regression plot - Correlation between Loudness and Energy
plt.figure(figsize=(10,6))
sns.regplot(data=sam, y='loudness', x='energy', color='c').set(title='Loudness vs Energy Correlation')

# %%
plt.figure(figsize=(10,6))
sns.regplot(data=sam, y='popularity', x='acousticness', color='b').set(title='Popularity vs Acousticness Correlation')
# %% Creating new column in tracks dataset (Year, Release Date)
df_tracks['dates']=df_tracks.index.get_level_values('release_date')
df_tracks.dates=pd.to_datetime(df_tracks.dates)
years=df_tracks.dates.dt.year
# %% 
df_tracks.head()
# %% Distibution plot - Visualize total number of songs on Spotify since 1992
# Number of songs has increased rapidly since 1920
sns.displot(years, discrete=True, aspect=2, height=5, kind='hist').set(title='Number of songs per year')

# %% 
tracks=df_tracks

# %% Change in Duration of songs wrt Years
total_dr = tracks.duration
fig_dims = (18,7)
fig, ax = plt.subplots(figsize=fig_dims)
fig = sns.barplot(x = years, y = total_dr, ax = ax, errwidth = False).set(title='Years vs Duration')
plt.xticks(rotation=90)


# %%
total_dr = tracks.duration
sns.set_style(style='whitegrid')
fig_dims = (10,5)
fig, ax = plt.subplots(figsize=fig_dims)
fig = sns.lineplot(x = years, y = total_dr, ax = ax).set(title='Years vs Duration')
plt.xticks(rotation=60)

# %% Spotify Features Dataset Analysis
# Duration of songs in different Genres
genre = pd.read_csv('Datasets/SpotifyFeatures.csv')
plt.title('Duration of songs in different Genres')
sns.color_palette('rocket', as_cmap=True)
sns.barplot(y='genre', x='duration_ms', data=genre)
plt.xlabel('Duration in milliseconds')
plt.ylabel('Genres')
# %%  Top 5 Genres by Popularity
sns.set_style(style='darkgrid')
plt.figure(figsize=(10,5))
popular = genre.sort_values('popularity', ascending=False).head(10)
sns.barplot(y = 'genre', x = 'popularity', data = popular).set(title='Top 5 Genres by Popularity')

# %%
