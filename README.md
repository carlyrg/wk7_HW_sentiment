
## Twitter News Mood
wk-7 homework

In this project, I analyzed the sentiment of tweets from the Twitter accounts of major news organizations. To obtain the data for this analysis, I pulled 100 most recent tweets each from BBC, CBS, CNN, Fox and the New York times using Twitter’s Tweepy API. I then analyzed the sentiment of each tweet using the Vader library. I then converted the collection of news tweets into a Pandas dataframe and exported  to a csv file.

I used Matplotlib to produce a scatter plot of the sentiment for each tweet. I then aggregated the data to find the overall sentiment of each news organization’s tweets and created a bar graph to illustrate those findings.

Some of my observations can be found at the bottom.

```python
# Dependencies
import tweepy
import pandas as pd
import time
from datetime import datetime
import matplotlib.pyplot as plt
from matplotlib import style
import seaborn as sns
import numpy as np

from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
analyzer = SentimentIntensityAnalyzer()

auth = tweepy.OAuthHandler(consumer_key, consumer_secret)
auth.set_access_token(app_key, app_secret)
api = tweepy.API(auth)
```


```python
target_terms = ("@BBCWorld", "@CBSNews", "@cnnbrk", "@FoxNews", "@nytimes")

```


```python
results_df = []

for target in target_terms:
    counter = 0
    temp_results = []
    for status in tweepy.Cursor(api.user_timeline, id=target).items(100):
        _status = status._json # dict
        _status['row_nbr'] = counter
        temp_results.append(_status)
        counter += 1
    
    df=pd.DataFrame([x for x in temp_results])[['text', 'created_at', 'row_nbr']]
    df['source account'] = target
    results_df.append(df)

all_data = pd.concat(results_df, axis=0)
all_data['polarity_scores'] = all_data['text'].map(lambda x: analyzer.polarity_scores(x))
all_data['pos'] = all_data.polarity_scores.map(lambda x: x.get('pos'))
all_data['neg'] = all_data.polarity_scores.map(lambda x: x.get('neg'))
all_data['neu'] = all_data.polarity_scores.map(lambda x: x.get('neu'))
all_data['compound'] = all_data.polarity_scores.map(lambda x: x.get('compound'))


all_data.head()

    
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>text</th>
      <th>created_at</th>
      <th>row_nbr</th>
      <th>source account</th>
      <th>polarity_scores</th>
      <th>pos</th>
      <th>neg</th>
      <th>neu</th>
      <th>compound</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Little hope ahead of polls in Mexico's Sinaloa...</td>
      <td>Fri Jun 29 23:58:03 +0000 2018</td>
      <td>0</td>
      <td>@BBCWorld</td>
      <td>{'neg': 0.0, 'neu': 0.775, 'pos': 0.225, 'comp...</td>
      <td>0.225</td>
      <td>0.000</td>
      <td>0.775</td>
      <td>0.3832</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Thai cave rescue: Drones, dogs, drilling and d...</td>
      <td>Fri Jun 29 23:55:36 +0000 2018</td>
      <td>1</td>
      <td>@BBCWorld</td>
      <td>{'neg': 0.226, 'neu': 0.526, 'pos': 0.248, 'co...</td>
      <td>0.248</td>
      <td>0.226</td>
      <td>0.526</td>
      <td>0.0772</td>
    </tr>
    <tr>
      <th>2</th>
      <td>The young Austrian leader sharing power with t...</td>
      <td>Fri Jun 29 23:50:03 +0000 2018</td>
      <td>2</td>
      <td>@BBCWorld</td>
      <td>{'neg': 0.0, 'neu': 0.781, 'pos': 0.219, 'comp...</td>
      <td>0.219</td>
      <td>0.000</td>
      <td>0.781</td>
      <td>0.4215</td>
    </tr>
    <tr>
      <th>3</th>
      <td>DR Congo: Oil drilling allowed in Virguna, Sal...</td>
      <td>Fri Jun 29 23:47:37 +0000 2018</td>
      <td>3</td>
      <td>@BBCWorld</td>
      <td>{'neg': 0.0, 'neu': 1.0, 'pos': 0.0, 'compound...</td>
      <td>0.000</td>
      <td>0.000</td>
      <td>1.000</td>
      <td>0.0000</td>
    </tr>
    <tr>
      <th>4</th>
      <td>Doug Ford: Brother of notorious Rob Ford takes...</td>
      <td>Fri Jun 29 21:51:12 +0000 2018</td>
      <td>4</td>
      <td>@BBCWorld</td>
      <td>{'neg': 0.419, 'neu': 0.581, 'pos': 0.0, 'comp...</td>
      <td>0.000</td>
      <td>0.419</td>
      <td>0.581</td>
      <td>-0.7579</td>
    </tr>
  </tbody>
</table>
</div>




```python
all_data.to_csv('News Mood.csv')
```


```python
news_sort = all_data.groupby("source account")
BBC = news_sort.get_group('@BBCWorld')
BBC_created = BBC['row_nbr']
BBC_polarity = BBC['compound']

CBS = news_sort.get_group('@CBSNews')
CBS_created = CBS['row_nbr']
CBS_polarity = CBS['compound']

CNN = news_sort.get_group('@cnnbrk')
CNN_created = CNN['row_nbr']
CNN_polarity = CNN['compound']

Fox = news_sort.get_group('@FoxNews')
Fox_created = Fox['row_nbr']
Fox_polarity = Fox['compound']

nyt = news_sort.get_group('@nytimes')
nyt_created = nyt['row_nbr']
nyt_polarity = nyt['compound']



# Create scatter plot
sns.set()
plt.figure(figsize=(15,8))

BBC_plot = plt.scatter(BBC_created, BBC_polarity,  
            c = 'green', alpha = 0.5, edgecolor='black',linewidths=.5)
CBS_plot = plt.scatter(CBS_created, CBS_polarity,  
            c = 'blue', alpha = 0.5, edgecolor='black',linewidths=.5)
CNN_plot = plt.scatter(CNN_created, CNN_polarity,  
            c = 'red', alpha = 0.5, edgecolor='black',linewidths=.5)
Fox_plot = plt.scatter(Fox_created, Fox_polarity,  
            c = 'gold', alpha = 0.5, edgecolor='black',linewidths=.5)
nytimes_plot = plt.scatter(nyt_created, nyt_polarity,  
            c = 'purple', alpha = 0.5, edgecolor='black',linewidths=.5)
                         
# format graph
now = datetime.now()
now = now.strftime("%B %d, %Y")
plt.title(f"NewsMood ({now})", fontsize = 16)
plt.ylabel("Tweet Polarity", fontsize = 12)
plt.xlabel("Tweets Ago", fontsize=12)
plt.xticks(rotation=45)

# legend
lgnd = plt.legend(('BBC News','CBS News','CNN', 'FOX', 'NYTimes'), title = 'News Channel', bbox_to_anchor=(1.0, 1.0))
lgnd.legendHandles[0]._sizes = [100]
lgnd.legendHandles[1]._sizes = [100]
lgnd.legendHandles[2]._sizes = [100]
lgnd.legendHandles[3]._sizes = [100]
lgnd.legendHandles[4]._sizes = [100]

plt.show()
plt.savefig('news_mood.png')
```


![png](output_5_0.png)



    <matplotlib.figure.Figure at 0x1ccd621b208>



```python
fig, ax = plt.subplots(figsize=(10,8))

avg1 = ax.bar(0, Fox['compound'].mean(), label='Fox')
avg2 = ax.bar(1, CBS['compound'].mean(), label = 'CBS')
avg3 = ax.bar(2, CNN['compound'].mean(), label = 'CNN')
avg4 = ax.bar(3, BBC['compound'].mean(), label = 'BBC')
avg5 = ax.bar(4, nyt['compound'].mean(), label = 'NYT')

ax.set_xlabel('News Source', fontsize=14)
ax.set_ylabel('Polarity',fontsize=14)
# ax.set_ylim([-1, 1])
ax.set_title(f'Overall Polarity by news source ({now})', fontsize=16)
ax.set_xticklabels(['x','Fox', 'CBS', 'CNN', 'BBC', "NTY"], fontsize=12)

totals=[]
for i in ax.patches:
    totals.append(i.get_height())

total = sum(totals)

for i in ax.patches:
    ax.text(i.get_x()+.3, i.get_height()-.01, str(round((i.get_height()/total), 2)), fontsize = 12, color='black')
     
plt.show()
plt.savefig('overall_sentiment.png')
```


![png](output_6_0.png)



    <matplotlib.figure.Figure at 0x1ccd67e7208>


### observations
1. Sentiment varries on the day and time and current news. So no direct observation can be stated about whether tweetse are generally positive or negative overall.
2. All news sources polled seemed to be pretty varried with positive and negative tweets throught the day, so probably reporting on a variety of subjects
3. There are not too many tweets that are beyond the +/- 0.70 mark and a surprising number are completely neutral.
