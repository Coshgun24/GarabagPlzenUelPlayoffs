import snscrape.modules.twitter as sntwitter
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import string
import re
import nltk
from nltk.util import pr
from nltk.corpus import stopwords
import warnings
warnings.filterwarnings('ignore')
stemmer = nltk.SnowballStemmer("english")
nltk.download('stopwords')
stopword=set(stopwords.words('english'))
from textblob import TextBlob
from wordcloud import WordCloud,STOPWORDS

maxTweets = 300
tweets_list_new = []

for i,tweet in enumerate(sntwitter.TwitterSearchScraper('ViktoriaPlzenQarabag since:2022-08-23 until:2022-08-24').get_items()):
    if i>maxTweets:
        break
    tweets_list_new.append([tweet.date, tweet.id, tweet.content, tweet.username])

df = pd.DataFrame(tweets_list_new, columns=['Datetime', 'Tweet Id', 'Text', 'Username'])
df.head()

#Export dataframe into a CSV
df.to_csv('tweets_uel.csv', sep=',', index=False)

df.Text

df.info()

def hashtag_extract(text_list):
    hashtags = []
  
    for text in text_list:
        ht = re.findall(r"#(\w+)", text)
        hashtags.append(ht)
    return hashtags
def generate_hashtag_freqdist(hashtags):
    a = nltk.FreqDist(hashtags)
    d = pd.DataFrame({'Hashtag': list(a.keys()),
    'Count': list(a.values())})
    
    d = d.nlargest(columns="Count", n = 25)
    plt.figure(figsize=(16,7))
    ax = sns.barplot(data=d, x= "Hashtag", y = "Count")
    plt.xticks(rotation=80)
    ax.set(ylabel = 'Count')
    plt.show()

hashtags = hashtag_extract(df["Text"])
hashtags = sum(hashtags, [])

generate_hashtag_freqdist(hashtags)

def clean(text):
    text = str(text).lower()
    text = re.sub('\[.*?\]', '', text)
    text = re.sub('https?://\S+|www\.\S+', '', text)
    text = re.sub('<.*?>+', '', text)
    text = re.sub('[%s]' % re.escape(string.punctuation), '', text)
    text = re.sub('\n', '', text)
    text = re.sub('\w*\d\w*', '', text)
    text = [word for word in text.split(' ') if word not in stopword]
    text=" ".join(text)
    text = [stemmer.stem(word) for word in text.split(' ')]
    text=" ".join(text)
    return text
df["Text"] = df["Text"].apply(clean)

def analyze_sentiment(tweet):
    analysis = TextBlob(clean(tweet))
    if analysis.sentiment.polarity > 0:
        return 1
    elif analysis.sentiment.polarity == 0:
        return 0
    else:
        return -1

df['Sentiment'] = df['Text'].apply(lambda x:analyze_sentiment(x))
df['Length'] = df['Text'].apply(len)
df['Word_counts'] = df['Text'].apply(lambda x:len(str(x).split()))

plt.figure(figsize = (9,6))
sns.countplot(data = df, x = 'Sentiment')
plt.show()

fig, ax = plt.subplots(figsize = (8, 6))
sizes = [count for count in df['Sentiment'].value_counts()]
labels = list(df['Sentiment'].value_counts().index)
explode = (0.1, 0.1, 0)
ax.pie(x = sizes, labels = labels, autopct = '%1.1f%%', explode = explode)
ax.set_title('Sentiment Polarity')
plt.show()

neutral = df[df['Sentiment'] == 0]
positive = df[df['Sentiment'] == 1]
negative = df[df['Sentiment'] == -1]

stop_word=['le','ht','rt','se','la','en','sa','del','de','fk','ft','fc','je','el','v']

stop_w=STOPWORDS

stop_w=stop_w.update(stop_word)

txt = ' '.join(text for text in df['Text'])
wordcloud = WordCloud(
background_color = 'black',
max_font_size = 100,
max_words = 100,
width = 800,
height = 500,stopwords=stop_w
).generate(txt)
plt.figure(figsize=(12, 7))
plt.imshow(wordcloud,interpolation = 'bicubic')
plt.axis('off')
plt.show()
#wordcloud.to_image()
#wordcloud.to_file("words.png")

positive_words =' '.join([text for text in df['Text'][df['Sentiment'] == 1]])
wordcloud = WordCloud(width=800, height=500, random_state=21, max_font_size=110,stopwords=stop_w).generate(positive_words)
plt.figure(figsize=(12, 7))
plt.imshow(wordcloud, interpolation="bilinear")
plt.axis('off')
plt.show()
#wordcloud.to_image()
#wordcloud.to_file("positive.png")

negative_words =' '.join([text for text in df['Text'][df['Sentiment'] == -1]])
wordcloud = WordCloud(width=800, height=500, random_state=21, max_font_size=110,stopwords=stop_w).generate(negative_words)
plt.figure(figsize=(12, 7))
plt.imshow(wordcloud, interpolation="bilinear")
plt.axis('off')
plt.show()
#wordcloud.to_image()
#wordcloud.to_file("negative.png")

neutral_words =' '.join([text for text in df['Text'][df['Sentiment'] == 0]])
wordcloud = WordCloud(width=800, height=500, random_state=21, max_font_size=110,stopwords=stop_w).generate(neutral_words)
plt.figure(figsize=(12, 7))
plt.imshow(wordcloud, interpolation="bilinear")
plt.axis('off')
plt.show()
#wordcloud.to_image()
#wordcloud.to_file("neutral.png")