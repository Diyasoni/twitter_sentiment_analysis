import streamlit as st
import tweepy
import textblob
from textblob import TextBlob
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import re
import nltk
import string
import os
import sys
from wordcloud import WordCloud,STOPWORDS
from PIL import Image
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from nltk.stem import SnowballStemmer
from sklearn.feature_extraction.text import CountVectorizer
from langdetect import detect
import base64
def get_base64(bin_file):
    with open(bin_file, 'rb') as f:
        data = f.read()
    return base64.b64encode(data).decode()


def set_background(png_file):
    bin_str = get_base64(png_file)
    page_bg_img = '''
    <style>
    .stApp {
    background-image: url("data:image/png;base64,%s");
    background-size: cover;
    }
    </style>
    ''' % bin_str
    st.markdown(page_bg_img, unsafe_allow_html=True)

set_background('tw4.jpg')

consumerKey="On6lgMcJjrSYoIEdadWhlYOkd"
consumerSecret="qwhlSUuWZZpNisuNsyiaz39pjs4a9DApTpF2C0MTHJCj2UDokS"
accessToken="1058627571569451008-ZuWA8w2nAOdait7Om37O88qbTjfCwm"
accessTokenSecret="pE1t7c0qXyc4G0bECVXJXWBSMcFcQ6xCatkl3mRZfmiKS"

auth=tweepy.OAuthHandler(consumerKey,consumerSecret)
auth.set_access_token(accessToken,accessTokenSecret)
api=tweepy.API(auth)

tab1, tab2 = st.tabs(["Real Time Tweet Analyzer", "Analyze your own Tweet"])

with tab1:
    st.title("TWEET ANALYZER")
    keyword=st.text_input("Enter a keyword to search")
    no_tweets=st.number_input('Enter number of tweets',min_value=10,step=1)

    def percentage(part,whole):
        return 100* float(part)/float(whole)

    select = st.radio(
         "Select any one",
        ('Fetch Tweets', 'Analyze','Visualize'))

    tweets=tweepy.Cursor(api.search_tweets,
              q=keyword,
              lang="en").items(no_tweets)

    if st.button('Submit'):
        positive=0
        negative=0
        neutral=0
        polarity=0
        tweets_list=[]
        positive_list=[]
        negative_list=[]
        neutral_list=[]
        for tweet in tweets:
            tweets_list.append(tweet.text)
            analysis=TextBlob(tweet.text)
            score = SentimentIntensityAnalyzer().polarity_scores(tweet.text)
            neg=score['neg']
            neu=score['neu']
            pos=score['pos']
            comp=score['compound']
            polarity += analysis.sentiment.polarity
            if neg>pos:
                negative_list.append(tweet.text)
                negative += 1
            elif pos>neg:
                positive_list.append(tweet.text)
                positive += 1
            elif pos==neg:
                neutral_list.append(tweet.text)
                neutral+=1
        positive=percentage(positive,no_tweets)
        negative=percentage(negative,no_tweets)
        neutral=percentage(neutral,no_tweets)
        polarity = percentage(polarity, no_tweets)
        positive = format(positive, '.1f')
        negative = format(negative, '.1f')
        neutral = format(neutral, '.1f')

        tweets_list = pd.DataFrame(tweets_list)
        positive_list = pd.DataFrame(positive_list)
        negative_list = pd.DataFrame(negative_list)
        neutral_list = pd.DataFrame(neutral_list)
        tw_list=pd.DataFrame(tweets_list)
        tw_list["text"]=tw_list[0]

        import emoji
        def cleantext(text):
            text=re.sub(r'@[A-Za-z0-9]+','',text) #remove @mentions
            text=re.sub(r'RT[\s]+','',text)   #remove RT for retweets
            text=re.sub(r'https?:\/\/\S+','',text)  #remove urls
            text=re.sub(r'#','',text)  #remove hashtags
            text=re.sub(emoji.get_emoji_regexp(),r'',text)
            text=text.translate(str.maketrans('','',string.punctuation))
            return text

        tw_list["text"]=tw_list["text"].apply(cleantext)

        tw_list[['polarity','subjectivity']]=tw_list['text'].apply(lambda text: pd.Series(TextBlob(text).sentiment))
        for index,row in tw_list['text'].iteritems():
            #iteritems func is used to iterate each item of each column
            score = SentimentIntensityAnalyzer().polarity_scores(row)
            neg=score['neg']
            neu=score['neu']
            pos=score['pos']
            comp=score['compound']
            if neg>pos:
                tw_list.loc[index,'sentiment']="negative"
            elif pos>neg:
                tw_list.loc[index,'sentiment']="positive"
            else:
                tw_list.loc[index,'sentiment']="neutral"
            tw_list.loc[index,'neg']=neg
            tw_list.loc[index,'neu']=neu
            tw_list.loc[index,'pos']=pos
            tw_list.loc[index,'compound']=comp

        if select=='Fetch Tweets':  
            st.write("Total number of tweets: ",len(tweets_list))
            st.write("Positive number of tweets: ",len(positive_list))
            st.write("Negative number of tweets: ", len(negative_list))
            st.write("Neutral number of tweets: ",len(neutral_list))
            positive_list
            negative_list
            neutral_list
        elif select=='Analyze':
        
            tw_list
        elif select=="Visualize":
            tw_list["text"]=tw_list["text"].apply(cleantext)
            positive_tw_list=tw_list[tw_list['sentiment']=="positive"]
            negative_tw_list=tw_list[tw_list['sentiment']=="negative"]
            neutral_tw_list=tw_list[tw_list['sentiment']=="neutral"]
            def wordcloud(text):
                mask=np.array(Image.open("cloud.jpg"))
                stopwords=set(STOPWORDS)
                wc=WordCloud(background_color="white",mask = mask, stopwords=stopwords)
                wc.generate(str(text))   
                wc.to_file("wc.jpg")
                path="wc.jpg"
                st.image(path)
            st.write('Generating wordcloud of all positive tweets')
            wordcloud(positive_tw_list["text"].values)
            st.write('Generating wordcloud of all negative tweets')
            wordcloud(negative_tw_list["text"].values)
            st.write('Generating wordcloud of all neutral tweets')
            wordcloud(neutral_tw_list["text"].values)

            def count_values(data,feature):
                total=data.loc[:,feature].value_counts(dropna=False)    
    #With normalize set to True, returns the relative frequency by dividing all values by the sum of values.
                percentage=round(data.loc[:,feature].value_counts(dropna=False,normalize=True)*100,2)
                return pd.concat([total,percentage],axis=1,keys=['Total','Percentage'])
            st.write(count_values(tw_list,"sentiment"))
            pc=count_values(tw_list,"sentiment")

            st.set_option('deprecation.showPyplotGlobalUse', False)

            st.bar_chart(pc["Percentage"])

with tab2:
    st.header('SENTIMENT ANALYSIS')
    EnterTweet=st.text_input("Type your own Tweet")
    if st.button('submit'):
        polarity=0
        analysis=TextBlob(EnterTweet)
        score = SentimentIntensityAnalyzer().polarity_scores(EnterTweet)
        neg=score['neg']
        neu=score['neu']
        pos=score['pos']
        comp=score['compound']
        polarity += analysis.sentiment.polarity
        if neg>pos:
            st.write('negative')
        elif pos>neg:
            st.write('positive')
        else:
            st.write('neutral')

 