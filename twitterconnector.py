import tweepy
import credentials
import csv

# Function for auhtentication to Twitter. Requires credentials.py at the same directory.
def authenticate():
    auth = tweepy.OAuthHandler(credentials.consumer_key, credentials.consumer_secret)
    auth.set_access_token(credentials.access_token_key, credentials.access_token_secret)
    api = tweepy.API(auth)
    return api

# Gets 100 tweets with a hashtag. Removes linebreaks from tweets.
def getTweetsAsList(hashtag):
    api = authenticate()
    max_tweets = 100
    tweet_texts = []
    for tweet in tweepy.Cursor(api.search, q=hashtag, lang="en", rpp=100).items(max_tweets):
        tweet_text = tweet.text.replace('\r', '').replace('\n', ' ')
        tweet_texts.append(tweet_text)
    #output to console for debugging
    for text in tweet_texts:
        print(text)
    return tweet_texts

# Gets 100 tweets with a hashtag and saves them to a csv file, one tweet per row.
def getTweetsAsCSV(filename, hashtag):
    tweetlist = getTweetsAsList(hashtag)
    with open(filename, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        for item in tweetlist:
            writer.writerow([item])