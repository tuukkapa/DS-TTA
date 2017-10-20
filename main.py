# This file is currently used for testing twitterconnector

import twitterconnector
import SentimentClassifier as sc
import sys
import pandas as pd

if __name__ == "__main__":
    #list = twitterconnector.getTweetsAsList(sys.argv[1])
    tweets = twitterconnector.getTweetsAsList('#DataScience')
    tweets_as_df = pd.DataFrame({'col':list})
    tokenized_tweets = sc.process(tweets_as_df)
    
    # TODO get sentiments
    #pos, neg = classify_tweets(tokenized_tweets)

    # TODO visualize tokenizedDF