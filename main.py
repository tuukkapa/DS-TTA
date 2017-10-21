# This file is currently used for testing twitterconnector

import twitterconnector
import tweet_processor
import sys
import pandas as pd

if __name__ == "__main__":
    #list = twitterconnector.getTweetsAsList(sys.argv[1])
    tweets = twitterconnector.getTweetsAsList('#Trump')
    tweets_as_df = pd.DataFrame({'col':list(tweets)})
    tokenized_tweets = tweet_processor.process_tweets(tweets_as_df)
    
    # TODO get sentiments
    #pred = classify_tweet()
    preds = classify_tweets(tokenized_tweets)


    # TODO visualize tokenizedDF

    
