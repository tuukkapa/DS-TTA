"""Driver module for running the program"""

import sys
import tweet_processor
import twitterconnector
import visualizer

if __name__ == "__main__":

    # Comment / uncomment to load new tweets
    twitterconnector.query_tweets_to_file('examples.txt', '#trump', 200)
    tweets = twitterconnector.read_tweets_from_file('examples.txt')
    tokenized_tweets = tweet_processor.process_raw_tweets(tweets)
    counts = tweet_processor.count_tokens(tokenized_tweets)
    visualizer.word_cloud_from_frequencies(counts, "example_cloud.png")

    # TODO get sentiments
    # pred = classify_tweet()
    # preds = classify_tweets(tokenized_tweets)
