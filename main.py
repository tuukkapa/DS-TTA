"""Driver module for running the program"""

import argparse
import os
import sys
import tweet_processor as tp
import twitterconnector as tc
import visualizer as vis
import pandas as pd
from SentimentClassifier import load_models, classify_tweets


def query_tweets_to_files(queries, count):
    if not os.path.isdir('tweets'):
        os.mkdir('tweets')
    for query in queries:
        tc.query_tweets_to_file(
            f'tweets/{query}.txt', query, count)


def analyze_tweets(tweets, model, w2v_model):
    """Analyze tweets

    Tweets is expected to be list of tuples (topic, tweets)
    """
    # TODO DO EVERYTHING HERE
    #tweets = [("StarWars", tc.query_tweets("StarWars"))]
    
    #tweets = tc.query_tweets('starwars')
    df = pd.DataFrame(columns=['pos', 'neu', 'neg'])
    for topic, topic_tweets in tweets:
        tokenized_tweets =  tp.process_raw_tweets(topic_tweets)
        df.loc[topic], dummy = classify_tweets(tokenized_tweets, model, w2v_model)
        vis.word_cloud_from_frequencies(tp.count_tokens(tokenized_tweets), f"{topic}_cloud.png", width=800, height=400,)
    
    vis.bar_plot_from_dataframe(df, 'results.png')
    
    print(df)


def main():
    """Allows doing things from command line"""
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest='operation')

    pparser = subparsers.add_parser('process',
                                    description='Processes tweets')
    inputgroup = pparser.add_mutually_exclusive_group()
    inputgroup.add_argument('-f', '--files', nargs='+')
    inputgroup.add_argument('-q', '--queries', nargs='+')
    pparser.add_argument('-wc', '--wordcloud', action='store_true')
    pparser.add_argument('-n', type=int, default=300, dest='count')

    gparser = subparsers.add_parser('get',
                                    description='Gets tweets and writes them into files')
    gparser.add_argument('queries', nargs='+')
    gparser.add_argument('-n', type=int, default=300, dest='count')

    args = parser.parse_args()
    
    model, w2v_model = load_models()
    if args.operation == 'get':
        query_tweets_to_files(args.queries, args.count)
    elif args.operation == 'process':
        if args.queries:
            tweets = [(query,
                       tc.query_tweets(query, args.count)) for query in args.queries]
        else:
            tweets = [(os.path.splitext(os.path.basename(f))[0],
                       tc.read_tweets_from_file(f)) for f in args.files]
        analyze_tweets(tweets, model, w2v_model)

    while True:
        cmd = input("Enter command:\n")
        if cmd == "exit":
            break
        elif cmd == "test":
            test_msg = input("Enter message to eval\n")
            test_msg = [test_msg]
            df = pd.DataFrame({'tokens': list(map(tp.tokenize, test_msg))})
            result, predictions = classify_tweets(df, model, w2v_model)
            print(predictions)

if __name__ == "__main__":
    main()
    # Comment / uncomment to load new tweets
    # twitterconnector.query_tweets_to_file('examples.txt', '#trump', 200)
    # tweets = twitterconnector.read_tweets_from_file('examples.txt')
    # tokenized_tweets = tweet_processor.process_raw_tweets(tweets)
    # counts = tweet_processor.count_tokens(tokenized_tweets)
    # visualizer.word_cloud_from_frequencies(counts, "example_cloud.png")

    # TODO get sentiments
    # pred = classify_tweet()
    # preds = classify_tweets(tokenized_tweets)
