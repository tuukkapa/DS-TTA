"""Driver module for running the program"""

import argparse
import os
import sys
import tweet_processor
import twitterconnector
import visualizer


def query_tweets_to_files(queries, count):
    for query in queries:
        twitterconnector.query_tweets_to_file(
            f'tweets/{query}.txt', query, count)


def analyze_tweets(tweets):
    """Analyze tweets

    Tweets is expected to be list of tuples (topic, tweets)
    """
    # TODO DO EVERYTHING HERE
    print(tweets)


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
    pparser.add_argument('-n', type=int, default=100, dest='count')

    gparser = subparsers.add_parser('get',
                                    description='Gets tweets and writes them into files')
    gparser.add_argument('queries', nargs='+')
    gparser.add_argument('-n', type=int, default=100, dest='count')

    args = parser.parse_args()
    if args.operation == 'get':
        query_tweets_to_files(args.queries, args.count)
    elif args.operation == 'process':
        if args.queries:
            tweets = [(query,
                       twitterconnector.query_tweets(query, args.count)) for query in args.queries]
        else:
            tweets = [(os.path.splitext(os.path.basename(f))[0],
                       twitterconnector.read_tweets_from_file(f)) for f in args.files]
        analyze_tweets(tweets)


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
