"""Module for connecting to twitter and retrieving tweets
"""
import argparse
import sys
import csv
import credentials
import tweepy


def authenticate():
    """Function for auhtentication to Twitter. Requires credentials.py at the
    same directory.
    """
    auth = tweepy.OAuthHandler(
        credentials.consumer_key, credentials.consumer_secret)
    auth.set_access_token(credentials.access_token_key,
                          credentials.access_token_secret)
    api = tweepy.API(auth)
    return api


def getTweetsAsList(hashtag, max_tweets=300):
    """Gets 100 tweets with a hashtag. Removes linebreaks from tweets.
    The rows at list needs to be decoded with decode()
    """
    api = authenticate()
    tweet_texts = []
    for tweet in tweepy.Cursor(api.search, q=hashtag, lang="en", rpp=100).items(max_tweets):
        tweet_text = tweet.text.replace('\r', '').replace(
            '\n', ' ').encode('utf-8', errors='ignore')
        tweet_texts.append(tweet_text)
    # output to console for debugging
    # for text in tweet_texts:
        # print(text.decode())
    return tweet_texts


def getTweetsAsCSV(filename, hashtag):
    """ Gets 100 tweets with a hashtag and saves them to a csv file, one tweet per row.
    The rows at CSV needs to be decoded with decode()
    """
    tweetlist = getTweetsAsList(hashtag)
    with open(filename, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        for item in tweetlist:
            writer.writerow([item])


def query_tweets(query, nresults=300):
    """Queries twitter for tweets and returns list of raw text contents"""
    api = authenticate()
    return [tweet.text.replace('\r', '').replace('\n', ' ')
            for tweet in tweepy.Cursor(api.search, q=query, lang="en").items(nresults)]


def query_tweets_to_file(outfile, query, nresults=300):
    """Queries twitter for tweets and saves raw text contents to file

    outfile accepts either a path or an opened file
    """
    try:
        with open(outfile, 'w', encoding='utf-8') as file:
            for tweet in query_tweets(query, nresults):
                file.write(tweet + '\n')
    except TypeError:
        for tweet in query_tweets(query, nresults):
            outfile.write(tweet + '\n')


def read_tweets_from_file(infile):
    """Reads tweets from file made with query_tweets_to_file

    infile accepts either a path or an opened file
    """
    try:
        with open(infile, 'r', encoding='utf-8') as file:
            return file.readlines()
    except TypeError:
        return infile.readlines()


def main():
    """Allows queries and file reading from command line"""
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest='operation')

    qparser = subparsers.add_parser('query',
                                    description='Queries tweets from the API')
    qparser.add_argument('query')
    qparser.add_argument('-n', type=int, default=100, dest='count')
    qparser.add_argument('-p', type=argparse.FileType('w', encoding='utf-8'),
                         default=sys.stdout, dest='outfile')

    lparser = subparsers.add_parser('read',
                                    description='Reads tweets from a file')
    lparser.add_argument('infile',
                         type=argparse.FileType('r', encoding='utf-8'))

    args = parser.parse_args()
    if args.operation == 'query':
        query_tweets_to_file(args.outfile, args.query, args.count)
    elif args.operation == 'read':
        for tweet in read_tweets_from_file(args.infile):
            print(tweet, end="")


if __name__ == '__main__':
    main()
