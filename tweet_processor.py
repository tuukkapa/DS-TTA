#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct 20 11:42:22 2017

@author: max
"""
from collections import Counter
import re
import string
from nltk.corpus import stopwords
from nltk.tokenize import TweetTokenizer
import numpy as np
import pandas as pd

twitter_username_re = re.compile(r'@([A-Za-z0-9_]+)')
url_re = re.compile(
    r'(https?:\/\/(?:www\.|(?!www))[a-zA-Z0-9][a-zA-Z0-9-]+[a-zA-Z0-9]\.[^\s]{2,}|www\.[a-zA-Z0-9][a-zA-Z0-9-]+[a-zA-Z0-9]\.[^\s]{2,}|https?:\/\/(?:www\.|(?!www))[a-zA-Z0-9]\.[^\s]{2,}|www\.[a-zA-Z0-9]\.[^\s]{2,})')
email_re = re.compile(r'(^[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+$)')
tokenizer = TweetTokenizer()


tokens = ['Hello', 'there', '@lol', 'lol@lol.com', 'you', 'suck']
tweet = "Hey loser you suck"

tokens = tokenizer.tokenize(tweet)


def tokenize(tweet):
    """ Tokenize a tweet(string) e.g "Matt SUCKS!" => ['Matt', 'SUCKS!']
        Also replaces emails, urls and @usernames with their own tokens """
    try:
        tweet = tweet.lower()
        tokens = tokenizer.tokenize(tweet)
        tokens = [email_re.sub('EMAIL_ADDRESS', t) for t in tokens]
        tokens = [url_re.sub('URL_ADDRESS', t) for t in tokens]
        tokens = [twitter_username_re.sub('AT_USERNAME', t) for t in tokens]
        return tokens
    except:
        return 'NC'


def process_tweets(data, n=1600000, training=False):
    """ Tokenize the first N tweets on a dataframe
        Specify training=true if processing training set only """
    data = data.head(n)
    if training:
        data['tokens'] = data['SentimentText'].map(tokenize)
    else:
        data['tokens'] = data['col'].map(tokenize)
    data = data[data.tokens != 'NC']
    data.reset_index(inplace=True)
    data.drop('index', inplace=True, axis=1)
    return data


def process_raw_tweets(tweets):
    """Cleans and tokenizes tweets and returns them in a dataframe"""
    tokenizer = TweetTokenizer(preserve_case=False, strip_handles=True)

    def process_raw_tweet(tweet):
        """Converts a single tweet into cleaned tokenized form"""
        bad_words = set(stopwords.words('english')) | set(
            string.punctuation) | {'rt', "’", '"', "'"}

        # Proved to be more difficult than anticipated, replaced with regex for
        # non_standard_character
        """emoji = re.compile("["
                                 u"\U0001F600-\U0001F64F"  # emoticons
                                 u"\U0001F300-\U0001F5FF"  # symbols & pictographs
                                 u"\U0001F900-\U0001F9FF"  # suplemental symbols & pictographs
                                 u"\U00002600-\U000026FF"  # miscellaneous Symbols
                                 u"\U00002700-\U000027BF"  # dingbats
                                 u"\U0001F680-\U0001F6FF"  # transport & map symbols
                                 u"\U0001F1E0-\U0001F1FF"  # flags
                                 "]+")"""

        non_standard_character = r'[^\u0020-\u02FF]'

        url = r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'

        imagefile = r'(?:([^:/?#]+):)?(?://([^/?#]*))?([^?#]*\.(?:jpg|gif|png))(?:\?([^#]*))?(?:#(.*))?'

        truncated_word = r'.*…'

        bad_word_regexes = [non_standard_character,
                            url, imagefile, truncated_word]
        return [token for token in tokenizer.tokenize(tweet) if token not in bad_words and not any(re.match(regex, token) for regex in bad_word_regexes)]

    return pd.DataFrame({'tokens': list(map(process_raw_tweet, tweets))})


def count_tokens(data_frame):
    """Returns a dictionary of token frequencies in the given dataframe"""
    counter = Counter()
    for tokens in data_frame['tokens']:
        counter.update(tokens)
    return counter
