#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct 20 11:42:22 2017

@author: max
"""
from nltk.tokenize import TweetTokenizer
import re
twitter_username_re = re.compile(r'@([A-Za-z0-9_]+)')
url_re = re.compile(r'(https?:\/\/(?:www\.|(?!www))[a-zA-Z0-9][a-zA-Z0-9-]+[a-zA-Z0-9]\.[^\s]{2,}|www\.[a-zA-Z0-9][a-zA-Z0-9-]+[a-zA-Z0-9]\.[^\s]{2,}|https?:\/\/(?:www\.|(?!www))[a-zA-Z0-9]\.[^\s]{2,}|www\.[a-zA-Z0-9]\.[^\s]{2,})')
email_re = re.compile(r'(^[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+$)')
tokenizer = TweetTokenizer()


tokens = ['Hello', 'there', '@lol', 'lol@lol.com', 'you', 'suck']
tweet = "Hey loser you suck"

tokens = tokenizer.tokenize(tweet)

def tokenize(tweet):
    try:
        tweet = tweet.lower()
        tokens = tokenizer.tokenize(tweet)
        tokens = [email_re.sub('EMAIL_ADDRESS', t) for t in tokens]
        tokens = [url_re.sub('URL_ADDRESS', t) for t in tokens]
        tokens = [twitter_username_re.sub('AT_USERNAME', t) for t in tokens]
        return tokens
    except:
        return 'NC'


## Tokenize the first N tweets on a dataframe 
## Specify training=true if processing training set only
def process_tweets(data, n=1600000, training = False):
    data = data.head(n)
    if training:
        data['tokens'] = data['SentimentText'].map(tokenize) 
    else:
        data = data.map(tokenize)
    data = data[data.tokens != 'NC']
    data.reset_index(inplace=True)
    data.drop('index', inplace=True, axis=1)
    return data

