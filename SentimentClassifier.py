# -*- coding: utf-8 -*-
"""
Created on Sun Oct 15 20:16:59 2017

@author: Crazyfist
"""
import pandas as pd
import numpy as np
import gensim
from keras.preprocessing import sequence
from nltk.tokenize import TweetTokenizer
tokenizer = TweetTokenizer()
max_tweet_length = 140


## Function to read a file of tweets and keep only sentiment and sentimenttext 
def read_data(file):
    data = pd.read_csv(file, error_bad_lines=False)
    data.drop(['ItemID', 'SentimentSource'], axis=1, inplace=True)
    data = data[data.Sentiment.isnull() == False]
    data['Sentiment'] = data['Sentiment'].map(int)
    data = data[data['SentimentText'].isnull() == False]
    data.reset_index(inplace=True)
    data.drop('index', axis=1, inplace=True)
    #print('dataset loaded with shape' + data.shape)    
    return data

## Tokenize a tweet(string) e.g "Matt SUCKS!" => ['Matt', 'SUCKS!']
## Would be good to replace @someusers with AT_USER string etc 
def tokenize(tweet):
    try:
        tweet = tweet.lower()
        tokens = tokenizer.tokenize(tweet)
        #tokens = filter(lambda t: not t.startswith('@'), tokens)
        #tokens = filter(lambda t: not t.startswith('#'), tokens)
        #tokens = filter(lambda t: not t.startswith('http'), tokens)
        return tokens
    except:
        return 'NC'
    
## Tokenize the first N tweets on a dataframe    
def process(data, n=1000000):
    data = data.head(n)
    data['tokens'] = data['SentimentText'].map(tokenize)
    data = data[data.tokens != 'NC']
    data.reset_index(inplace=True)
    data.drop('index', inplace=True, axis=1)
    return data

def convert_data_to_index(string_data, wv):
    index_data = []
    for word in string_data:
        if str(word) in wv:
            index_data.append(wv.vocab[word].index)
    return index_data

data = read_data("Sentiment Analysis Dataset.csv")
data = process(data)

#### WIP plz ignore ( or help )
sentences = data['tokens'].head(800000)
w2v_model = gensim.models.Word2Vec(sentences, size=200, window=5, min_count=5, workers=16)
w2v_model.wv.most_similar_cosmul(positive=['facebook'])

index_data = convert_data_to_index(sentences, w2v_model.wv)
print(sentences[:4], index_data[:4])

#w2v_model.save('/' + 'w2vmodel_1')
#model = gensim.models.Word2Vec.load('/' + "w2vmodel_1")

#IMPORTANT
## Convert wv word vectors to np matrix
## So we can insert it into Keras model somehow
embedding_matrix = np.zeros((len(w2v_model.wv.vocab), 200))
for i in range(len(w2v_model.wv.vocab)):
    embedding_vector = w2v_model.wv[w2v_model.wv.index2word[i]]
    if embedding_vector is not None:
        embedding_matrix[i] = embedding_vector

from keras.preprocessing.sequence import pad_sequences
sentences = pad_sequences(index_data, maxlen=max_tweet_length)

#padded = pad_sequences(sentences, maxlen=140)
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(np.array(data.head(1000000).tokens),
                                                    np.array(data.head(1000000).Sentiment),
                                                    test_size=0.2)

from keras.layers import Convolution1D, Flatten, Dropout, Dense
from keras.layers.embeddings import Embedding
from keras.models import Sequential
len(w2v_model.wv.vocab)
embedding_layer = Embedding((len(w2v_model.wv.vocab) + 1),
                            200,
                            weights=embedding_matrix)


### END WIP ##

model = Sequential()
model.add(embedding_layer)
model.add(Convolution1D(64, 3, border_mode='same'))
model.add(Convolution1D(32, 3, border_mode='same'))
model.add(Convolution1D(16, 3, border_mode='same'))
model.add(Flatten())
model.add(Dropout(0.2))
model.add(Dense(180,activation='sigmoid'))
model.add(Dropout(0.2))
model.add(Dense(1,activation='sigmoid'))
model.layers[1]
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

model.fit(X_train, y_train, batch_size = 32, epochs = 10)
