# -*- coding: utf-8 -*-
"""
Created on Sun Oct 15 20:16:59 2017

@author: Crazyfist
"""
import pandas as pd
import numpy as np
import gensim
from gensim.models.doc2vec import TaggedDocument
from tqdm import tqdm
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

def tagTweets(tweets, tag_type):
    tagged = []
    for i,v in tqdm(enumerate(tweets)):
        tag = '%s_%s'%(tag_type,i)
        tagged.append(TaggedDocument(v, [tag]))
    return tagged

def buildWordVector(tokens, size):
    vec = np.zeros(size).reshape((1, size))
    count = 0.
    for word in tokens:
        try:
            vec += w2v_model[word].reshape((1, size)) * tfidf[word]
            count += 1.
        except KeyError: # handling the case where the token is not
                         # in the corpus. useful for testing.
            continue
    if count != 0:
        vec /= count
    return vec

data = read_data("Sentiment Analysis Dataset.csv")
data = process(data)

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(np.array(data.head(1000000).tokens),
                                                    np.array(data.head(1000000).Sentiment),
                                                    test_size=0.2)

X_train = tagTweets(X_train, 'TRAIN')
X_test = tagTweets(X_test, 'TEST')

#### WIP plz ignore ( or help )

#index_data = convert_data_to_index(sentences, w2v_model.wv)
#print(sentences[:4], index_data[:4])


from sklearn.feature_extraction.text import TfidfVectorizer
vectorizer = TfidfVectorizer(analyzer=lambda x: x, min_df=10)
matrix = vectorizer.fit_transform([x.words for x in X_train])
tfidf = dict(zip(vectorizer.get_feature_names(), vectorizer.idf_))
len(tfidf)

X_train[0]

w2v_model = gensim.models.Word2Vec(size=200, min_count=10)
w2v_model.build_vocab([x.words for x in tqdm(X_train)])
w2v_model.train([x.words for x in tqdm(X_train)], total_examples=len(X_train), epochs=w2v_model.iter)


from sklearn.preprocessing import scale
train_vecs_w2v = np.concatenate([buildWordVector(z, 200) for z in tqdm(map(lambda x: x.words, X_train))])
train_vecs_w2v = scale(train_vecs_w2v)

test_vecs_w2v = np.concatenate([buildWordVector(z, 200) for z in tqdm(map(lambda x: x.words, X_test))])
test_vecs_w2v = scale(test_vecs_w2v)

#w2v_model = gensim.models.Word2Vec(sentences, size=200, window=5, min_count=5, workers=16)
#w2v_model.wv.most_similar_cosmul(positive=['facebook'])

#w2v_model.save('/' + 'w2vmodel_1')
#model = gensim.models.Word2Vec.load('/' + "w2vmodel_1")

#IMPORTANT needed for CNN somehow (for embedding layer)
## Convert wv word vectors to np matrix
## So we can insert it into Keras model somehow
embedding_matrix = np.zeros((len(w2v_model.wv.vocab), 200))
for i in range(len(w2v_model.wv.vocab)):
    embedding_vector = w2v_model.wv[w2v_model.wv.index2word[i]]
    if embedding_vector is not None:
        embedding_matrix[i] = embedding_vector
        
        

#from keras.preprocessing.sequence import pad_sequences
#sentences = pad_sequences(index_data, maxlen=max_tweet_length)

#padded = pad_sequences(sentences, maxlen=140)


from keras.layers import Conv1D, Flatten, Dropout, Dense
from keras.layers.embeddings import Embedding
from keras.models import Sequential
#len(w2v_model.wv.vocab)
#embedding_layer = Embedding((len(w2v_model.wv.vocab) + 1),
#                            200,
#                            weights=embedding_matrix)

 
### END WIP ##
len(w2v_model.wv.vocab)
model = Sequential()
#model.add(Conv1D(64, 3, input_dim=200, border_mode='same'))
#model.add(Conv1D(32, 3, border_mode='same'))
#model.add(Conv1D(16, 3, border_mode='same'))
#model.add(Flatten())
model.add(Dense(256,activation='relu', input_dim=200))
model.add(Dropout(0.4))
model.add(Dense(256,activation='relu'))
model.add(Dropout(0.4))
model.add(Dense(128,activation='relu'))
model.add(Dropout(0.3))
model.add(Dense(1,activation='sigmoid'))
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

model.fit(train_vecs_w2v, y_train, batch_size = 256, epochs = 20)

score = model.evaluate(test_vecs_w2v, y_test, batch_size=128, verbose=2)
print(score) # currently about 80% val acc
