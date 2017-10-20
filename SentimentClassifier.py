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
from tweet_processor import process_tweets
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

def convert_data_to_index(string_data, wv):
    indexed_sequence = []
    for word in string_data:
        if str(word) in wv:
            indexed_sequence.append(wv.vocab[word].index)
    return indexed_sequence

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

def convert_to_index(tokens):
    sequence = []
    for t in tokens:
        if t in w2v_model.wv.vocab:
            sequence.append(w2v_model.wv.vocab[t].index)
    return sequence

data = read_data("Sentiment Analysis Dataset.csv")
data = process_tweets(data, training=True)


#w2v_model = gensim.models.Word2Vec(size=200, min_count=10)
#w2v_model.build_vocab([x.words for x in tqdm(data['tokens'])])
#w2v_model.train([x.words for x in tqdm(data['tokens'])], total_examples=len(data), epochs=w2v_model.iter)

#This works
#w2v_model = gensim.models.Word2Vec(data['tokens'], size=300, window=6, min_count=5, workers=16)
#w2v_model.save("" + "w2v_model")
w2v_model = gensim.models.Word2Vec.load("" + "w2v_model")
w2v_model.wv.most_similar_cosmul(positive=['facebook'])
#sequences = [convert_to_index(s) for s in data['tokens']]
from keras.preprocessing.sequence import pad_sequences
sequences = pad_sequences(sequences, maxlen=16)
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(np.array(sequences),
                                                    np.array(data.Sentiment),
                                                    test_size=0.2)


## Convert wv word vectors to np matrix
## So we can insert it into Keras model somehow
embedding_matrix = np.zeros((len(w2v_model.wv.vocab), 300))
for i in range(len(w2v_model.wv.vocab)):
    embedding_vector = w2v_model.wv[w2v_model.wv.index2word[i]]
    if embedding_vector is not None:
        embedding_matrix[i] = embedding_vector


##Not used currently, could boost performance
#from sklearn.feature_extraction.text import TfidfVectorizer
#vectorizer = TfidfVectorizer(analyzer=lambda x: x, min_df=10)
#matrix = vectorizer.fit_transform([x.words for x in X_train])
#tfidf = dict(zip(vectorizer.get_feature_names(), vectorizer.idf_))
#len(tfidf)

## This was for the old ANN model, dont delete yet
#from sklearn.preprocessing import scale
#train_vecs_w2v = np.concatenate([buildWordVector(z, 200) for z in tqdm(map(lambda x: x.words, X_train))])
#train_vecs_w2v = scale(train_vecs_w2v)

#test_vecs_w2v = np.concatenate([buildWordVector(z, 200) for z in tqdm(map(lambda x: x.words, X_test))])
#test_vecs_w2v = scale(test_vecs_w2v)




from keras.layers import Conv1D, Flatten, Dropout, Dense, BatchNormalization, LeakyReLU, MaxPooling1D
from keras.layers.embeddings import Embedding
from keras.models import Sequential
#len(w2v_model.wv.vocab)
#embedding_layer = Embedding((len(w2v_model.wv.vocab) + 1),
#                            200,
#                            weights=embedding_matrix)

embedding_matrix.shape
### END WIP ##
len(w2v_model.wv.vocab)
model = Sequential()
model.add(Embedding(input_dim=embedding_matrix.shape[0], output_dim=embedding_matrix.shape[1],
                    weights=[embedding_matrix],
                    input_length=16))
model.add(Conv1D(128, 3, activation='elu', border_mode='same'))
model.add(BatchNormalization())
model.add(Dropout(0.3))
model.add(Conv1D(128, 3, activation='elu', border_mode='same'))
model.add(BatchNormalization())
model.add(Dropout(0.3))
model.add(Conv1D(128, 3, activation='elu', border_mode='same'))
model.add(BatchNormalization())
model.add(Dropout(0.3))
model.add(Conv1D(128, 3, activation='elu', border_mode='same'))
model.add(BatchNormalization())
model.add(Dropout(0.3))
model.add(Conv1D(128, 3, activation='elu', border_mode='same'))
model.add(BatchNormalization())
model.add(Dropout(0.3))
model.add(MaxPooling1D())
model.add(Conv1D(256, 3, activation='elu', border_mode='same'))
model.add(BatchNormalization())
model.add(Dropout(0.4))
model.add(Flatten())
model.add(Dense(256,activation='elu'))
model.add(Dropout(0.4))
model.add(Dense(256,activation='elu'))
model.add(Dropout(0.4))
model.add(Dense(256,activation='elu'))
model.add(Dropout(0.4))
model.add(Dense(1,activation='sigmoid'))
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
model.layers[0].trainable = False

model.fit(X_train, y_train, batch_size=2048, epochs = 30,
          validation_split=0.2, callbacks=[])

score = model.evaluate(X_test, y_test, batch_size=128, verbose=2)
score
print(score) # currently about 80% val acc

