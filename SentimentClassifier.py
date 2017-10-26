# -*- coding: utf-8 -*-
"""
Created on Sun Oct 15 20:16:59 2017

@author: madkoppa
"""
import pandas as pd
import numpy as np
import gensim
from gensim.models.doc2vec import TaggedDocument
from tqdm import tqdm
from keras.preprocessing import sequence
from nltk.tokenize import TweetTokenizer
from tweet_processor import process_tweets
from keras.layers import Conv1D, Flatten, Dropout, Dense, BatchNormalization, LeakyReLU, MaxPooling1D
from keras.layers.embeddings import Embedding
from keras.models import Sequential
from keras.optimizers import Nadam
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import GridSearchCV
from keras.models import load_model

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

def convert_to_index(tokens, w2v):
    sequence = []
    for t in tokens:
        if t in w2v.wv.vocab:
            sequence.append(w2v.wv.vocab[t].index)
    return sequence

def build_classifier(optimizer='rmsprop', activation='relu'):
    model = Sequential()
    model.add(Embedding(input_dim=embedding_matrix.shape[0], output_dim=embedding_matrix.shape[1],
                        weights=[embedding_matrix],
                        input_length=16))
    model.add(Conv1D(1024, 3, activation=activation, padding='same'))
    model.add(BatchNormalization())
    model.add(Dropout(0.5))
    model.add(Conv1D(512, 3, activation=activation, padding='same'))
    model.add(BatchNormalization())
    model.add(Dropout(0.5))
    model.add(Conv1D(256, 3, activation=activation, padding='same'))
    model.add(BatchNormalization())
    model.add(Dropout(0.5))
    model.add(Flatten())
    model.add(Dense(256, activation=activation))
    model.add(BatchNormalization())
    model.add(Dropout(0.5))
    model.add(Dense(1,activation='sigmoid'))
    model.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=['accuracy'])
    model.layers[0].trainable = False
    return model

## Convert wv word vectors to np matrix
## So we can insert it into Keras embeddings layer as input
def create_embedding_matrix(w2v_model):
    embedding_matrix = np.zeros((len(w2v_model.wv.vocab), 300))
    for i in range(len(w2v_model.wv.vocab)):
        embedding_vector = w2v_model.wv[w2v_model.wv.index2word[i]]
        if embedding_vector is not None:
            embedding_matrix[i] = embedding_vector

def classify_tweets(tokenized_tweets, model, w2v):
    sequences = [convert_to_index(s, w2v) for s in tokenized_tweets['tokens']]
    from keras.preprocessing.sequence import pad_sequences
    sequences = pad_sequences(sequences, maxlen=16)
    predictions = model.predict(sequences, batch_size=32, verbose=1)
    preds = {"pos":0, "neu": 0, "neg":0}
    for p in predictions:
        if p>0.65:
            preds["pos"] = preds["pos"] + 1
        elif p <0.35:
            preds["neu"] = preds["neu"] + 1
        else:
            preds["neg"] = preds["neg"] + 1
    return preds

def load_models():
    cnn = load_model('tweet_w2v_cnn_85.h5')
    w2v_model = gensim.models.Word2Vec.load("" + "w2v_model")
    return cnn, w2v_model

#Todo split this, currently only in method so it wont run on import
def training_code():
    data = read_data("Sentiment Analysis Dataset.csv")
    data = process_tweets(data, training=True)
    
    
    #Train, save or load w2v model
    #w2v_model = gensim.models.Word2Vec(data['tokens'], size=300, window=6, min_count=5, workers=16)
    #w2v_model.save("" + "w2v_model")
    w2v_model = gensim.models.Word2Vec.load("" + "w2v_model")
    #w2v_model.wv.most_similar_cosmul(positive=['facebook'])
    
    #
    sequences = [convert_to_index(s) for s in data['tokens']]
    from keras.preprocessing.sequence import pad_sequences
    sequences = pad_sequences(sequences, maxlen=16)
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(np.array(sequences),
                                                        np.array(data.Sentiment),
                                                        test_size=0.2)
    
    embedding_matrix = create_embedding_matrix(w2v_model)
    
    
    model = build_classifier(optimizer='rmsprop', activation='relu')
    
    model.fit(X_train, y_train, batch_size=8000, epochs = 10,
              validation_split=0.1, verbose=2)
    
    score = model.evaluate(X_test, y_test, batch_size=512, verbose=1)
    print(score) # currently about 83% val acc
    #model.save('tweet_w2v_cnn_83')
    model = load_model('tweet_w2v_cnn_85.h5')
    
    nadam = Nadam()
    classifier = KerasClassifier(build_fn = build_classifier)
    parameters = {'optimizer': ['adam', 'rmsprop'],
                  'activation': ['relu', 'elu', 'sigmoid', 'tanh', 'selu'],
                  'epochs': [5],
                  'batch_size': [4096]}
    grid_search = GridSearchCV(estimator = classifier,
                               param_grid = parameters,
                               scoring = 'accuracy',
                               cv = 3)
    grid_search = grid_search.fit(X_train, y_train)
    best_parameters = grid_search.best_params_
    best_accuracy = grid_search.best_score_

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

