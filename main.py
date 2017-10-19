# This file is currently used for testing twitterconnector

import twitterconnector
import SentimentClassifier as sc
import sys
import pandas as pd

if __name__ == "__main__":
    #list = twitterconnector.getTweetsAsList(sys.argv[1])
    list = twitterconnector.getTweetsAsList('#DataScience')
    listAsDF = pd.DataFrame({'col':list})
    tokenizedDF = sc.process(listAsDF)

    # TODO visualize tokenizedDF