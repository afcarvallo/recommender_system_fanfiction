import os
import nltk
import sklearn
import gensim
import string
import numpy as np
import pandas as pd
from nltk.corpus import stopwords
from scipy.sparse import csr_matrix
from collections import Counter
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer 
from gensim import corpora, models, similarities
from sklearn.neighbors import NearestNeighbors
from sklearn.model_selection import train_test_split
import ndcg
import MAP
from functools import reduce
import json
import time
from ast import literal_eval




if __name__ == '__main__':

    data_path = './datasets_recsys/books_info_bow.csv'
    books_info = pd.read_csv(data_path, encoding='latin', sep='|')

    corpus = books_info['bow'].apply(lambda x: literal_eval(x))

    for topic_number in [10, 15, 20]:
        lda_model = models.ldamulticore.LdaMulticore(corpus, num_topics=topic_number, workers=3)
        books_info['lda'] = lda_model[corpus.tolist()]
        books_info[['story_title', 'story_id', 'lda']].to_csv('./datasets_recsys/books_lda_model_{}_topics.csv'.format(topic_number), sep='|', index=False)