from ast import literal_eval
import os
import sklearn
import gensim
import string
import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from collections import Counter
from gensim import corpora, models, similarities
from sklearn.neighbors import NearestNeighbors
from sklearn.model_selection import train_test_split
import ndcg
import MAP
from functools import reduce
import json
import time
import pickle


def to_sparse(matrix, N):
    return csr_matrix([gensim.matutils.sparse2full(row, length=N) for row in matrix]) 


def get_samples(user_id, fav_books, books_info, p, p_favs_in_test, all_):
    
    # Historias favoritas del usuario objetivo.
    user_fav_stories = fav_books[fav_books['user_id'] == user_id]
    # print(len(user_fav_stories))
    
    X_train, X_test = train_test_split(user_fav_stories, test_size=p, random_state=10)
    profile_train_stories = X_train['story_id']
    profile_test_stories = X_test['story_id']
    
    # print('len train: ', len(profile_train_stories))
    # print('len test: ', len(profile_test_stories))
    
    Nt = int(len(X_test) / p_favs_in_test) - len(X_test)

    # Sample de historias que no son favoritas del usuario objetivo.
    test_stories_nofav = books_info[~books_info['story_id'].isin(user_fav_stories['story_id'].values)]
    
    #print(test_stories_nofav)

    if not all_:
        test_samples_stories = test_stories_nofav.sample(n=Nt)


    profile_train_samples = books_info[books_info['story_id'].isin(profile_train_stories)]
    # print(len(profile_train_samples))
    #test_samples = pd.concat(
    #               [books_info[books_info['story_id'].isin(test_samples_stories)],\
    #                books_info[books_info['story_id'].isin(profile_test_stories)]])
    
    #test_samples = pd.concat([books_info[~books_info['story_id'].isin(user_fav_stories['story_id'].values)],\
    #               books_info[books_info['story_id'].isin(profile_test_stories)]])
                                      
    test_samples = pd.concat([test_samples_stories, \
                              books_info[books_info['story_id'].isin(profile_test_stories)]])

    # print('len test sample: ', len(test_samples))
    # print(len(df_stories[df_stories['story_id'].isin(test_samples_stories)]))
    # print(len(df_stories[df_stories['story_id'].isin(profile_test_stories)]))
    return profile_train_samples, test_samples


def get_recommendations(model, q_recommendations, metric, N_dict, X_train, X_test, nearest_neighbors):
    X = to_sparse(X_test[model].tolist(), N)
    document_index = NearestNeighbors(n_neighbors=(nearest_neighbors+1),\
                                      algorithm='brute', metric=metric).fit(X)
    dense_train_avg = np.array([gensim.matutils.sparse2full(row, length=N_dict) for row \
                                in X_train[model].tolist()]).mean(axis=0)
    dists, neighbors = document_index.kneighbors(dense_train_avg.reshape(1, -1))
    stories = []
    for i in neighbors[0]:
        stories.append(X_test.iloc[i]['story_id'])
    return stories


def recommend_stories_users(users, fav_books, books_info, p, Nt, **kwargs):
    recommendation_dict = {}
    counter = 0
    t1 = time.time()
    for user in users:
        train_samples, test_samples = get_samples(user, fav_books, books_info, p, Nt, False)
        recommendation = get_recommendations(kwargs['model'], kwargs['nearest_neighbors'],\
                                           kwargs['metric'], kwargs['N'], train_samples,\
                                           test_samples, kwargs['nearest_neighbors'])
        recommendation_dict[user] = recommendation
        # print('finish')
        counter += 1
        if counter % 100 == 0:
            t2 = time.time()
            print('Finished {} users  in {}'.format(counter, t2 - t1))
            t1 = time.time()

    with open('./{}_{}_{}.pickle'.format(kwargs['model'], kwargs['topics'],\
                                      kwargs['metric']), 'wb') as fp:
        pickle.dump(recommendation_dict, fp)
    return recommendation_dict
    



if __name__ == '__main__':
    #data_path = './datasets_recsys/LDA_model_books_20_feats.csv'

    #books_info = pd.read_csv(data_path, encoding='latin', sep='|')

    fav_books = pd.read_csv('./datasets_recsys/favorite_stories_books_sample.csv', encoding='utf-8', sep=';')
    # Usuarios con >= 5 historias favoritas para test.
    users_test_id = fav_books.groupby('user_id').count().sort_values(by='story_id', ascending=False).index.values

    stemm = False
    dic_file = 'lda_resources/dictionary-stemm.p' if stemm else 'lda_resources/dictionary.p'
    if os.path.isfile(dic_file):
        dictionary = corpora.dictionary.Dictionary().load(dic_file)
    else:
        dictionary = corpora.dictionary.Dictionary(documents=books_info.tokenised_abstract.tolist())
        dictionary.save(dic_file)

    # corpus = books_info['bow'].tolist()

    # corpus = books_info['bow'].apply(lambda x: literal_eval(x))

    N = len(dictionary)

    for topic_number in [10, 15, 20]:
        data_path = './datasets_recsys/books_lda_model_{}_topics.csv'.format(topic_number)
        print('Loading files...')
        books_info = pd.read_csv(data_path, encoding='latin', sep='|')
        print('Applying literal_eval...')
        books_info['lda'] = books_info['lda'].apply(lambda x: literal_eval(x))
        print('Starting recommendation...')
        for metric in ['euclidean', 'cosine']:
            args = {
            'model': 'lda',
            'nearest_neighbors': 10,
            'metric': metric,
            'N': len(dictionary),
            'topics': topic_number
            }
            users = fav_books['user_id'].unique()
            test_users = np.random.choice(users, 2000)
            t1 = time.time()
            dic = recommend_stories_users(test_users, fav_books, books_info, 0.3, 0.05, **args)
            t2 = time.time()
            print('Recommendation with {} topics and {} metric finished in {} segs'.format(topic_number, metric, t2-t1))