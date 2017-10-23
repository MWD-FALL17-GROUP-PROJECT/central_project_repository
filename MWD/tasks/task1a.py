# -*- coding: utf-8 -*-
from collections import defaultdict
from computations import decompositions
from data import DataHandler
from operator import itemgetter
from util import constants
import numpy as np

def task1a(genre, method):
    if(method=="SVD"):
        task1a_svd(genre)
    elif(method=="PCA"):
        task1a_pca(genre)
    elif(method=="LDA"):
        genre_spaceTags_LDA_tf(genre)
    else:
        print("Invalid method. Please use SVD or PCA or LDA") 
    return

def prettyPrintTagVector(vector, tagsInDf, tagIdTagsDf):
    vectorLen = len(vector)
    for index in range(0,vectorLen):
        tagId = tagsInDf[index]
        tagName = tagIdTagsDf[tagIdTagsDf['tagId']==tagId].iloc[0][1]
        print(tagName + ':' + str(vector[index]), end=', ')
    print('.')
    return

def task1a_svd(genre):
    DataHandler.vectors()
    DataHandler.createDictionaries1()
    
    genre_movie_map = DataHandler.getGenreMoviesMap()
    if (genre not in genre_movie_map.keys()):
        print("genre " + genre + " not present in data")
        return
    movie_tag_df = DataHandler.load_movie_tag_df()
    tagIdTagsDf = DataHandler.tag_id_df
    tagsInDf = list(movie_tag_df.transpose().index)
    
    movies = genre_movie_map.get(genre)
    genre_movie_tags_df = (movie_tag_df.loc[movies]).dropna(how='any')
    U, Sigma, genre_semantics = decompositions.SVDDecomposition(genre_movie_tags_df, 4)
    
    print("The 4 semantics for genre:" + genre + " are")
    index = 1
    for semantic in np.matrix(genre_semantics).tolist():
        print("semantic " + str(index) + ": ")
        prettyPrintTagVector(semantic, tagsInDf, tagIdTagsDf)
        print("")
        index = index + 1
    return

def task1a_pca(genre):
    DataHandler.vectors()
    DataHandler.createDictionaries1()
    
    genre_movie_map = DataHandler.getGenreMoviesMap()
    if (genre not in genre_movie_map.keys()):
        print("genre " + genre + " not present in data")
        return
    movie_tag_df = DataHandler.load_movie_tag_df()
    tagIdTagsDf = DataHandler.tag_id_df
    tagsInDf = list(movie_tag_df.transpose().index)
    
    movies = genre_movie_map.get(genre)
    genre_movie_tags_df = (movie_tag_df.loc[movies]).dropna(how='any')
    genre_semantics = decompositions.PCADecomposition(genre_movie_tags_df, 4)
    
    print("The 4 semantics for genre:" + genre + " are")
    index = 1
    for semantic in np.matrix(genre_semantics).tolist():
        print("semantic " + str(index) + ": ")
        prettyPrintTagVector(semantic, tagsInDf, tagIdTagsDf)
        index = index + 1
    return


def genre_spaceTags_LDA_tf(genre):
    movie_tag_map,tag_id_map,actor_movie_rank_map,movie_actor_rank_map = DataHandler.get_dicts()
    genre_movie_map = DataHandler.getGenreMoviesMap()
    if (genre not in genre_movie_map.keys()):
        print("genre " + genre + " not present in data")
        return
    df = DataHandler.load_genre_matrix_tf(genre)
    ldaModel,doc_term_matrix,id_Term_map  =  decompositions.LDADecomposition(df,4,constants.genreTagsSpacePasses)
    topic_terms = defaultdict(set)
    for i in range(0,4):
        for tuples in ldaModel.get_topic_terms(i):#get_topics_terms returns top n(default = 10) words of the topics
            term = tag_id_map.get(id_Term_map.get(tuples[0]))
            topic_terms[i].add((term,tuples[1]))
    for i in range(0,4):
        print(sorted(topic_terms.get(i),key = itemgetter(1),reverse=True))
        print('\n')
