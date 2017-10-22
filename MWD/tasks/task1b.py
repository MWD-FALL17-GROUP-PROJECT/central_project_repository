from collections import defaultdict
from computations import decompositions
from data import DataHandler
from operator import itemgetter
from util import constants
import numpy as np
import pandas as pd

def task1b(genre, method):
    if(method=="SVD"):
        task1b_svd(genre)
    elif(method=="PCA"):
        task1b_pca(genre)
    elif(method=="LDA"):
        genre_spaceActors_LDA_tf(genre)
    else:
        print("Invalid method. Please use SVD or PCA or LDA") 
    return

def prettyPrintActorVector(vector, actorsInDf, actorIdActorsDf):
    vectorLen = len(vector)
    for index in range(0, vectorLen):
        actorId = actorsInDf[index]
        actorName = actorIdActorsDf[actorIdActorsDf['id']==actorId].iloc[0][1]
        print(actorName + ": " + str(vector[index]), end=', ')
    print('.')

def task1b_svd(genre):
    DataHandler.vectors()
    DataHandler.createDictionaries1()
    
    actorIdActorsDf = DataHandler.actor_info_df
    
    genre_actor_tags_df = DataHandler.load_genre_actor_matrix(genre)
    actorsInDf = list(genre_actor_tags_df.transpose().index)
    genre_semantics = decompositions.PCADecomposition(genre_actor_tags_df, 5)
    
    print("The 5 semantics for genre:" + genre + " are")
    index = 1
    for semantic in np.matrix(genre_semantics).tolist():
        print("semantic " + str(index) + ": ")
        prettyPrintActorVector(semantic, actorsInDf, actorIdActorsDf)
        index = index + 1
    return

def task1b_pca(genre):
    DataHandler.vectors()
    DataHandler.createDictionaries1()
    
    actorIdActorsDf = DataHandler.actor_info_df
    
    genre_actor_tags_df = DataHandler.load_genre_actor_matrix(genre)
    actorsInDf = list(genre_actor_tags_df.transpose().index)
    u, sigma, genre_semantics = decompositions.SVDDecomposition(genre_actor_tags_df, 5)
    
    print("The 5 semantics for genre:" + genre + " are")
    index = 1
    for semantic in np.matrix(genre_semantics).tolist():
        print("semantic " + str(index) + ": ")
        prettyPrintActorVector(semantic, actorsInDf, actorIdActorsDf)
        index = index + 1
    return

def genre_spaceActors_LDA_tf(genre):
    DataHandler.vectors()
    movie_tag_map,tag_id_map,actor_movie_rank_map,movie_actor_rank_map = DataHandler.get_dicts()
    df = DataHandler.load_genre_actor_matrix_tf(genre)
    ldaModel,doc_term_matrix,id_Term_map  =  decompositions.LDADecomposition(df,5,constants.genreActorSpacePasses)
    topic_terms = defaultdict(set)
    for i in range(0,5):
        for tuples in ldaModel.get_topic_terms(i):#get_topics_terms returns top n(default = 10) words of the topics
            term = id_Term_map.get(tuples[0])
            topic_terms[i].add((term,tuples[1]))
    for i in range(0,5):
        print(sorted(topic_terms.get(i),key = itemgetter(1),reverse=True))
        print('\n')