# -*- coding: utf-8 -*-

from computations import decompositions
from data import DataHandler
import pandas as pd
from collections import defaultdict
from operator import itemgetter
from util import constants
import numpy as np

DataHandler.vectors()

def genre_spaceTags_LDA(genre):
    movie_tag_map,tag_id_map,actor_movie_rank_map,movie_actor_rank_map = DataHandler.get_dicts()
    df = DataHandler.load_genre_matrix(genre)
    ldaModel,doc_term_matrix,id_Term_map  =  decompositions.LDADecomposition(df,5,constants.genreTagsSpacePasses)
    topic_terms = defaultdict(set)
    for i in range(0,5):
        for tuples in ldaModel.get_topic_terms(i):#get_topics_terms returns top n(default = 10) words of the topics
            term = tag_id_map.get(id_Term_map.get(tuples[0]))
            topic_terms[i].add((term,tuples[1]))
    for i in range(0,5):
        print(sorted(topic_terms.get(i),key = itemgetter(1),reverse=True))
        print('\n')

def genre_spaceActors_LDA(genre):
    movie_tag_map,tag_id_map,actor_movie_rank_map,movie_actor_rank_map = DataHandler.get_dicts()
    df = DataHandler.load_genre_actor_matrix(genre)
    ldaModel,doc_term_matrix,id_Term_map  =  decompositions.LDADecomposition(df,constants.genreActorSpacePasses)
    topic_terms = defaultdict(set)
    for i in range(0,5):
        for tuples in ldaModel.get_topic_terms(i):#get_topics_terms returns top n(default = 10) words of the topics
            term = id_Term_map.get(tuples[0])
            topic_terms[i].add((term,tuples[1]))
    for i in range(0,5):
        print(sorted(topic_terms.get(i),key = itemgetter(1),reverse=True))
        print('\n')
        
def top10_Actors_LDA(givenActor):
    DataHandler.create_actor_actorid_map()
    top10SimilarActors_similarity = DataHandler.similarActors_LDA(givenActor)
    print('Actors similar to '+str(DataHandler.actor_actorid_map[givenActor]))
    for actor,sim in top10SimilarActors_similarity:
        print(DataHandler.actor_actorid_map[actor]+' '+str(sim))
        
def top5LatentCP(tensorIdentifier, space):
    if (tensorIdentifier == 'AMY'):
        u = decompositions.CPDecomposition(DataHandler.getTensor_ActorMovieYear(),constants.RANK)
        if (space == 'Actor'):
            actorRank = np.array(u[0])
            print(actorRank.T)
            return
        if (space == 'Movie'):
            movieRank = np.array(u[1])
            print(movieRank.T)
            return
        if (space == 'Year'):
            YearRank = np.array(u[0])
            print(YearRank.T)
            return
        else:
            print('Wrong Space')
            return
    if (tensorIdentifier == 'TMR'):
        u = decompositions.CPDecomposition(DataHandler.getTensor_TagMovieRanking(),constants.RANK)
        if (space == 'Tag'):
            tagRank = np.array(u[0])
            print(tagRank.T)
            return
        if (space == 'Movie'):
            movieRank = np.array(u[1])
            print(movieRank.T)
            return
        if (space == 'Ranking'):
            RankingRank = np.array(u[0])
            print(RankingRank.T)
            return
        else:
            print('Wrong Space')
            return
    else:
        print('Wrong Tensor Identifier')