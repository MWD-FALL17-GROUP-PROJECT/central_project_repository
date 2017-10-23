# -*- coding: utf-8 -*-

from computations import tasksBusiness
from data import DataHandler

def task3a(seed):
    DataHandler.createDictionaries1()
    actor_movie_rank_map = DataHandler.actor_movie_rank_map
    for s in seed:
        if s not in actor_movie_rank_map:
            print('Invalid seed actor id : '+str(s))
            return
    tasksBusiness.PersnalizedPageRank_top10_SimilarActors(seed)
    
def task3b(seed):
    DataHandler.createDictionaries1()
    actor_movie_rank_map = DataHandler.actor_movie_rank_map
    for s in seed:
        if s not in actor_movie_rank_map:
            print('Invalid seed actor id : '+str(s))
            return
    tasksBusiness.PersnalizedPageRank_top10_SimilarCoActors(seed)
