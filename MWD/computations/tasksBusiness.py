# -*- coding: utf-8 -*-

from computations import decompositions,pagerank
from data import DataHandler
import pandas as pd
from collections import defaultdict
from operator import itemgetter
from util import constants
import numpy as np
from computations import metrics
import operator

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

'''
prints the top 10 actor names with their cosine similarity weight related to
an input actor id based on top 5 underlying semantics
Underlying decomposition for semantic extraction is SVD.
Library used for decomposition is scikit-learn
'''
def actor_task1c_SVD(actor_id):
    DataHandler.vectors()
    acdf = DataHandler.actor_tag_df()
    indexList=list(acdf.index)
    U, Sigma, VT = decompositions.SVDDecomposition(acdf, 5)
    
    simAndActor = []
    for index in range(0, len(U)):
        simAndActor.append((metrics.cosineSim(U[indexList.index(actor_id)], U[index]), indexList[index]))
    
    result = sorted(simAndActor, key=operator.itemgetter(0), reverse=True)
    resultNames = []
    DataHandler.create_actor_actorid_map()    
    for weightActorTuple in result:
        if (weightActorTuple[1]!=actor_id):
            resultNames.append((weightActorTuple[0], DataHandler.actor_actorid_map.get(weightActorTuple[1])))
    print("Actors similar to " + str(DataHandler.actor_actorid_map.get(actor_id)) + " are:")
    print(resultNames[0:10])
    return

def task1dImplementation_SVD(movie_id):
    DataHandler.vectors()
    actor_tag_df = DataHandler.actor_tag_df()
    movie_tag_df = DataHandler.load_movie_tag_df()
    
    moviesIndexList=list(movie_tag_df.index)
    actorsIndexList = list(actor_tag_df.index)
    actorsSize = len(actorsIndexList)
    
    actorU, actorSigma, actorV = decompositions.SVDDecomposition(actor_tag_df, 5)
    movieU, movieSigma, movieV = decompositions.SVDDecomposition(movie_tag_df, 5)

    
    movieSemanticsToActorSemanticsMapping = np.matrix(movieV) * np.matrix(actorV.transpose())
    movieInMovieSemantics = np.matrix(movieU[moviesIndexList.index(movie_id)])
    movieInActorSemanticsMatrix = movieInMovieSemantics * movieSemanticsToActorSemanticsMapping
    movieInActorSemantics = (movieInActorSemanticsMatrix.tolist())[0]

    actorsInSemantics = np.matrix(actorU)
    
    actorsWithScores = []
    
    DataHandler.createDictionaries1()
    DataHandler.create_actor_actorid_map()
    actorsForMovie = DataHandler.movie_actor_map.get(movie_id)    
    
    for index in range(0, actorsSize):
        actor_id = actorsIndexList[index]
        if actor_id in actorsForMovie:
            continue
        actorMatrix = actorsInSemantics[index]
        actor = (actorMatrix.tolist())[0]
        actorName = DataHandler.actor_actorid_map.get(actor_id)
        actorsWithScores.append((metrics.cosineSim(actor, movieInActorSemantics), actorName))
    resultActors = sorted(actorsWithScores, key=operator.itemgetter(0), reverse=True)
    print("10 Actors similar to movie " + str(movie_id) + " are: ")
    print(resultActors[0:10])
    return

'''
prints the top 10 actos related to the given actor
Result is printed as a list of score, actor name pairs
Underlying decompositon to extract semantics is PCA
'''
def task1c_pca(actor_id):
    DataHandler.vectors()
    actorTagDataframe = DataHandler.actor_tag_df()
    actorTagMatrix = np.matrix(DataHandler.actor_tag_df().as_matrix())
    
    actorIndexList = list(actorTagDataframe.index)
    
    components = decompositions.PCADecomposition(actorTagDataframe, 5)
    
    #using transpose since according to page 158, p inverse = p transpose
    pMatrix = np.matrix(components).transpose()
    actorsInSemantics = (actorTagMatrix * pMatrix).tolist()
    
    simAndActor = [] 
    concernedActorInSemantics = actorsInSemantics[actorIndexList.index(actor_id)] 
    DataHandler.create_actor_actorid_map()
    
    for index in range(0, len(actorsInSemantics)):
        comparisonActorId = actorIndexList[index]
        if (actor_id == comparisonActorId):
            continue
        comparisonActorSemantics = actorsInSemantics[index]
        comparisonActorName = DataHandler.actor_actorid_map.get(comparisonActorId)
        simAndActor.append((metrics.l2Norm(concernedActorInSemantics, comparisonActorSemantics), comparisonActorName))
    
    result = sorted(simAndActor, key=operator.itemgetter(0), reverse=False)
    
    print("10 Actors similar to " + str(DataHandler.actor_actorid_map.get(actor_id)) + " are: ")
    print(result[0:10])
    return


def PPR_top10_SimilarActors(seed):
    DataHandler.createDictionaries1()
    DataHandler.create_actor_actorid_map()
    actact = DataHandler.actor_actor_similarity_matrix()
    actor_actorid_map = DataHandler.actor_actorid_map
    alpha = constants.ALPHA
    act_similarities = pagerank.PPR(actact,seed,alpha)
    print('Actors similar to the following seed actors: '+str([actor_actorid_map.get(i) for i in seed]))
    for index,sim in act_similarities:
        print(actor_actorid_map.get(actact.columns[index])+' '+ str(sim))
        
def PPR_top10_SimilarCoActors(seed):
    DataHandler.createDictionaries1()
    DataHandler.create_actor_actorid_map()
    actact = DataHandler.actor_actor_similarity_matrix()
    actor_actorid_map = DataHandler.actor_actorid_map
    alpha = constants.ALPHA
    act_similarities = pagerank.PPR(actact,seed,alpha)
    print('Co Actors similar to the following seed actors: '+str([actor_actorid_map.get(i) for i in seed]))
    for index,sim in act_similarities.items():
        print(actor_actorid_map.get(actact.columns[index])+' '+ str(sim))

#userMovies = user_rated_or_tagged_map.get(67348)
def top5SimilarMovies(userMovies):
    DataHandler.createDictionaries1()
    u = decompositions.CPDecomposition(DataHandler.getTensor_ActorMovieGenreYear(),5)
    movies = sorted(list(DataHandler.movie_actor_map.keys()))
    u1= u[1]
    movieNewDSpace = pd.DataFrame(u1,index = movies)
    movie_movie_similarity = DataHandler.movie_movie_Similarity(movieNewDSpace)
    movieid_name_map = DataHandler.movieid_name_map
    alpha = constants.ALPHA
    movie_similarities = pagerank.PPR(movie_movie_similarity,userMovies,alpha)
    print('Movies similar to the following seed movies: '+str([movieid_name_map.get(i) for i in userMovies]))
    for index,sim in movie_similarities:
        if (movie_movie_similarity.columns[index] not in userMovies):
            print(movieid_name_map.get(movie_movie_similarity.columns[index])+' '+ str(sim))
            
