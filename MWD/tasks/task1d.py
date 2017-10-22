# -*- coding: utf-8 -*-

from computations import decompositions
from computations import metrics
from data import DataHandler
import numpy as np
import operator

def task1d(movie_id, method):
    if(method=="SVD"):
        task1dImplementation_SVD(movie_id)
    elif(method=="PCA"):
        task1d_pca(movie_id)
    elif(method=="LDA"):
        similarMovieActor_LDA(movie_id)
    elif(method=="TFIDF"):
        task1d_tfidf(movie_id)
    else:
        print("Invalid method. Please use SVD or PCA or LDA or TFIDF") 
    return

def task1dImplementation_SVD(movie_id):
    DataHandler.vectors()
    DataHandler.createDictionaries1()
    
    actor_tag_df = DataHandler.actor_tag_df()
    movie_tag_df = DataHandler.load_movie_tag_df()
    
    moviesIndexList=list(movie_tag_df.index)
    actorsIndexList = list(actor_tag_df.index)
    actorsSize = len(actorsIndexList)
    
    actorU, actorSigma, actorV = decompositions.SVDDecomposition(actor_tag_df, 5)

    tagsToActorSemantics = (np.matrix(actorV)).transpose()
    movieTagMatrix= np.matrix(movie_tag_df.as_matrix())
    movieInTags = movieTagMatrix[moviesIndexList.index(movie_id)]
    movieInActorSemantics = (movieInTags * tagsToActorSemantics).tolist()[0]
    actorsInSemantics = np.matrix(actorU)
    
    actorsWithScores = []
    
    DataHandler.create_actor_actorid_map()
    actorsForMovie = DataHandler.movie_actor_map.get(movie_id)    
    
    for index in range(0, actorsSize):
        actor_id = actorsIndexList[index]
        if actor_id in actorsForMovie:
            continue
        actorMatrix = actorsInSemantics[index]
        actor = (actorMatrix.tolist())[0]
        actorName = DataHandler.actor_actorid_map.get(actor_id)
        similarityScore = metrics.l2Norm(actor, movieInActorSemantics)
        actorsWithScores.append((similarityScore, actorName))
    
    resultActors = sorted(actorsWithScores, key=operator.itemgetter(0), reverse=False)
    top10Actors = resultActors[0:10]
    movieid_name_map = DataHandler.movieid_name_map
    print("10 Actors similar to movie " + str(movieid_name_map.get(movie_id)) + " are: ")
    for tup in top10Actors:
        print(tup[1] + " : " + str(tup[0]))
    return

def task1d_pca(movie_id):
    DataHandler.vectors()
    DataHandler.createDictionaries1()
    
    actor_tag_df = DataHandler.actor_tag_df()
    movie_tag_df = DataHandler.load_movie_tag_df()
    
    actorTagMatrix = np.matrix(actor_tag_df.as_matrix())
    movieTagMatrix= np.matrix(movie_tag_df.as_matrix())
    
    actorIndexList = list(actor_tag_df.index)
    movieIndexList = list(movie_tag_df.index)
    
    actorSemantics = decompositions.PCADecomposition(actor_tag_df, 5)
    
    actorP = np.matrix(actorSemantics).transpose()
    movieInTags = movieTagMatrix[movieIndexList.index(movie_id)]
    movieInActorSemantics = (movieInTags * actorP).tolist()[0]
    actorsInActorSemantics = (actorTagMatrix * actorP).tolist()
    
    DataHandler.create_actor_actorid_map()
    actorsForMovie = DataHandler.movie_actor_map.get(movie_id)
    
    DataHandler.create_actor_actorid_map()
    actorsSize = len(actorsInActorSemantics)
    simAndActor = []
    for index in range(0, actorsSize):
        actorId = actorIndexList[index]
        if (actorId in actorsForMovie):
            continue
        actorInSemantics = actorsInActorSemantics[index]
        actorName = DataHandler.actor_actorid_map.get(actorId)
        score = metrics.l2Norm(actorInSemantics, movieInActorSemantics)
        simAndActor.append((score, actorName))
    
    result = sorted(simAndActor, key=operator.itemgetter(0), reverse=False)
    
    movieid_name_map = DataHandler.movieid_name_map
    print("Top 10 actors similar to movie: " + str(movieid_name_map.get(movie_id)) + " are: ")
    top10Actors = result[0:10]
    for tup in top10Actors:
        print(tup[1] + " : " + str(tup[0]))
    return

def task1d_tfidf(movie_id):
    DataHandler.vectors()
    DataHandler.createDictionaries1()
    actorTagDataframe = DataHandler.actor_tag_df()
    movie_tag_df = DataHandler.load_movie_tag_df()
    
    actorsTags = np.matrix(actorTagDataframe.as_matrix()).tolist()
    actorIndexList = list(actorTagDataframe.index)
    movieIndexList = list(movie_tag_df.index)
    movieTagMatrix= np.matrix(movie_tag_df.as_matrix())
    
    
    actorsForMovie = DataHandler.movie_actor_map.get(movie_id)
    simAndActor = []
    movieInTags = movieTagMatrix[movieIndexList.index(movie_id)].tolist()[0]
    totalActors = len(actorIndexList)
    DataHandler.create_actor_actorid_map()
    
    for index in range(0, totalActors):
        actorId = actorIndexList[index]
        if (actorId in actorsForMovie):
            continue
        actorName = DataHandler.actor_actorid_map.get(actorId)
        actorinTags = actorsTags[index]
        comparisonScore = metrics.l2Norm(movieInTags, actorinTags)
        simAndActor.append((comparisonScore, actorName))
        
    result = sorted(simAndActor, key=operator.itemgetter(0), reverse=False)
    
    top10Actors = result[0:10]
    movieid_name_map = DataHandler.movieid_name_map
    print("Top 10 actors similar to " + str(movieid_name_map.get(movie_id)) + " are: ")
    for tup in top10Actors:
        print(tup[1] + " : " + str(tup[0]))
    return
	
def similarMovieActor_LDA(givenMovie):
    vectors()
    createDictionaries1()
    givenActor_similarity = defaultdict(float)
    actor_tag_dff = actor_tag_df()
    movie_tag_dff = load_movie_tag_df()
    actorTagMatrix = np.matrix(actor_tag_dff.as_matrix())
    movieTagMatrix= np.matrix(movie_tag_dff.as_matrix())
    
    actorIndexList = list(actor_tag_dff.index)
    movieIndexList = list(movie_tag_dff.index)
    movieInTags = movieTagMatrix[movieIndexList.index(givenMovie)]
    actorsForMovie = movie_actor_map.get(givenMovie)
    
    ldaModel,doc_term_matrix,id_Term_map  =  decompositions.LDADecomposition(actor_tag_dff,5,constants.actorTagsSpacePasses)
    for otherActor in actorIndexList:
        mo1 = representDocInLDATopics(movie_tag_dff,givenMovie,ldaModel)
        if otherActor not in actorsForMovie:
            ac2 = representDocInLDATopics(actor_tag_dff,otherActor,ldaModel)
            givenActor_similarity[otherActor]=(metrics.simlarity_kullback_leibler(mo1,ac2))
    #print(sorted(givenActor_similarity.items(),key = itemgetter(1),reverse=True))
    top10 = sorted(givenActor_similarity.items(),key = itemgetter(1),reverse=False)[0:11]
    for actors in top10:
        print(actor_actorid_map.get(actors[0]), actors[1])
    return top10
	
