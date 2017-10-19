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
from computations import personalizedpagerank as ppr
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
    actorInSemantics = U[indexList.index(actor_id)]
    DataHandler.create_actor_actorid_map()
    for index in range(0, len(U)):
        comparisonActorId = indexList[index]
        if (comparisonActorId == actor_id):
            continue
        actorName = DataHandler.actor_actorid_map.get(comparisonActorId)
        similarityScore = metrics.l2Norm(actorInSemantics, U[index])
        simAndActor.append((similarityScore, actorName))
    
    result = sorted(simAndActor, key=operator.itemgetter(0), reverse=False)
    print("Top 10 Actors similar to " + str(DataHandler.actor_actorid_map.get(actor_id)) + " are:")
    top10Actors = result[0:10]
    for tup in top10Actors:
        print(tup[1] + " : " + str(tup[0]))
    print(result[0:10])
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
    
    top10Actors = result[0:10]
    print("Top 10 actors similar to " + str(DataHandler.actor_actorid_map.get(actor_id)) + " are: ")
    for tup in top10Actors:
        print(tup[1] + " : " + str(tup[0]))
    return

def task1c_tfidf(actor_id):
    DataHandler.vectors()
    actorTagDataframe = DataHandler.actor_tag_df()
    actorsTags = np.matrix(actorTagDataframe.as_matrix()).tolist()
    actorIndexList = list(actorTagDataframe.index)
    
    simAndActor = []
    concernedActor = actorsTags[actorIndexList.index(actor_id)]
    totalActors = len(actorIndexList)
    DataHandler.create_actor_actorid_map()
    
    for index in range(0, totalActors):
        comparisonActorId = actorIndexList[index]
        if(actor_id == comparisonActorId):
            continue
        comparisonActorName = DataHandler.actor_actorid_map.get(comparisonActorId)
        comparisonActor = actorsTags[index]
        comparisonScore = metrics.l2Norm(concernedActor, comparisonActor)
        simAndActor.append((comparisonScore, comparisonActorName))
        
    result = sorted(simAndActor, key=operator.itemgetter(0), reverse=False)
    
    top10Actors = result[0:10]
    print("Top 10 actors similar to " + str(DataHandler.actor_actorid_map.get(actor_id)) + " are: ")
    for tup in top10Actors:
        print(tup[1] + " : " + str(tup[0]))
    return
        

def PPR_top10_SimilarActors(seed):
    DataHandler.createDictionaries1()
    DataHandler.create_actor_actorid_map()
    actact = DataHandler.actor_actor_similarity_matrix()
    actor_actorid_map = DataHandler.actor_actorid_map
    alpha = constants.ALPHA
    act_similarities = pagerank.PPR(actact,seed,alpha)
    print('Top 10 actors similar to the following seed actors: '+str([actor_actorid_map.get(i) for i in seed]))
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
    
    simAndActor = []
    movieInTags = movieTagMatrix[movieIndexList.index(movie_id)].tolist()[0]
    totalActors = len(actorIndexList)
    DataHandler.create_actor_actorid_map()
    
    for index in range(0, totalActors):
        actorId = actorIndexList[index]
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
	
def PersnalizedPageRank_top10_SimilarActors(seed):
    DataHandler.createDictionaries1()
    DataHandler.create_actor_actorid_map()
    actact = DataHandler.actor_actor_similarity_matrix()
    actor_actorid_map = DataHandler.actor_actorid_map
    alpha = constants.ALPHA
    act_similarities = ppr.personalizedPageRank(actact,seed,alpha)
    actors = list(actact.index)
    actorDF = pd.DataFrame(pd.Series(actors),columns = ['Actor'])
    actorDF['Actor'] = actorDF['Actor'].map(lambda x:actor_actorid_map.get(x))
    Result = pd.concat([act_similarities,actorDF],axis = 1)
    sortedResult=Result.sort_values(by=0,ascending=False).head(15)
    seedAcotorNames = [actor_actorid_map.get(i) for i in seed]
    print('Actors similar to the following seed actors: '+str(seedAcotorNames))
    for index in sortedResult.index:
        if sortedResult.loc[index,'Actor'] not in seedAcotorNames:
            print(sortedResult.loc[index,'Actor']+' '+ str(sortedResult.loc[index,0]))
        
def PersnalizedPageRank_top10_SimilarCoActors(seed):
    DataHandler.createDictionaries1()
    DataHandler.create_actor_actorid_map()
    coactcoact = DataHandler.coactor_siilarity_matrix()
    actor_actorid_map = DataHandler.actor_actorid_map
    alpha = constants.ALPHA
    act_similarities = ppr.PPpersonalizedPageRankR(coactcoact,seed,alpha)
    actors = list(coactcoact.index)
    actorDF = pd.DataFrame(pd.Series(actors),columns = ['Actor'])
    actorDF['Actor'] = actorDF['Actor'].map(lambda x:actor_actorid_map.get(x))
    Result = pd.concat([act_similarities,actorDF],axis = 1)
    sortedResult=Result.sort_values(by=0,ascending=False).head(15)
    seedAcotorNames = [actor_actorid_map.get(i) for i in seed]
    print('Co Actors similar to the following seed actors: '+str(seedAcotorNames))
    for index in sortedResult.index:
        if sortedResult.loc[index,'Actor'] not in seedAcotorNames:
            print(sortedResult.loc[index,'Actor']+' '+ str(sortedResult.loc[index,0]))

#userMovies = user_rated_or_tagged_map.get(67348)
#userMovies = user_rated_or_tagged_map.get(3)
def PersnalizedPageRank_top5SimilarMovies(userMovies):
    DataHandler.createDictionaries1()
    u = decompositions.CPDecomposition(DataHandler.getTensor_ActorMovieGenreYearRankRating(),5)
    movies = sorted(list(DataHandler.movie_actor_map.keys()))
    u1= u[1]
    movieNewDSpace = pd.DataFrame(u1,index = movies)
    movie_movie_similarity = DataHandler.movie_movie_Similarity(movieNewDSpace)
    movieid_name_map = DataHandler.movieid_name_map
    alpha = constants.ALPHA
    movie_similarities = ppr.personalizedPageRank(movie_movie_similarity,userMovies,alpha)
    movies = list(movie_movie_similarity.index)
    movieDF = pd.DataFrame(pd.Series(movies),columns = ['movies'])
    movieDF['movies'] = movieDF['movies'].map(lambda x:movieid_name_map.get(x))
    Result = pd.concat([movie_similarities,movieDF],axis = 1)
    sortedResult=Result.sort_values(by=0,ascending=False).head(15)
    seedmovieNames = [movieid_name_map.get(i) for i in userMovies]
    print('Movies similar to the following seed movies: '+str(seedmovieNames))
    movie_genre_map = DataHandler.movie_genre_map
    genreForSeedMovies = [movie_genre_map.get(i) for i in userMovies]    
    print('Genres for seed movies: '+str(genreForSeedMovies))
    for index in sortedResult.index:
        if sortedResult.loc[index,'movies'] not in seedmovieNames:
            print(sortedResult.loc[index,'movies']+' '+ str(sortedResult.loc[index,0])+' '+str(movie_genre_map.get(movies[index])))


def top5SimilarMovies1(userMovies):
    DataHandler.createDictionaries1()
    u = decompositions.CPDecomposition(DataHandler.getTensor_ActorMovieGenreYearRankRating(),5)
    movies = sorted(list(DataHandler.movie_actor_map.keys()))
    u1= u[1]
    movieNewDSpace = pd.DataFrame(u1,index = movies)
    movie_movie_similarity = DataHandler.movie_movie_Similarity1(movieNewDSpace)
    movieid_name_map = DataHandler.movieid_name_map
    alpha = constants.ALPHA
    movie_similarities = pagerank.PPR(movie_movie_similarity,userMovies,alpha)
    print('Movies similar to the following seed movies: '+str([movieid_name_map.get(i) for i in userMovies]))
    for index,sim in movie_similarities:
        if (movie_movie_similarity.columns[index] not in userMovies):
            print(movieid_name_map.get(movie_movie_similarity.columns[index])+' '+ str(sim))

            
def PersnalizedPageRank_top5SimilarMovies1(userMovies):
    DataHandler.createDictionaries1()
    u = decompositions.CPDecomposition(DataHandler.getTensor_ActorMovieGenreYearRankRating(),5)
    movies = sorted(list(DataHandler.movie_actor_map.keys()))
    u1= u[1]
    movieNewDSpace = pd.DataFrame(u1,index = movies)
    movie_movie_similarity = DataHandler.movie_movie_Similarity1(movieNewDSpace)
    movieid_name_map = DataHandler.movieid_name_map
    alpha = constants.ALPHA
    movie_similarities = ppr.personalizedPageRank(movie_movie_similarity,userMovies,alpha)
    movies = list(movie_movie_similarity.index)
    movieDF = pd.DataFrame(pd.Series(movies),columns = ['movies'])
    movieDF['movies'] = movieDF['movies'].map(lambda x:movieid_name_map.get(x))
    Result = pd.concat([movie_similarities,movieDF],axis = 1)
    sortedResult=Result.sort_values(by=0,ascending=False).head(15)
    seedmovieNames = [movieid_name_map.get(i) for i in userMovies]
    print('Movies similar to the following seed movies: '+str(seedmovieNames))
    movie_genre_map = DataHandler.movie_genre_map
    genreForSeedMovies = [movie_genre_map.get(i) for i in userMovies]    
    print('Genres for seed movies: '+str(genreForSeedMovies))
    for index in sortedResult.index:
        if sortedResult.loc[index,'movies'] not in seedmovieNames:
            print(sortedResult.loc[index,'movies']+' '+ str(sortedResult.loc[index,0])+' '+str(movie_genre_map.get(movies[index])))

def task1a_svd(genre):
    DataHandler.vectors()
    DataHandler.createDictionaries1()
    
    genre_movie_map = DataHandler.getGenreMoviesMap()
    movie_tag_df = DataHandler.load_movie_tag_df()
    tagIdTagsDf = DataHandler.tag_id_df
    
    movies = genre_movie_map.get(genre)
    genre_movie_tags_df = (movie_tag_df.loc[movies]).dropna(how='any')
    U, Sigma, genre_semantics = decompositions.SVDDecomposition(genre_movie_tags_df, 5)
    
    print("The 5 semantics for genre:" + genre + " are")
    index = 1
    for semantic in np.matrix(genre_semantics).tolist():
        print("semantic " + str(index) + ": ")
        prettyPrintTagVector(semantic, tagIdTagsDf)
        print("")
        index = index + 1
    return

def task1a_pca(genre):
    DataHandler.vectors()
    DataHandler.createDictionaries1()
    
    genre_movie_map = DataHandler.getGenreMoviesMap()
    movie_tag_df = DataHandler.load_movie_tag_df()
    tagIdTagsDf = DataHandler.tag_id_df
    tagsInDf = list(movie_tag_df.transpose().index)
    
    movies = genre_movie_map.get(genre)
    genre_movie_tags_df = (movie_tag_df.loc[movies]).dropna(how='any')
    genre_semantics = decompositions.PCADecomposition(genre_movie_tags_df, 5)
    
    print("The 5 semantics for genre:" + genre + " are")
    index = 1
    for semantic in np.matrix(genre_semantics).tolist():
        print("semantic " + str(index) + ": ")
        prettyPrintTagVector(semantic, tagsInDf, tagIdTagsDf)
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

def prettyPrintTagVector(vector, tagsInDf, tagIdTagsDf):
    vectorLen = len(vector)
    for index in range(0,vectorLen):
        tagId = tagsInDf[index]
        tagName = tagIdTagsDf[tagIdTagsDf['tagId']==tagId].iloc[0][1]
        print(tagName + ':' + str(vector[index]), end=', ')
    print('.')
    
def prettyPrintActorVector(vector, actorsInDf, actorIdActorsDf):
    vectorLen = len(vector)
    for index in range(0, vectorLen):
        actorId = actorsInDf[index]
        actorName = actorIdActorsDf[actorIdActorsDf['id']==actorId].iloc[0][1]
        print(actorName + ": " + str(vector[index]), end=', ')
    print('.')