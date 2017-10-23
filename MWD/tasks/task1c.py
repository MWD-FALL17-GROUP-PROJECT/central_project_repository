from computations import decompositions
from computations import metrics
from data import DataHandler
import numpy as np
import operator

def task1c(actor_id, method):
    if(method=="SVD"):
        actor_task1c_SVD(actor_id)
    elif(method=="PCA"):
        task1c_pca(actor_id)
    elif(method=="LDA"):
        top10_Actors_LDA_tf(actor_id)
    elif(method=="TFIDF"):
        task1c_tfidf(actor_id)
    else:
        print("Invalid method. Please use SVD or PCA or LDA or TFIDF") 
    return


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
    if (actor_id not in indexList):
        print("Invalid actor id or no tags present for actor. Returning")
        return
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
    if (actor_id not in actorIndexList):
        print("Invalid actor id or no tags present for actor. Returning")
        return
    
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

    if (actor_id not in actorIndexList):
        print("Invalid actor id or no tags present for actor. Returning")
        return
    
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

def top10_Actors_LDA_tf(givenActor):
    DataHandler.createDictionaries1()
    actor_movie_rank_map = DataHandler.actor_movie_rank_map
    if givenActor not in actor_movie_rank_map:
        print('Invalid seed actor id : '+str(givenActor))
        return
    DataHandler.create_actor_actorid_map()
    top10SimilarActors_similarity = DataHandler.similarActors_LDA_tf(givenActor)
    print('Actors similar to '+str(DataHandler.actor_actorid_map[givenActor]))
    for actor,sim in top10SimilarActors_similarity:
        print(DataHandler.actor_actorid_map[actor]+' '+str(sim))
    return