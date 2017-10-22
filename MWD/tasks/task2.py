# -*- coding: utf-8 -*-

from computations import decompositions, tasksBusiness
from data import DataHandler
from util import formatter
import numpy as np

def prettyPrintActorVector(vector, actorsInDf, actorIdActorsDf):
    vectorLen = len(vector)
    for index in range(0, vectorLen):
        actorId = actorsInDf[index]
        actorName = actorIdActorsDf[actorIdActorsDf['id']==actorId].iloc[0][1]
        print(actorName + ": " + str(vector[index]), end=', ')
    print('.')

def task2a():
    DataHandler.vectors()
    DataHandler.createDictionaries1()
    
    actor_actor_sim_df, actorList = DataHandler.actor_actor_similarity_matrix()
    u, sigma, vt = decompositions.SVDDecomposition(actor_actor_sim_df, 3)
    semantics = np.matrix(vt).tolist()
    
    actorIdActorsDf = DataHandler.actor_info_df
    actorsInDf = list(actor_actor_sim_df.index)
    print("Top 3 semantics are:")
    for semantic in semantics:
        prettyPrintActorVector(semantic, actorsInDf, actorIdActorsDf)
        print("")
    
    split_group_with_index = formatter.splitGroup(u,3)
    
    print("The three clusters are:")
    print(tasksBusiness.get_partition_on_ids(split_group_with_index, actorList))

def task2b():
    DataHandler.vectors()
    DataHandler.createDictionaries1()
    
    coactor_similarity_df, actorList = DataHandler.coactor_siilarity_matrix()
    u, sigma, vt = decompositions.SVDDecomposition(coactor_similarity_df, 3)
    semantics = np.matrix(vt).tolist()
    
    actorIdActorsDf = DataHandler.actor_info_df
    actorsInDf = list(coactor_similarity_df.index)
    print("Top 3 semantics are:")
    
    for semantic in semantics:
        prettyPrintActorVector(semantic, actorsInDf, actorIdActorsDf)
        print("")
    
    split_group_with_index = formatter.splitGroup(u,3)
    
    print("The three clusters are:")
    print(tasksBusiness.get_partition_on_ids(split_group_with_index, actorList))
    
def task2c(space):
    tasksBusiness.top5LatentCP("AMY", space)
    tasksBusiness.get_partition_subtasks()
    
def task2d(space):
    tasksBusiness.top5LatentCP("TMR", space)
    tasksBusiness.get_partition_subtasks()