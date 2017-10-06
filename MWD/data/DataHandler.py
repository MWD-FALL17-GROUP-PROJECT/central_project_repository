from collections import defaultdict
import pandas as pd
import dateutil
import sys
import time
import math
import csv
from computations import metrics
from computations import decompositions
from util import constants
from util import formatter


max_rank = 0
min_rank = sys.maxsize

max_date = 0
min_date = sys.maxsize
tag_count = 0

tagset_genre = defaultdict(set)
actor_weight_vector_tf = dict()
actor_weight_vector_tf_idf = dict()
genre_weight_vector_tf = dict()
genre_weight_vector_tf_idf = dict()
user_tag_map_tf = defaultdict()
user_tag_map_tf_idf = defaultdict()

movie_actor_df = pd.read_csv(constants.DIRECTORY + "movie-actor.csv")
tag_movie_df = pd.read_csv(constants.DIRECTORY + "mltags.csv")
genre_movie_df = pd.read_csv(constants.DIRECTORY + "mlmovies.csv")
tag_id_df = pd.read_csv(constants.DIRECTORY + "genome-tags.csv")
user_ratings_df = pd.read_csv(constants.DIRECTORY + "mlratings.csv")

actor_movie_rank_map = defaultdict(set)
movie_actor_rank_map = defaultdict(set)
movie_tag_map = defaultdict(set)
genre_movie_map = defaultdict(set)
user_tag_map = defaultdict(set)
tag_user_map = defaultdict(set)
genre_tagset = defaultdict(set)
tag_movie_map = defaultdict(list)
user_rated_or_tagged_map = defaultdict(set)
tag_id_map = dict()
id_tag_map = dict()


def vectors():
	global max_rank
	global min_rank
	global tag_count
	global max_date
	global min_date
	t = time.time()

	for row in tag_id_df.itertuples():
		tag_id_map[row.tagId] = row.tag
		id_tag_map[row.tag] = row.tagId

	for row in user_ratings_df.itertuples():
		user_rated_or_tagged_map[row.userid].add(row.movieid)

	tagset = set()

	for row in tag_movie_df.itertuples():
		date_time = dateutil.parser.parse(row.timestamp).timestamp()
		if date_time > max_date:
			max_date = date_time
		if date_time < min_date:
			min_date = date_time
		tagset.add(row.tagid)
		user_rated_or_tagged_map[row.userid].add(row.movieid)
		tag_movie_map[row.tagid].append((row.movieid, date_time))
		user_tag_map[row.userid].add((row.tagid, date_time))
		tag_user_map[row.tagid].add((row.userid, date_time))
		movie_tag_map[row.movieid].add((row.tagid, date_time))

	tag_count = tagset.__len__()
	tagset.clear()
	print('Main : ', time.time() - t)


# def load_genre_count_matrix(given_genre):
# 	for row in genre_movie_df.itertuples():
# 		genres_list = row.genres.split("|")
# 		for genre in genres_list:
# 			genre_movie_map[genre].add(row.movieid)
#
# 	tagList = sorted(list(tag_movie_map.keys()))
# 	for movie in genre_movie_map[given_genre]:
# 		tag_count_movie = dict([(tag, len(filter(lambda x: x == movie, tag_movie_map[tag]))) for tag in tagList])


def load_genre_matrix(given_genre):
	movieCount = movie_tag_map.keys().__len__()
	for row in genre_movie_df.itertuples():
		genres_list = row.genres.split("|")
		for genre in genres_list:
			genre_movie_map[genre].add(row.movieid)

	tagList = sorted(list(tag_movie_map.keys()))
	movieList = []
	df = pd.DataFrame(columns=tagList)
	for movie in genre_movie_map[given_genre]:
		tagsInMovie = movie_tag_map[movie]
		tf_idf_map = dict()
		if tagsInMovie:
			movieList.append(movie)
			for tag in tagList:
				moviesInTagCount = len(tag_movie_map[tag])
				tf_numerator = 0
				for temp_movie, datetime in tag_movie_map[tag]:
					if movie == temp_movie:
						tf_numerator += formatter.normalizer(min_date, max_date, datetime)
				tf = tf_numerator / len(tagsInMovie)
				tf_idf = tf * math.log2(movieCount / moviesInTagCount)
				tf_idf_map[tag] = tf_idf
			df = df.append(tf_idf_map, ignore_index=True)
	df.index = movieList
	return df


def load_genre_actor_matrix(given_genre):
	global max_rank
	global min_rank
	global tag_count
	global max_date
	global min_date
	for row in genre_movie_df.itertuples():
		genres_list = row.genres.split("|")
		for genre in genres_list:
			genre_movie_map[genre].add(row.movieid)

	for row in movie_actor_df.itertuples():
		if row.actor_movie_rank < min_rank:
			min_rank = row.actor_movie_rank
		if row.actor_movie_rank > max_rank:
			max_rank = row.actor_movie_rank
		actor_movie_rank_map[row.actorid].add((row.movieid, row.actor_movie_rank))
		movie_actor_rank_map[row.movieid].add((row.actorid, row.actor_movie_rank))

	actorList = sorted(list(actor_movie_rank_map.keys()))
	df = pd.DataFrame(columns=actorList)
	movieCount = movie_tag_map.keys().__len__()
	movieList = []

	for movieInGenre in genre_movie_map[given_genre]:
		movieList.append(movieInGenre)
		actorsInMovieList = movie_actor_rank_map[movieInGenre]
		actorCountOfMovie = len(actorsInMovieList)
		tf_idf_map = dict.fromkeys(actorList, 0.0)
		for actor, rank in actorsInMovieList:
			movieCountOfActor = len(actor_movie_rank_map[actor])
			tf_numerator = (1 / formatter.normalizer(min_rank, max_rank, rank))
			tf_idf = (tf_numerator / actorCountOfMovie) * math.log2(movieCount / movieCountOfActor)
			tf_idf_map[actor] = tf_idf
		df = df.append(tf_idf_map, ignore_index=True)
	df.index = movieList
	return df

def actor_tag_df():
	actor_weight_vector_tf_idf = actor_tagVector()
	tagList = sorted(list(tag_movie_map.keys()))
	actorList = sorted(list(actor_movie_rank_map.keys()))
	df = pd.DataFrame(columns=tagList)
	dictList = []

	for actor in actorList:
		actor_tag_dict = dict.fromkeys(tagList,0.0)
		for tag,weight in actor_weight_vector_tf_idf[actor]:
			actor_tag_dict[tag] = weight
		dictList.append(actor_tag_dict)
	df = df.append(dictList,ignore_index=True)

def actor_similarity_tagVector(actor_id_given):
	actor_weight_vector_tf_idf = actor_tagVector()
	actor_vector = actor_weight_vector_tf_idf[actor_id_given]
	print(list(map(lambda x:tag_id_map[x[0]], sorted(actor_vector,key=lambda x:x[1]))))

	actorsList = actor_movie_rank_map.keys()
	return sorted([(actor, metrics.euclidean(actor_vector, actor_weight_vector_tf_idf[actor])) for actor in actorsList],
				  key=lambda x: x[0])


def actor_similarity_matrix(actor_id_given):
	actor_weight_vector_tf_idf = actor_tagVector()
	tagList = sorted(list(tag_movie_map.keys()))
	actorList = sorted(list(actor_movie_rank_map.keys()))
	df = pd.DataFrame(columns=tagList)
	dictList = []

	for actor in actorList:
		actor_tag_dict = dict.fromkeys(tagList,0.0)
		for tag,weight in actor_weight_vector_tf_idf[actor]:
			actor_tag_dict[tag] = weight
		dictList.append(actor_tag_dict)
	df = df.append(dictList,ignore_index=True)

	df = pd.DataFrame(decompositions.PCADimensionReduction(df,5),index=actorList)
	actor_vector = df.loc[actor_id_given]
	return sorted([(actor, metrics.euclidean(actor_vector, df.loc[actor])) for actor in actorList],
				  key=lambda x: x[0])


def actor_tagVector():
	global max_rank
	global min_rank

	for row in movie_actor_df.itertuples():
		if row.actor_movie_rank < min_rank:
			min_rank = row.actor_movie_rank
		if row.actor_movie_rank > max_rank:
			max_rank = row.actor_movie_rank
		actor_movie_rank_map[row.actorid].add((row.movieid, row.actor_movie_rank))
		movie_actor_rank_map[row.movieid].add((row.actorid, row.actor_movie_rank))

	total_actor_count = len(actor_movie_rank_map)
	for actorID, movies_list in actor_movie_rank_map.items():

		tag_counter = 0
		tag_weight_tuple_tf = defaultdict(float)
		tag_weight_tuple_tf_idf = defaultdict(float)
		for movie in movies_list:
			tag_counter += len(movie_tag_map[movie[0]])

		for movieID, rank in movies_list:
			if movieID in movie_tag_map:
				for tag_id, timestamp in movie_tag_map[movieID]:
					actor_count = 0
					aSetOfTags = set()
					for mov in tag_movie_map[tag_id]:
						aSetOfTags.update([k for (k, v) in movie_actor_rank_map[mov[0]]])
					actor_count = aSetOfTags.__len__()
					tf = (formatter.normalizer(min_date, max_date, timestamp)
						  / formatter.normalizer(min_rank, max_rank, rank)) / tag_counter
					tag_weight_tuple_tf[tag_id] += tf
					tag_weight_tuple_tf_idf[tag_id] += tf * math.log2(
						total_actor_count / actor_count)
		actor_weight_vector_tf_idf[actorID] = [(k, v) for k, v in tag_weight_tuple_tf_idf.items()]

	return actor_weight_vector_tf_idf