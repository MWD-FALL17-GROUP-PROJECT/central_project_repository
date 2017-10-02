from util import constants

def feature_combine(vec1, vec2):
	return set.intersection(set(vec1.keys()), set(vec2.keys()))

def euclidean(vec1,vec2):
	vec1 = dict(vec1)
	vec2 = dict(vec2)
	shared_features = feature_combine(vec1, vec2)
	return sum(list(map(lambda x: (vec2.get(x,0.0) + vec1.get(x,0.0)),shared_features)))

