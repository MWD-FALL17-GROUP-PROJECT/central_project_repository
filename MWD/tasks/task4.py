from computations import tasksBusiness

def task4(user_id):
    movies = tasksBusiness.Recommender(user_id)
    print([k for (k,y) in movies[0:5]])
