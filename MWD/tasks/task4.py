from computations import tasksBusiness
from data import DataHandler
def task4(user_id):
    DataHandler.vectors()
    if (user_id in DataHandler.user_rated_or_tagged_map):
        movies = tasksBusiness.Recommender(user_id)
        print([k for (k,y) in movies[0:5]])
    else:
        print('Invalid User ID : ' + str(user_id))
