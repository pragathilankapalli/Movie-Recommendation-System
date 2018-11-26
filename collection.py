import pandas as pd
import numpy as np

class Collection:
    def __init__(self, ratings, collections, movies):
        self.ratings = ratings
        self.collections = collections
        self.movies = movies
        self.recommendations = {}
        self.usersList = list(set(ratings['userId'].values))
        for userId in self.usersList:
            self.recommendations[str(userId)] = self.collection(userId)

    def collection(self, userId):
        movies_watched = self.ratings[self.ratings['userId']==userId]['movieId'].values
        collections_watched = self.movies[self.movies['moveId'].isin(movies_watched)]['collectionId'].values
        collections_watched = collections_watched[~np.isnan(collections_watched)].astype(int)
        unique_elements, counts_elements = np.unique(collections_watched, return_counts=True)
        collections_watched = pd.DataFrame(counts_elements, index=unique_elements, columns= ['count'])
        collections_watched = collections_watched.sort_values(by=['count'], ascending= False)
        collections_watched = collections_watched[:max( min(20, len(collections_watched)), int(len(collections_watched)*.20))]

        movies_ratings = self.ratings[self.ratings['userId']==userId]
        collections_ratings = []
        for collectionId in collections_watched.index.values:
            collection_movies = self.movies[self.movies['collectionId'] == collectionId]['moveId'].values
            collection_ratings = movies_ratings[movies_ratings['movieId'].isin(collection_movies)]['rating'].values
            collections_ratings.append(np.mean(collection_ratings))

        collections_ratings = pd.DataFrame(collections_ratings,index=collections_watched.index , columns=['rating'])
        collections_watched = collections_watched.join(collections_ratings)
        collections_watched = collections_watched.sort_values(by=['count', 'rating'], ascending= False)

        return collections_watched.index.values[:min(10, len(collections_watched))]
