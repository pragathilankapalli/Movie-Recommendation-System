import pandas as pd
import numpy as np

class Content:
    def __init__(self, ratings, movie_genre, movie_language):
        self.ratings = ratings
        self.movie_genre = movie_genre
        self.movie_language = movie_language
        self.recommendations = {}
        self.usersList = list(set(ratings['userId'].values))
        self.error = []
        for userId in self.usersList:
            self.error.append(self.calculate_error(userId))


    def content(self, userId):
        movies_watched = self.ratings[self.ratings['userId']==userId]['movieId'].values
        movies_ratings = self.ratings[self.ratings['userId']==userId]['rating'].values

        movies = np.array([movies_watched, movies_ratings]).T

        np.random.shuffle(movies)
        breakpoint = int(len(movies)*.8)
        watched_training = movies[:breakpoint, 0]
        ratings_training = movies[:breakpoint, 1]
        watched_testing = movies[breakpoint:, 0]
        ratings_testing = movies[breakpoint:, 1]

        user_ratings = self.movie_genre[watched_training]*ratings_training
        user_ratings = user_ratings.dropna(how='all')
        genre_ratings = user_ratings.apply(lambda x: pd.Series([len(x.dropna().values), x.dropna().values.mean()], index = ['length', 'mean']), axis = 1)

        user_ratings = self.movie_language[watched_training]*ratings_training
        user_ratings = user_ratings.dropna(how='all')
        language_ratings = user_ratings.apply(lambda x: pd.Series([len(x.dropna().values), x.dropna().values.mean()], index = ['length', 'mean']), axis = 1)

        indices = []
        for index, row in genre_ratings.iterrows():
            if row['length'] < max(genre_ratings['length'])*.25:
                indices.append(index)

        genre_ratings = genre_ratings.drop(indices)
        indices = genre_ratings.index
        genre_ratings_mean = pd.DataFrame(genre_ratings['mean'])
        rated_genre_movies = self.movie_genre.loc[indices]
        rated_genre_movies = pd.DataFrame(rated_genre_movies.values*genre_ratings_mean.values, columns=rated_genre_movies.columns, index=rated_genre_movies.index)

        indices = []
        for index, row in language_ratings.iterrows():
            if row['length'] < max(language_ratings['length'])*.25:
                indices.append(index)

        language_ratings = language_ratings.drop(indices)
        indices = language_ratings.index
        language_ratings_mean = pd.DataFrame(language_ratings['mean'])
        rated_language_movies = self.movie_language.loc[indices]
        rated_language_movies = pd.DataFrame(rated_language_movies.values*language_ratings_mean.values, columns=rated_language_movies.columns, index=rated_language_movies.index)

        genre_rated_movies = pd.DataFrame(np.array([rated_genre_movies.mean().values, rated_language_movies.mean().values]).T, columns=['genre', 'language'], index=rated_genre_movies.mean().index).sort_values(by = 'genre', ascending=False)
        recoms = genre_rated_movies.dropna().index
        recoms = np.append(np.array(recoms), np.array(movies_watched))
        indexes = np.unique(recoms, return_index=True)[1]
        recoms = [recoms[index] for index in sorted(indexes)]
        self.recommendations[str(userId)] = recoms[:10]

        return genre_rated_movies, watched_testing, ratings_testing

    def calculate_error(self,userId):
        recs, watched_testing, ratings_testing = self.content(userId)
        er = np.array(ratings_testing-recs.loc[watched_testing]['genre'].values)
        er = er[~np.isnan(er)]
        return np.sqrt(np.mean(er**2))
