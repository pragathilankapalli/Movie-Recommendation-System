import pandas as pd
import numpy as np

class Content:
    def __init__(self, ratings, movie_genre, top250_movies, links, movies):
        self.movies = movies
        self.ratings = ratings
        self.movie_genre = movie_genre
        self.recommendations = {}
        usersList = list(set(ratings['userId'].values))
        top250_movies_ratings = top250_movies['imdb_rating'].values
        top250_movies_ratings = self.normalize(top250_movies_ratings, 'standard')
        top250_movies['imdb_rating'] = top250_movies_ratings
        self.new_links = links[links['imdbId'].isin(top250_movies.index)]
        self.new_links = self.new_links.join(top250_movies, on='imdbId')
        for userId in usersList:
            print(userId)
            self.recommendations[str(userId)] = self.top250(userId)

    def normalize(self,data, method):
        if method == 'min-max':
            return (data-min(data))/(max(data) - min(data))
        if method == 'standard':
            if round(np.std(data)) == 0:
                return data/max(data)
            else:
                return (data-np.mean(data))/np.std(data)

    def top250(self, userId):
        movies_watched = self.ratings[self.ratings['userId']==userId]['movieId'].values
        movies_ratings = self.ratings[self.ratings['userId']==userId]['rating'].values

        watched = self.ratings[self.ratings['userId']==userId]
        watched = watched.drop(['userId', 'timestamp'], axis=1)
        movies_ratings = self.ratings[self.ratings['userId']==userId]['rating'].values
        movies_ratings_norm = self.normalize(movies_ratings, 'standard')
        watched['rating'] = movies_ratings_norm
        watched_links = self.new_links[self.new_links['movieId'].isin(watched['movieId'].values)]
        watched_links = watched_links.set_index('movieId')
        watched_links = watched.join(watched_links, on='movieId')
        watched_links = watched_links.dropna()
        watched_links['weighted_rating'] = watched_links['rating']/watched_links['imdb_rating']
        watched_links = watched_links.replace([np.inf, -np.inf], np.nan).dropna()
        weight = watched_links['weighted_rating'].values
        if len(weight) > 0:
            weight = round(np.mean(self.normalize(weight,'min-max')), 2)
        else:
            weight = 0.1
        not_watched_links = self.new_links[~self.new_links['movieId'].isin(watched['movieId'].values)]

        self.movie_genre = pd.DataFrame()
        for i in range(0,len(self.movies)):
            movie =  self.movies.loc[i]
            movie_id = movie['movieId']
            genres = movie['genres'].split('|')
            df = pd.DataFrame({movie_id:np.ones(len(genres))}, index = genres)
            self.movie_genre = pd.concat([self.movie_genre, df], axis=1, sort=False)

        user_ratings = self.movie_genre[movies_watched]*movies_ratings
        user_ratings = user_ratings.dropna(how='all')
        genre_ratings = user_ratings.apply(lambda x: pd.Series([len(x.dropna().values), x.dropna().values.mean()], index = ['length', 'mean']), axis = 1)

        indices = []
        for index, row in genre_ratings.iterrows():
            if row['length'] < max(genre_ratings['length'])*.25:
                indices.append(index)

        genre_ratings = genre_ratings.drop(indices)
        indices = genre_ratings.index
        genre_ratings_mean = pd.DataFrame(genre_ratings['mean'])
        rated_genre_movies = self.movie_genre.loc[indices]
        rated_genre_movies = pd.DataFrame(rated_genre_movies.values*genre_ratings_mean.values, columns=rated_genre_movies.columns, index=rated_genre_movies.index)

        genre_rated_movies = pd.DataFrame(np.array([rated_genre_movies.mean().values]).T, columns=['genre'], index=rated_genre_movies.mean().index).sort_values(by = 'genre', ascending=False)
        not_watched_genre = genre_rated_movies.loc[not_watched_links['movieId'].values]
        not_watched_imdb = self.new_links[~self.new_links['movieId'].isin(watched['movieId'].values)]
        ratings_df = not_watched_imdb.join(not_watched_genre, on='movieId')
        ratings_df['imdb_rating'] = self.normalize(ratings_df['imdb_rating'].values, 'min-max')
        ratings_df['genre'] = self.normalize(ratings_df['genre'].values, 'min-max')
        ratings_df['final_ratings'] = ratings_df['imdb_rating']*weight + ratings_df['genre']*(1-weight)
        recoms = ratings_df.sort_values(by='final_ratings', ascending= 0)['movieId']
        return list(recoms[:10])
