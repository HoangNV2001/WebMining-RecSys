import tensorflow as tf
import numpy as np

class MovieGenreEmbedding(tf.keras.Model):
    def __init__(self, len_movies, len_genres, embedding_dim):
        super(MovieGenreEmbedding, self).__init__()
        
        # embedding
        self.m_embedding = tf.keras.layers.Embedding(name='movie_embedding', input_dim=len_movies, output_dim=embedding_dim)
        self.g_embedding = tf.keras.layers.Embedding(name='genre_embedding', input_dim=len_genres, output_dim=embedding_dim)
        # dot product
        self.m_g_merge = tf.keras.layers.Dot(name='movie_genre_dot', normalize=True, axes=1)
        # output
        self.m_g_fc = tf.keras.layers.Dense(1, activation='sigmoid')
        
    def call(self, x):
        # x is a list: [movie_id, genre_id]
        memb = self.m_embedding(x[0])
        gemb = self.g_embedding(x[1])
        m_g = self.m_g_merge([memb, gemb])
        return self.m_g_fc(m_g)

class UserMovieEmbedding(tf.keras.Model):
    def __init__(self, len_users, len_movies, embedding_dim):
        super(UserMovieEmbedding, self).__init__()
        # embedding
        self.u_embedding = tf.keras.layers.Embedding(name='user_embedding', input_dim=len_users, output_dim=embedding_dim)
        self.m_embedding = tf.keras.layers.Embedding(name='movie_embedding', input_dim=len_movies, output_dim=embedding_dim)
        # dot product
        self.m_u_merge = tf.keras.layers.Dot(name='movie_user_dot', normalize=False, axes=1)
        # output
        self.m_u_fc = tf.keras.layers.Dense(1, activation='sigmoid')
        
    def call(self, x):
        # x is a list: [user_id, movie_id]
        # Truy cập trực tiếp các phần tử trong list x
        uemb = self.u_embedding(x[0])
        memb = self.m_embedding(x[1])
        m_u = self.m_u_merge([memb, uemb])
        return self.m_u_fc(m_u)