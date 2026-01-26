import { Movie, User, Rating } from '../types';

export const parseMovies = (content: string): Movie[] => {
  const lines = content.split('\n');
  const movies: Movie[] = [];
  
  for (const line of lines) {
    if (!line.trim()) continue;
    const parts = line.split('::');
    if (parts.length >= 3) {
      movies.push({
        id: parseInt(parts[0], 10),
        title: parts[1],
        genres: parts[2].split('|')
      });
    }
  }
  return movies;
};

export const parseUsers = (content: string): User[] => {
  const lines = content.split('\n');
  const users: User[] = [];

  for (const line of lines) {
    if (!line.trim()) continue;
    const parts = line.split('::');
    if (parts.length >= 5) {
      users.push({
        id: parseInt(parts[0], 10),
        gender: parts[1],
        age: parseInt(parts[2], 10),
        occupation: parseInt(parts[3], 10),
        zipCode: parts[4]
      });
    }
  }
  return users;
};

export const parseRatings = (content: string): Rating[] => {
  const lines = content.split('\n');
  const ratings: Rating[] = [];
  
  // Optimization: If dataset is too large, we might want to sample. 
  // For standard 1M, browser JS can handle it but it takes a few seconds.
  for (const line of lines) {
    if (!line.trim()) continue;
    const parts = line.split('::');
    if (parts.length >= 4) {
      ratings.push({
        userId: parseInt(parts[0], 10),
        movieId: parseInt(parts[1], 10),
        rating: parseInt(parts[2], 10),
        timestamp: parseInt(parts[3], 10)
      });
    }
  }
  return ratings;
};