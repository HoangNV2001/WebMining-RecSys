export interface Movie {
  id: number;
  title: string;
  genres: string[];
}

export interface User {
  id: number;
  gender: string;
  age: number; // Mapped from code
  occupation: number;
  zipCode: string;
}

export interface Rating {
  userId: number;
  movieId: number;
  rating: number;
  timestamp: number;
}

export interface Interaction {
  movieId: number;
  action: 'like' | 'dislike' | 'rate' | 'click';
  rating?: number;
  timestamp: number;
}

export interface Recommendation {
  movie: Movie;
  score: number;
  reason: string;
}

export type AppMode = 'DEMO' | 'API';

export type ApiConnectionStatus = 'IDLE' | 'CHECKING' | 'CONNECTED' | 'ERROR';

export enum View {
  IMPORT = 'IMPORT',
  TRENDING = 'TRENDING',
  USERS = 'USERS',
  SESSION = 'SESSION',
}

export interface DatasetStats {
  userCount: number;
  movieCount: number;
  ratingCount: number;
  minTimestamp: number;
  maxTimestamp: number;
}