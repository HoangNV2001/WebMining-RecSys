import { Movie, Rating, Recommendation, User, Interaction } from '../types';

// Helper to shuffle array
const shuffle = <T,>(array: T[]): T[] => {
  let currentIndex = array.length,  randomIndex;
  while (currentIndex !== 0) {
    randomIndex = Math.floor(Math.random() * currentIndex);
    currentIndex--;
    [array[currentIndex], array[randomIndex]] = [
      array[randomIndex], array[currentIndex]];
  }
  return array;
};

export class RecommendationEngine {
  private movies: Map<number, Movie>;
  private ratings: Rating[];
  private userRatings: Map<number, Rating[]>;
  private maxTimestamp: number;

  constructor(movies: Movie[], ratings: Rating[]) {
    this.movies = new Map(movies.map(m => [m.id, m]));
    this.ratings = ratings;
    this.userRatings = new Map();
    this.maxTimestamp = 0;

    // Index ratings by user and find max timestamp
    ratings.forEach(r => {
      if (!this.userRatings.has(r.userId)) {
        this.userRatings.set(r.userId, []);
      }
      this.userRatings.get(r.userId)?.push(r);
      if (r.timestamp > this.maxTimestamp) {
        this.maxTimestamp = r.timestamp;
      }
    });
  }

  getTrending(limit: number = 10, days: number = 30, genreFilter: string | null = null): Recommendation[] {
    const secondsInDay = 86400;
    const cutoff = this.maxTimestamp - (days * secondsInDay);

    const recentRatings = this.ratings.filter(r => r.timestamp >= cutoff);
    const movieStats = new Map<number, { count: number; sum: number }>();

    recentRatings.forEach(r => {
      const stats = movieStats.get(r.movieId) || { count: 0, sum: 0 };
      stats.count++;
      stats.sum += r.rating;
      movieStats.set(r.movieId, stats);
    });

    const trending: Recommendation[] = [];
    movieStats.forEach((stats, movieId) => {
      const movie = this.movies.get(movieId);
      if (movie) {
        // Apply Genre Filter if exists
        if (genreFilter && !movie.genres.includes(genreFilter)) {
          return;
        }

        // Dampened mean: (Sum + global_avg*C) / (Count + C)
        // Simple Demo logic: Count * Avg
        const avg = stats.sum / stats.count;
        const score = stats.count * avg; 
        trending.push({
          movie,
          score,
          reason: `${stats.count} recent ratings`
        });
      }
    });

    return trending.sort((a, b) => b.score - a.score).slice(0, limit);
  }

  // Collaborative Filtering Simulation (Item-based / Content-based Hybrid approximation)
  getRecommendations(
    userId: number, 
    currentInteractions: Interaction[], 
    limit: number = 10
  ): Recommendation[] {
    const history = this.userRatings.get(userId) || [];
    
    // 1. Calculate User Genre Preferences
    const genreScores = new Map<string, number>();
    
    // Mix static history and dynamic session interactions
    const effectiveHistory = [
      ...history.map(h => ({ movieId: h.movieId, score: h.rating })),
      ...currentInteractions.map(i => ({ 
        movieId: i.movieId, 
        score: i.action === 'like' ? 5 : i.action === 'dislike' ? 1 : i.rating || 3 
      }))
    ];

    const seenMovieIds = new Set(effectiveHistory.map(h => h.movieId));

    if (effectiveHistory.length === 0) {
      // Cold start: Return trending with noise
      const trending = this.getTrending(50, 365);
      return shuffle(trending).slice(0, limit).map(r => ({...r, reason: 'Popular with everyone (Cold Start)'}));
    }

    effectiveHistory.forEach(h => {
      const movie = this.movies.get(h.movieId);
      if (!movie) return;
      const weight = (h.score - 3); // Center around 0 (-2 to +2)
      movie.genres.forEach(g => {
        genreScores.set(g, (genreScores.get(g) || 0) + weight);
      });
    });

    // 2. Score candidate movies
    const candidates: Recommendation[] = [];
    const moviesArr = Array.from(this.movies.values());
    
    // Optimization: Don't scan everything in browser every time, sample 2000 movies
    const sampleMovies = shuffle(moviesArr).slice(0, 2000);

    for (const movie of sampleMovies) {
      if (seenMovieIds.has(movie.id)) continue;

      let score = 0;
      let matchedGenres = 0;

      movie.genres.forEach(g => {
        const pref = genreScores.get(g);
        if (pref) {
          score += pref;
          matchedGenres++;
        }
      });

      // Add popularity bias
      score += (Math.random() * 2); // Random noise for exploration

      if (score > 0) {
        candidates.push({
          movie,
          score,
          reason: `Matches your taste in ${movie.genres[0]}`
        });
      }
    }

    // Sort and return
    return candidates.sort((a, b) => b.score - a.score).slice(0, limit);
  }
}