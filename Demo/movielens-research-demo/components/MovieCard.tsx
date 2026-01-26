import React, { useState } from 'react';
import { Movie, Interaction } from '../types';
import { ThumbsUp, ThumbsDown, Star, Play, Info } from 'lucide-react';
import { useApp } from '../AppContext';

interface MovieCardProps {
  movie: Movie;
  rank?: number;
  score?: number;
  reason?: string;
  readonly?: boolean;
}

const MovieCard: React.FC<MovieCardProps> = ({ movie, rank, score, reason, readonly = false }) => {
  const { addInteraction, interactions } = useApp();
  const [rated, setRated] = useState<number | null>(null);

  // Check if interacted in current session
  const hasInteracted = interactions.some(i => i.movieId === movie.id);

  const handleAction = (action: Interaction['action'], ratingVal?: number) => {
    if (readonly) return;
    
    addInteraction({
      movieId: movie.id,
      action,
      rating: ratingVal,
      timestamp: Date.now() / 1000
    });
    
    if (ratingVal) setRated(ratingVal);
  };

  // Generate a deterministic color for placeholder based on id
  const hue = (movie.id * 137.5) % 360;

  return (
    <div className={`group relative bg-white dark:bg-slate-800 rounded-xl overflow-hidden border border-slate-200 dark:border-slate-700 shadow-sm hover:shadow-md transition-all ${hasInteracted ? 'opacity-50 grayscale' : ''}`}>
      <div className="flex h-32">
        {/* Placeholder Art */}
        <div 
          className="w-24 shrink-0 flex items-center justify-center text-white font-bold text-2xl"
          style={{ backgroundColor: `hsl(${hue}, 70%, 60%)` }}
        >
          {movie.title.charAt(0)}
        </div>
        
        <div className="p-4 flex-1 flex flex-col justify-between">
          <div>
            <div className="flex justify-between items-start">
              <h4 className="font-semibold text-slate-900 dark:text-slate-100 line-clamp-1" title={movie.title}>
                {rank && <span className="text-indigo-500 mr-2">#{rank}</span>}
                {movie.title}
              </h4>
              {score && (
                <span className="text-xs font-mono text-slate-400 bg-slate-100 dark:bg-slate-700 px-1.5 py-0.5 rounded">
                  {score.toFixed(1)}
                </span>
              )}
            </div>
            <div className="flex flex-wrap gap-1 mt-1">
              {movie.genres.slice(0, 3).map(g => (
                <span key={g} className="text-[10px] uppercase tracking-wider text-slate-500 border border-slate-200 dark:border-slate-700 px-1.5 rounded-sm">
                  {g}
                </span>
              ))}
            </div>
          </div>
          
          {reason && (
            <div className="flex items-center gap-1.5 mt-2 text-xs text-emerald-600 dark:text-emerald-400">
              <Info size={12} />
              <span className="truncate">{reason}</span>
            </div>
          )}
        </div>
      </div>

      {/* Interactions Overlay */}
      {!readonly && !hasInteracted && (
        <div className="absolute inset-0 bg-slate-900/80 backdrop-blur-[2px] opacity-0 group-hover:opacity-100 transition-opacity flex flex-col items-center justify-center gap-3">
          <div className="flex gap-4">
            <button 
              onClick={() => handleAction('like')}
              className="p-3 bg-white/10 hover:bg-green-500/80 rounded-full text-white transition transform hover:scale-110"
            >
              <ThumbsUp size={20} />
            </button>
            <button 
               onClick={() => handleAction('dislike')}
              className="p-3 bg-white/10 hover:bg-red-500/80 rounded-full text-white transition transform hover:scale-110"
            >
              <ThumbsDown size={20} />
            </button>
            <button 
               onClick={() => handleAction('click')}
              className="p-3 bg-white/10 hover:bg-blue-500/80 rounded-full text-white transition transform hover:scale-110"
            >
              <Play size={20} />
            </button>
          </div>
          <div className="flex gap-1">
            {[1, 2, 3, 4, 5].map(star => (
              <button key={star} onClick={() => handleAction('rate', star)} className="text-slate-400 hover:text-yellow-400 transition">
                <Star size={16} fill="currentColor" />
              </button>
            ))}
          </div>
        </div>
      )}
    </div>
  );
};

export default MovieCard;