import React, { useState, useEffect, useMemo } from 'react';
import { useApp } from '../AppContext';
import { Movie, Recommendation, Interaction } from '../types';
import { OCCUPATION_MAP, AGE_MAP } from '../constants';
import { Sparkles, RefreshCw, User, History, Film } from 'lucide-react';
import MovieCard from './MovieCard';
import { API_CONFIG, RecommendationRequest, RecommendationResponse, HistoryResponse } from '../apiConfig';

const SessionView: React.FC = () => {
  const { currentUser, interactions, engine, mode, movies, ratings } = useApp();
  const [recommendations, setRecommendations] = useState<Recommendation[]>([]);
  const [loading, setLoading] = useState(false);
  const [reqCount, setReqCount] = useState(0);
  const [history, setHistory] = useState<Interaction[]>([]);

  // Helper to look up movie details
  const getMovieById = (id: number) => movies.find(m => m.id === id);

  // 1. Fetch User History (Offline or API)
  useEffect(() => {
    if (!currentUser) return;
    
    // Clear previous history immediately when user switches
    setHistory([]);

    const fetchHistory = async () => {
      if (mode === 'API') {
        try {
          const res = await fetch(`${API_CONFIG.ENDPOINTS.HISTORY}?user_id=${currentUser.id}&limit=20`);
          if (res.ok) {
            const data: HistoryResponse = await res.json();
            const mappedHistory: Interaction[] = data.history.map(h => ({
              movieId: h.movie_id,
              action: 'rate',
              rating: h.rating,
              timestamp: h.timestamp
            }));
            setHistory(mappedHistory);
          }
        } catch (e) {
          console.warn("Failed to fetch history from API", e);
        }
      } else {
        // DEMO Mode: Filter from local ratings
        // Small timeout to simulate load
        setTimeout(() => {
          const userHistory = ratings
            .filter(r => r.userId === currentUser.id)
            .sort((a, b) => b.timestamp - a.timestamp)
            .slice(0, 50)
            .map(r => ({
              movieId: r.movieId,
              action: 'rate' as const,
              rating: r.rating,
              timestamp: r.timestamp
            }));
          setHistory(userHistory);
        }, 100);
      }
    };

    fetchHistory();
  }, [currentUser, mode, ratings]);

  // Combine real-time session interactions with historical data
  const displayActivity = useMemo(() => {
    // Current session on top (reversed), then history
    return [...interactions.slice().reverse(), ...history];
  }, [interactions, history]);

  // 2. Fetch Recommendations
  const fetchRecommendations = async () => {
    if (!currentUser) return;
    setLoading(true);

    // Short delay to prevent UI flickering and allow Feedback API to initiate first
    await new Promise(r => setTimeout(r, 300));

    if (mode === 'API') {
      const requestPayload: RecommendationRequest = {
        user_id: currentUser.id,
        top_k: 10
      };

      try {
        console.log(`[API CALL] Fetching Recommendations for User ${currentUser.id}...`);
        // ACTUAL API CALL
        const response = await fetch(API_CONFIG.ENDPOINTS.RECOMMEND, {
          method: 'POST',
          headers: {
            'Content-Type': 'application/json',
          },
          body: JSON.stringify(requestPayload),
        });

        if (!response.ok) {
          throw new Error(`Server returned ${response.status}`);
        }

        const data: RecommendationResponse = await response.json();
        console.log(`[API RESPONSE] Received ${data.recommendations.length} items`);

        // Map backend IDs to full Movie objects
        // AND Client-side filter: Remove movies already in interactions to ensure "Fresh Batch"
        const apiRecs: Recommendation[] = data.recommendations
          .map(item => {
            const movie = getMovieById(item.movie_id);
            if (!movie) return null;
            return {
              movie,
              score: item.score,
              reason: `ML Model Score: ${item.score.toFixed(4)}`
            };
          })
          .filter((r): r is Recommendation => r !== null)
          .filter(r => !interactions.some(i => i.movieId === r.movie.id)); // Client-side safeguard

        setRecommendations(apiRecs);

      } catch (error) {
        console.warn("API Call Failed, falling back to local engine:", error);
        
        // Fallback to local engine so the UI doesn't break
        if (engine) {
          const recs = engine.getRecommendations(currentUser.id, interactions);
          setRecommendations(recs.map(r => ({...r, reason: 'Offline Fallback (Connection Error)'})));
        }
      }
    } else {
      // Demo Mode
      if (engine) {
        console.log("[DEMO] Generating local recommendations...");
        const recs = engine.getRecommendations(currentUser.id, interactions);
        setRecommendations(recs);
      }
    }
    setLoading(false);
  };

  // Re-fetch when user explicitly asks OR when interactions change (Real-time feedback loop)
  // Added 'mode' to dependency to trigger refresh when switching between API/Demo
  useEffect(() => {
    fetchRecommendations();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [reqCount, interactions, mode]); 

  if (!currentUser) return <div>No user selected</div>;

  return (
    <div className="grid grid-cols-1 lg:grid-cols-4 gap-8 h-[calc(100vh-100px)]">
      {/* Sidebar: User Profile */}
      <div className="lg:col-span-1 space-y-6">
        <div className="bg-white dark:bg-slate-800 p-6 rounded-xl border border-slate-200 dark:border-slate-700 shadow-sm sticky top-6">
          <div className="flex items-center gap-4 mb-6">
            <div className="w-16 h-16 bg-gradient-to-br from-indigo-500 to-purple-600 rounded-full flex items-center justify-center text-white text-2xl font-bold shadow-lg">
              {currentUser.id % 100}
            </div>
            <div>
              <h3 className="font-bold text-lg text-slate-900 dark:text-white">User #{currentUser.id}</h3>
              <span className="px-2 py-0.5 bg-slate-100 dark:bg-slate-700 text-slate-600 dark:text-slate-300 text-xs rounded-full">
                {displayActivity.length} interactions
              </span>
            </div>
          </div>

          <div className="space-y-4 text-sm">
            <div className="flex justify-between border-b border-slate-100 dark:border-slate-700 pb-2">
              <span className="text-slate-500">Gender</span>
              <span className="font-medium dark:text-slate-200">{currentUser.gender === 'M' ? 'Male' : 'Female'}</span>
            </div>
            <div className="flex justify-between border-b border-slate-100 dark:border-slate-700 pb-2">
              <span className="text-slate-500">Age Group</span>
              <span className="font-medium dark:text-slate-200">{AGE_MAP[currentUser.age]}</span>
            </div>
            <div className="flex justify-between border-b border-slate-100 dark:border-slate-700 pb-2">
              <span className="text-slate-500">Occupation</span>
              <span className="font-medium dark:text-slate-200 text-right">{OCCUPATION_MAP[currentUser.occupation]}</span>
            </div>
            <div className="flex justify-between pb-2">
              <span className="text-slate-500">Zip Code</span>
              <span className="font-medium dark:text-slate-200">{currentUser.zipCode}</span>
            </div>
          </div>

          <div className="mt-8">
             <h4 className="text-xs font-semibold uppercase text-slate-400 mb-3 flex items-center gap-2">
               <History size={12} /> Activity History
             </h4>
             <div className="space-y-3 max-h-60 overflow-y-auto pr-1 custom-scrollbar">
               {displayActivity.length === 0 && <p className="text-xs text-slate-400 italic">No activity yet.</p>}
               {displayActivity.map((act, i) => {
                 const movie = getMovieById(act.movieId);
                 const isCurrentSession = i < interactions.length; 
                 return (
                   <div key={i} className={`flex items-start gap-2 border-b border-slate-50 dark:border-slate-700/50 pb-2 last:border-0 ${!isCurrentSession ? 'opacity-75' : ''}`}>
                     <div className="mt-0.5 text-slate-400">
                       <Film size={12} />
                     </div>
                     <div className="flex-1 min-w-0">
                       <p className="text-xs font-medium text-slate-700 dark:text-slate-300 truncate" title={movie?.title}>
                         {movie ? movie.title : `Movie #${act.movieId}`}
                       </p>
                       <div className="flex justify-between items-start mt-1">
                          {/* Updated Genre Display: Show as Tags */}
                          <div className="flex flex-wrap gap-1 max-w-[70%]">
                            {movie?.genres.slice(0, 2).map(g => (
                              <span key={g} className="text-[9px] px-1 py-0.5 rounded-sm bg-slate-100 text-slate-500 dark:bg-slate-700 dark:text-slate-400 leading-none">
                                {g}
                              </span>
                            ))}
                            {movie && movie.genres.length > 2 && (
                              <span className="text-[9px] text-slate-400 leading-none py-0.5">+{movie.genres.length - 2}</span>
                            )}
                          </div>

                          <span className={`text-[10px] font-bold uppercase shrink-0 ${
                            act.action === 'like' ? 'text-green-600 bg-green-100 dark:bg-green-900/30 px-1 rounded' : 
                            act.action === 'dislike' ? 'text-red-600 bg-red-100 dark:bg-red-900/30 px-1 rounded' : 
                            act.action === 'rate' ? 'text-amber-600 bg-amber-100 dark:bg-amber-900/30 px-1 rounded' :
                            'text-blue-600 bg-blue-100 dark:bg-blue-900/30 px-1 rounded'
                          }`}>
                            {act.action === 'rate' ? `${act.rating} ★` : act.action}
                          </span>
                       </div>
                     </div>
                   </div>
                 );
               })}
             </div>
          </div>
        </div>
      </div>

      {/* Main: Recommendations */}
      <div className="lg:col-span-3 space-y-6 flex flex-col">
        <div className="flex items-center justify-between">
          <div className="flex items-center gap-3">
            <h2 className="text-2xl font-bold text-slate-900 dark:text-white">Recommended for you</h2>
            {mode === 'DEMO' && (
              <span className="px-2 py-1 bg-amber-100 dark:bg-amber-900/40 text-amber-700 dark:text-amber-400 text-xs font-bold rounded uppercase tracking-wider">
                Demo Logic
              </span>
            )}
             {mode === 'API' && (
              <span className="px-2 py-1 bg-green-100 dark:bg-green-900/40 text-green-700 dark:text-green-400 text-xs font-bold rounded uppercase tracking-wider">
                Model API
              </span>
            )}
          </div>
          <button 
            onClick={() => setReqCount(c => c + 1)}
            disabled={loading}
            className="flex items-center gap-2 px-4 py-2 bg-indigo-600 hover:bg-indigo-700 text-white rounded-lg font-medium transition shadow-lg shadow-indigo-500/20 disabled:opacity-70"
          >
            <RefreshCw size={18} className={loading ? 'animate-spin' : ''} />
            {loading ? 'Thinking...' : 'Refresh List'}
          </button>
        </div>

        <div className="flex-1 overflow-y-auto pr-2 pb-10">
          <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 xl:grid-cols-4 gap-4">
            {recommendations.map((rec, idx) => (
              <MovieCard 
                key={rec.movie.id} 
                movie={rec.movie} 
                score={rec.score} 
                reason={rec.reason}
              />
            ))}
          </div>
          
          {recommendations.length > 0 && (
            <div className="mt-12 p-8 border-2 border-dashed border-slate-200 dark:border-slate-700 rounded-xl text-center">
              <Sparkles className="mx-auto text-slate-400 mb-2" size={32} />
              <p className="text-slate-500 dark:text-slate-400 mb-4">Interact with movies above to refine the next batch.</p>
              <button 
                 onClick={() => setReqCount(c => c + 1)}
                 className="text-indigo-600 dark:text-indigo-400 font-semibold hover:underline"
              >
                Load Next Batch &rarr;
              </button>
            </div>
          )}
        </div>
      </div>
    </div>
  );
};

export default SessionView;