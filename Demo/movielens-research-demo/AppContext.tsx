import React, { createContext, useContext, useState, useMemo, useEffect } from 'react';
import { AppMode, DatasetStats, Interaction, Movie, Rating, User, View, ApiConnectionStatus } from './types';
import { RecommendationEngine } from './services/engine';
import { API_CONFIG, FeedbackRequest } from './apiConfig';

interface AppState {
  mode: AppMode;
  setMode: (mode: AppMode) => void;
  apiStatus: ApiConnectionStatus;
  checkApiConnection: () => Promise<void>;
  
  view: View;
  setView: (view: View) => void;
  
  // Data
  users: User[];
  movies: Movie[];
  ratings: Rating[];
  stats: DatasetStats | null;
  setData: (u: User[], m: Movie[], r: Rating[]) => void;
  
  // Session
  currentUser: User | null;
  selectUser: (u: User) => void;
  interactions: Interaction[];
  addInteraction: (i: Interaction) => void;
  
  // Engine
  engine: RecommendationEngine | null;
}

const AppContext = createContext<AppState | undefined>(undefined);

export const AppProvider: React.FC<{ children: React.ReactNode }> = ({ children }) => {
  const [mode, setMode] = useState<AppMode>('DEMO');
  const [apiStatus, setApiStatus] = useState<ApiConnectionStatus>('IDLE');
  const [view, setView] = useState<View>(View.IMPORT);
  
  const [users, setUsers] = useState<User[]>([]);
  const [movies, setMovies] = useState<Movie[]>([]);
  const [ratings, setRatings] = useState<Rating[]>([]);
  const [stats, setStats] = useState<DatasetStats | null>(null);
  
  const [currentUser, setCurrentUser] = useState<User | null>(null);
  const [interactions, setInteractions] = useState<Interaction[]>([]);

  const engine = useMemo(() => {
    if (movies.length > 0 && ratings.length > 0) {
      return new RecommendationEngine(movies, ratings);
    }
    return null;
  }, [movies, ratings]);

  const checkApiConnection = async () => {
    setApiStatus('CHECKING');
    try {
      const controller = new AbortController();
      const timeoutId = setTimeout(() => controller.abort(), 2000); // 2s timeout
      
      const res = await fetch(API_CONFIG.ENDPOINTS.HEALTH, { 
        method: 'GET',
        signal: controller.signal 
      });
      
      clearTimeout(timeoutId);

      if (res.ok) {
        setApiStatus('CONNECTED');
      } else {
        setApiStatus('ERROR');
      }
    } catch (e) {
      console.warn("API Health Check Failed:", e);
      setApiStatus('ERROR');
    }
  };

  // Auto-check when switching to API mode
  useEffect(() => {
    if (mode === 'API') {
      checkApiConnection();
    } else {
      setApiStatus('IDLE');
    }
  }, [mode]);

  const setData = (u: User[], m: Movie[], r: Rating[]) => {
    setUsers(u);
    setMovies(m);
    setRatings(r);
    
    // Calc stats iteratively to avoid call stack overflow with large arrays
    let minTimestamp = Infinity;
    let maxTimestamp = -Infinity;

    if (r.length > 0) {
      for (let i = 0; i < r.length; i++) {
        const val = r[i].timestamp;
        if (val < minTimestamp) minTimestamp = val;
        if (val > maxTimestamp) maxTimestamp = val;
      }
    } else {
      minTimestamp = 0;
      maxTimestamp = 0;
    }

    setStats({
      userCount: u.length,
      movieCount: m.length,
      ratingCount: r.length,
      minTimestamp,
      maxTimestamp,
    });
    
    setView(View.TRENDING);
  };

  const selectUser = (u: User) => {
    setCurrentUser(u);
    setInteractions([]); // Reset session interactions
    setView(View.SESSION);
  };

  const addInteraction = (i: Interaction) => {
    setInteractions(prev => [...prev, i]);
    
    // Construct the payload matching the API config
    if (currentUser) {
        const payload: FeedbackRequest = {
            user_id: currentUser.id,
            movie_id: i.movieId,
            action: i.action,
            rating: i.rating,
            timestamp: i.timestamp
        };

        if (mode === 'API') {
            // Attempt to send, but don't block UI if it fails
            fetch(API_CONFIG.ENDPOINTS.FEEDBACK, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify(payload)
            }).catch(e => console.warn("Feedback API failed:", e));
            
            console.log(`[API CALL] POST ${API_CONFIG.ENDPOINTS.FEEDBACK}`, JSON.stringify(payload, null, 2));
        }
    }
  };

  return (
    <AppContext.Provider value={{
      mode, setMode, apiStatus, checkApiConnection,
      view, setView,
      users, movies, ratings, stats, setData,
      currentUser, selectUser,
      interactions, addInteraction,
      engine
    }}>
      {children}
    </AppContext.Provider>
  );
};

export const useApp = () => {
  const context = useContext(AppContext);
  if (!context) throw new Error("useApp must be used within AppProvider");
  return context;
};