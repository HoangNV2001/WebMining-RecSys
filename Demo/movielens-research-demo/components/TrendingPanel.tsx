import React, { useMemo, useState } from 'react';
import { useApp } from '../AppContext';
import { BarChart, Bar, XAxis, YAxis, Tooltip, ResponsiveContainer, Cell } from 'recharts';
import { Calendar, Film, Star, Users, Filter } from 'lucide-react';
import MovieCard from './MovieCard';

const StatCard = ({ icon: Icon, label, value }: { icon: any, label: string, value: string }) => (
  <div className="bg-white dark:bg-slate-800 p-6 rounded-xl border border-slate-200 dark:border-slate-700 shadow-sm">
    <div className="flex items-center gap-4">
      <div className="p-3 bg-indigo-50 dark:bg-indigo-900/30 text-indigo-600 dark:text-indigo-400 rounded-lg">
        <Icon size={24} />
      </div>
      <div>
        <p className="text-sm text-slate-500 dark:text-slate-400 font-medium">{label}</p>
        <p className="text-2xl font-bold text-slate-900 dark:text-white">{value}</p>
      </div>
    </div>
  </div>
);

const TrendingPanel: React.FC = () => {
  const { stats, engine, movies } = useApp();
  const [timeWindow, setTimeWindow] = useState<number>(30); // days
  const [selectedGenre, setSelectedGenre] = useState<string>('');

  // Extract all available genres from the movie list
  const allGenres = useMemo(() => {
    const genres = new Set<string>();
    movies.forEach(m => m.genres.forEach(g => genres.add(g)));
    return Array.from(genres).sort();
  }, [movies]);

  const trendingMovies = useMemo(() => {
    // Pass selectedGenre to the engine filter
    return engine ? engine.getTrending(6, timeWindow, selectedGenre || null) : [];
  }, [engine, timeWindow, selectedGenre]);

  const genreDist = useMemo(() => {
    if (!engine) return [];
    const counts: Record<string, number> = {};
    trendingMovies.forEach(rec => {
      rec.movie.genres.forEach(g => counts[g] = (counts[g] || 0) + 1);
    });
    return Object.entries(counts).map(([name, value]) => ({ name, value }));
  }, [trendingMovies, engine]);

  if (!stats) return null;

  return (
    <div className="space-y-8 animate-in fade-in duration-500">
      <div className="grid grid-cols-1 md:grid-cols-4 gap-4">
        <StatCard icon={Users} label="Total Users" value={stats.userCount.toLocaleString()} />
        <StatCard icon={Film} label="Movies" value={stats.movieCount.toLocaleString()} />
        <StatCard icon={Star} label="Ratings" value={stats.ratingCount.toLocaleString()} />
        <StatCard icon={Calendar} label="Date Range" value={`${new Date(stats.minTimestamp * 1000).getFullYear()} - ${new Date(stats.maxTimestamp * 1000).getFullYear()}`} />
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-3 gap-8">
        <div className="lg:col-span-2 space-y-6">
          <div className="flex flex-wrap items-center justify-between gap-4">
            <h2 className="text-xl font-bold text-slate-900 dark:text-white flex items-center gap-2">
              <span className="w-2 h-6 bg-rose-500 rounded-full"></span>
              Trending Now
            </h2>
            
            <div className="flex items-center gap-3">
              {/* Genre Filter */}
              <div className="relative">
                <Filter className="absolute left-2.5 top-1/2 -translate-y-1/2 text-slate-400" size={14} />
                <select 
                  value={selectedGenre}
                  onChange={(e) => setSelectedGenre(e.target.value)}
                  className="pl-8 pr-4 py-1.5 text-sm bg-white dark:bg-slate-800 border border-slate-200 dark:border-slate-700 rounded-lg outline-none focus:ring-2 focus:ring-indigo-500 text-slate-700 dark:text-slate-300"
                >
                  <option value="">All Genres</option>
                  {allGenres.map(g => (
                    <option key={g} value={g}>{g}</option>
                  ))}
                </select>
              </div>

              {/* Time Window Filter */}
              <div className="flex bg-white dark:bg-slate-800 rounded-lg border border-slate-200 dark:border-slate-700 p-1">
                {[7, 30, 365].map(d => (
                  <button
                    key={d}
                    onClick={() => setTimeWindow(d)}
                    className={`px-3 py-1 text-xs font-medium rounded-md transition ${
                      timeWindow === d 
                        ? 'bg-slate-900 text-white dark:bg-slate-100 dark:text-slate-900' 
                        : 'text-slate-500 hover:text-slate-900 dark:text-slate-400'
                    }`}
                  >
                    {d === 365 ? 'All Time' : `${d} Days`}
                  </button>
                ))}
              </div>
            </div>
          </div>

          <div className="grid grid-cols-1 sm:grid-cols-2 gap-4">
            {trendingMovies.length === 0 ? (
              <div className="col-span-2 py-10 text-center text-slate-400 border border-dashed border-slate-200 dark:border-slate-800 rounded-xl">
                No trending movies found for this filter combination.
              </div>
            ) : (
              trendingMovies.map((rec, idx) => (
                <MovieCard 
                  key={rec.movie.id} 
                  movie={rec.movie} 
                  rank={idx + 1} 
                  score={rec.score} 
                  reason={rec.reason}
                  readonly
                />
              ))
            )}
          </div>
        </div>

        <div className="space-y-6">
          <div className="bg-white dark:bg-slate-800 p-6 rounded-xl border border-slate-200 dark:border-slate-700 shadow-sm h-full">
            <h3 className="font-semibold text-slate-900 dark:text-white mb-6">Trending Genres Distribution</h3>
            <div className="h-64">
              <ResponsiveContainer width="100%" height="100%">
                <BarChart data={genreDist}>
                  <XAxis dataKey="name" fontSize={10} interval={0} stroke="#94a3b8" />
                  <YAxis fontSize={12} stroke="#94a3b8" />
                  <Tooltip 
                    contentStyle={{ borderRadius: '8px', border: 'none', boxShadow: '0 4px 6px -1px rgb(0 0 0 / 0.1)' }}
                    cursor={{fill: 'transparent'}}
                  />
                  <Bar dataKey="value" radius={[4, 4, 0, 0]}>
                    {genreDist.map((entry, index) => (
                      <Cell key={`cell-${index}`} fill={['#6366f1', '#8b5cf6', '#ec4899', '#f43f5e'][index % 4]} />
                    ))}
                  </Bar>
                </BarChart>
              </ResponsiveContainer>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
};

export default TrendingPanel;