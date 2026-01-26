import React, { useState } from 'react';
import { useApp } from '../AppContext';
import { parseMovies, parseRatings, parseUsers } from '../services/dataParser';
import { Upload, FileText, CheckCircle, AlertCircle, Loader2 } from 'lucide-react';
import { Movie, Rating, User } from '../types';

const ImportView: React.FC = () => {
  const { setData } = useApp();
  const [loading, setLoading] = useState(false);
  const [files, setFiles] = useState<{
    users: File | null;
    movies: File | null;
    ratings: File | null;
  }>({ users: null, movies: null, ratings: null });

  const handleFileChange = (type: 'users' | 'movies' | 'ratings') => (e: React.ChangeEvent<HTMLInputElement>) => {
    if (e.target.files && e.target.files[0]) {
      setFiles(prev => ({ ...prev, [type]: e.target.files![0] }));
    }
  };

  const readFile = (file: File): Promise<string> => {
    return new Promise((resolve) => {
      const reader = new FileReader();
      reader.onload = (e) => resolve(e.target?.result as string);
      reader.readAsText(file);
    });
  };

  const processFiles = async () => {
    if (!files.users || !files.movies || !files.ratings) return;
    setLoading(true);

    try {
      // Small delay to allow UI to update
      await new Promise(r => setTimeout(r, 100));

      const [usersText, moviesText, ratingsText] = await Promise.all([
        readFile(files.users),
        readFile(files.movies),
        readFile(files.ratings)
      ]);

      const parsedUsers = parseUsers(usersText);
      const parsedMovies = parseMovies(moviesText);
      const parsedRatings = parseRatings(ratingsText);

      setData(parsedUsers, parsedMovies, parsedRatings);
    } catch (e) {
      console.error(e);
      alert("Error parsing files. Ensure format is correct (:: separator).");
    } finally {
      setLoading(false);
    }
  };

  const FileInput = ({ type, label, file }: { type: 'users' | 'movies' | 'ratings', label: string, file: File | null }) => (
    <div className="border border-slate-200 dark:border-slate-700 rounded-lg p-4 flex items-center justify-between bg-white dark:bg-slate-800 shadow-sm">
      <div className="flex items-center gap-3">
        <div className={`p-2 rounded-full ${file ? 'bg-green-100 text-green-600' : 'bg-slate-100 text-slate-400'} dark:bg-slate-900`}>
          {file ? <CheckCircle size={20} /> : <FileText size={20} />}
        </div>
        <div>
          <p className="font-medium text-slate-900 dark:text-slate-100">{label}</p>
          <p className="text-xs text-slate-500">{file ? file.name : "Waiting for upload..."}</p>
        </div>
      </div>
      <label className="cursor-pointer bg-indigo-50 dark:bg-indigo-900/30 text-indigo-600 dark:text-indigo-400 px-3 py-1.5 rounded-md text-sm font-medium hover:bg-indigo-100 dark:hover:bg-indigo-900/50 transition">
        Browse
        <input type="file" className="hidden" accept=".dat" onChange={handleFileChange(type)} />
      </label>
    </div>
  );

  return (
    <div className="max-w-2xl mx-auto py-12 px-4">
      <div className="text-center mb-10">
        <h1 className="text-3xl font-bold text-slate-900 dark:text-white mb-2">Initialize Dataset</h1>
        <p className="text-slate-500 dark:text-slate-400">
          Upload the <code className="bg-slate-100 dark:bg-slate-800 px-1 rounded">.dat</code> files from MovieLens 1M to begin the research session.
          Data is processed locally in your browser.
        </p>
      </div>

      <div className="space-y-4 mb-8">
        <FileInput type="users" label="Users File (users.dat)" file={files.users} />
        <FileInput type="movies" label="Movies File (movies.dat)" file={files.movies} />
        <FileInput type="ratings" label="Ratings File (ratings.dat)" file={files.ratings} />
      </div>

      <button
        onClick={processFiles}
        disabled={loading || !files.users || !files.movies || !files.ratings}
        className={`w-full py-3 rounded-lg flex items-center justify-center gap-2 font-semibold transition-all
          ${loading || !files.users || !files.movies || !files.ratings 
            ? 'bg-slate-200 text-slate-400 cursor-not-allowed dark:bg-slate-800 dark:text-slate-600' 
            : 'bg-indigo-600 text-white hover:bg-indigo-700 shadow-lg shadow-indigo-500/20'}`}
      >
        {loading ? <><Loader2 className="animate-spin" /> Processing 1M Records...</> : <><Upload size={20} /> Import & Start Environment</>}
      </button>

      <div className="mt-6 p-4 bg-amber-50 dark:bg-amber-900/20 border border-amber-200 dark:border-amber-800 rounded-md flex gap-3 text-amber-800 dark:text-amber-200 text-sm">
        <AlertCircle className="shrink-0" size={20} />
        <p>Ensure you use the standard MovieLens 1M format (:: separator). Large files may take 3-5 seconds to process depending on your CPU.</p>
      </div>
    </div>
  );
};

export default ImportView;