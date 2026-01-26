import React, { useMemo, useState } from 'react';
import { useApp } from '../AppContext';
import { SAMPLE_USERS_COUNT, OCCUPATION_MAP } from '../constants';
import { User } from '../types';
import { Search } from 'lucide-react';

const UserSelectView: React.FC = () => {
  const { users, selectUser } = useApp();
  const [filter, setFilter] = useState('');

  const displayUsers = useMemo(() => {
    // Deterministic shuffle based on ID to always show same "random" users
    let sample = users.slice(0, SAMPLE_USERS_COUNT * 2); 
    if (filter) {
      sample = users.filter(u => u.id.toString().includes(filter));
    }
    return sample.slice(0, SAMPLE_USERS_COUNT);
  }, [users, filter]);

  return (
    <div className="max-w-6xl mx-auto space-y-6">
      <div className="flex justify-between items-end">
        <div>
           <h2 className="text-2xl font-bold text-slate-900 dark:text-white">Select Test User</h2>
           <p className="text-slate-500 dark:text-slate-400">Impersonate a user to test personalization.</p>
        </div>
        <div className="relative">
          <Search className="absolute left-3 top-1/2 -translate-y-1/2 text-slate-400" size={18} />
          <input 
            type="text" 
            placeholder="Search User ID..." 
            value={filter}
            onChange={(e) => setFilter(e.target.value)}
            className="pl-10 pr-4 py-2 rounded-lg border border-slate-200 dark:border-slate-700 bg-white dark:bg-slate-800 focus:ring-2 focus:ring-indigo-500 outline-none"
          />
        </div>
      </div>

      <div className="grid grid-cols-2 md:grid-cols-3 lg:grid-cols-5 gap-4">
        {displayUsers.map(user => (
          <button
            key={user.id}
            onClick={() => selectUser(user)}
            className="flex flex-col items-center p-6 bg-white dark:bg-slate-800 rounded-xl border border-slate-200 dark:border-slate-700 hover:border-indigo-500 dark:hover:border-indigo-500 hover:shadow-md transition group"
          >
            <div className="w-14 h-14 rounded-full bg-slate-100 dark:bg-slate-700 text-slate-500 dark:text-slate-300 flex items-center justify-center font-bold text-lg mb-3 group-hover:bg-indigo-100 group-hover:text-indigo-600 transition-colors">
              {user.id}
            </div>
            <div className="text-center">
              <p className="font-semibold text-slate-900 dark:text-slate-100">User #{user.id}</p>
              <p className="text-xs text-slate-500 capitalize">{user.gender === 'M' ? 'Male' : 'Female'}, Age {user.age}</p>
              <p className="text-xs text-slate-400 mt-1 truncate max-w-[120px]">{OCCUPATION_MAP[user.occupation]}</p>
            </div>
          </button>
        ))}
      </div>
    </div>
  );
};

export default UserSelectView;