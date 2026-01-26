import React from 'react';
import { useApp } from '../AppContext';
import { View } from '../types';
import { Activity, Database, Users, PlayCircle, Moon, Sun, MonitorPlay, Zap, Wifi, WifiOff, Loader2 } from 'lucide-react';
import { API_BASE_URL } from '../apiConfig';

const Layout: React.FC<{ children: React.ReactNode }> = ({ children }) => {
  const { view, setView, mode, setMode, apiStatus, checkApiConnection } = useApp();
  const [dark, setDark] = React.useState(false);

  React.useEffect(() => {
    if (dark) document.documentElement.classList.add('dark');
    else document.documentElement.classList.remove('dark');
  }, [dark]);

  const NavItem = ({ target, icon: Icon, label }: { target: View, icon: any, label: string }) => (
    <button
      onClick={() => setView(target)}
      className={`flex items-center gap-3 px-4 py-3 rounded-lg w-full transition-colors ${
        view === target 
          ? 'bg-indigo-50 text-indigo-700 dark:bg-indigo-900/30 dark:text-indigo-300 font-medium' 
          : 'text-slate-600 dark:text-slate-400 hover:bg-slate-100 dark:hover:bg-slate-800'
      }`}
    >
      <Icon size={20} />
      <span>{label}</span>
    </button>
  );

  return (
    <div className="flex h-screen bg-slate-50 dark:bg-slate-950 text-slate-900 dark:text-slate-100 font-sans">
      {/* Sidebar */}
      <aside className="w-64 bg-white dark:bg-slate-900 border-r border-slate-200 dark:border-slate-800 flex flex-col">
        <div className="p-6 border-b border-slate-100 dark:border-slate-800">
          <div className="flex items-center gap-2 text-indigo-600 dark:text-indigo-400 font-bold text-xl">
            <Activity />
            <span>RecSys<span className="text-slate-400 font-normal">Lab</span></span>
          </div>
        </div>

        <nav className="flex-1 p-4 space-y-1">
          <NavItem target={View.IMPORT} icon={Database} label="Dataset" />
          <NavItem target={View.TRENDING} icon={Activity} label="Trending" />
          <NavItem target={View.USERS} icon={Users} label="User Browser" />
          <NavItem target={View.SESSION} icon={PlayCircle} label="Session" />
        </nav>

        <div className="p-4 border-t border-slate-100 dark:border-slate-800 space-y-4">
          {/* Mode Toggle */}
          <div className="bg-slate-100 dark:bg-slate-800 p-1 rounded-lg flex">
            <button
              onClick={() => setMode('DEMO')}
              className={`flex-1 text-xs font-medium py-1.5 rounded-md flex items-center justify-center gap-1 transition-all ${
                mode === 'DEMO' ? 'bg-white dark:bg-slate-700 shadow-sm' : 'text-slate-500'
              }`}
            >
              <MonitorPlay size={12} /> Demo
            </button>
            <button
              onClick={() => setMode('API')}
              className={`flex-1 text-xs font-medium py-1.5 rounded-md flex items-center justify-center gap-1 transition-all ${
                mode === 'API' ? 'bg-indigo-600 text-white shadow-sm' : 'text-slate-500'
              }`}
            >
              <Zap size={12} /> API
            </button>
          </div>

          {/* API Status Indicator */}
          {mode === 'API' && (
            <div className={`text-xs px-3 py-2 rounded-lg border flex items-center justify-between transition-colors ${
              apiStatus === 'CONNECTED' 
                ? 'bg-green-50 text-green-700 border-green-200 dark:bg-green-900/20 dark:border-green-800'
                : apiStatus === 'ERROR'
                ? 'bg-red-50 text-red-700 border-red-200 dark:bg-red-900/20 dark:border-red-800'
                : 'bg-slate-50 text-slate-600 border-slate-200 dark:bg-slate-800/50'
            }`}>
              <div className="flex items-center gap-2">
                {apiStatus === 'CHECKING' && <Loader2 size={12} className="animate-spin" />}
                {apiStatus === 'CONNECTED' && <Wifi size={12} />}
                {apiStatus === 'ERROR' && <WifiOff size={12} />}
                <span>
                  {apiStatus === 'CHECKING' ? 'Connecting...' : 
                   apiStatus === 'CONNECTED' ? 'Online' : 
                   'Disconnected'}
                </span>
              </div>
              {apiStatus === 'ERROR' && (
                <button 
                  onClick={() => checkApiConnection()} 
                  className="hover:underline opacity-80"
                  title="Retry connection"
                >
                  Retry
                </button>
              )}
            </div>
          )}

          <div className="flex items-center justify-between px-2">
            <span className="text-xs text-slate-400">Appearance</span>
            <button onClick={() => setDark(!dark)} className="p-2 rounded-full hover:bg-slate-100 dark:hover:bg-slate-800 text-slate-500">
              {dark ? <Moon size={16} /> : <Sun size={16} />}
            </button>
          </div>
        </div>
      </aside>

      {/* Main Content */}
      <main className="flex-1 overflow-auto bg-slate-50 dark:bg-slate-950 p-8">
        {mode === 'DEMO' && (
          <div className="mb-6 px-4 py-2 bg-amber-50 dark:bg-amber-900/20 border border-amber-200 dark:border-amber-800 rounded-md text-sm text-amber-800 dark:text-amber-200 flex items-center gap-2 animate-in slide-in-from-top-2">
            <MonitorPlay size={16} />
            <span><strong>DEMO MODE:</strong> Running entirely in-browser. No external ML API connected.</span>
          </div>
        )}
        {mode === 'API' && apiStatus === 'ERROR' && (
          <div className="mb-6 px-4 py-2 bg-red-50 dark:bg-red-900/20 border border-red-200 dark:border-red-800 rounded-md text-sm text-red-800 dark:text-red-200 flex items-center gap-2 animate-in slide-in-from-top-2">
            <WifiOff size={16} />
            <span><strong>API DISCONNECTED:</strong> Could not reach {API_BASE_URL}. Using Demo Logic fallback.</span>
          </div>
        )}
        {children}
      </main>
    </div>
  );
};

export default Layout;