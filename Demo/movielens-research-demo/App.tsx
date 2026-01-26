import React from 'react';
import { AppProvider, useApp } from './AppContext';
import Layout from './components/Layout';
import ImportView from './components/ImportView';
import TrendingPanel from './components/TrendingPanel';
import UserSelectView from './components/UserSelectView';
import SessionView from './components/SessionView';
import { View } from './types';

const MainContent: React.FC = () => {
  const { view, users } = useApp();

  // Route guard
  if (view !== View.IMPORT && users.length === 0) {
    return <ImportView />;
  }

  switch (view) {
    case View.IMPORT: return <ImportView />;
    case View.TRENDING: return <TrendingPanel />;
    case View.USERS: return <UserSelectView />;
    case View.SESSION: return <SessionView />;
    default: return <ImportView />;
  }
};

const App: React.FC = () => {
  return (
    <AppProvider>
      <Layout>
        <MainContent />
      </Layout>
    </AppProvider>
  );
};

export default App;