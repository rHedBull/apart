/**
 * Apart Dashboard - Main Application with Routing
 */

import { Routes, Route } from 'react-router-dom';
import { RunsListPage } from './pages/RunsListPage';
import { RunDetailPage } from './pages/RunDetailPage';

function App() {
  return (
    <Routes>
      <Route path="/" element={<RunsListPage />} />
      <Route path="/runs/:runId" element={<RunDetailPage />} />
    </Routes>
  );
}

export default App;
