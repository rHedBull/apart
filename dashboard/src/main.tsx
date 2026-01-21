import React from 'react';
import ReactDOM from 'react-dom/client';
import { BrowserRouter } from 'react-router-dom';
import '@cloudscape-design/global-styles/index.css';
import './index.css';
import App from './App';

// Initialize dark mode from stored preference
import { useDarkMode } from './hooks/useDarkMode';
useDarkMode.getState(); // Triggers rehydration and applies mode

ReactDOM.createRoot(document.getElementById('root')!).render(
  <React.StrictMode>
    <BrowserRouter>
      <App />
    </BrowserRouter>
  </React.StrictMode>
);
