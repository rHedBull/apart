/**
 * Dark mode state management with Cloudscape integration
 */

import { create } from 'zustand';
import { persist } from 'zustand/middleware';
import { applyMode, Mode } from '@cloudscape-design/global-styles';

interface DarkModeState {
  mode: 'light' | 'dark' | 'system';
  effectiveMode: 'light' | 'dark';
  setMode: (mode: 'light' | 'dark' | 'system') => void;
}

function getSystemPreference(): 'light' | 'dark' {
  if (typeof window !== 'undefined' && window.matchMedia) {
    return window.matchMedia('(prefers-color-scheme: dark)').matches ? 'dark' : 'light';
  }
  return 'light';
}

function applyCloudscapeMode(effectiveMode: 'light' | 'dark') {
  applyMode(effectiveMode === 'dark' ? Mode.Dark : Mode.Light);
}

export const useDarkMode = create<DarkModeState>()(
  persist(
    (set) => ({
      mode: 'system',
      effectiveMode: getSystemPreference(),

      setMode: (mode) => {
        const effectiveMode = mode === 'system' ? getSystemPreference() : mode;
        applyCloudscapeMode(effectiveMode);
        set({ mode, effectiveMode });
      },
    }),
    {
      name: 'apart-dark-mode',
      onRehydrateStorage: () => (state) => {
        // Apply mode on rehydration
        if (state) {
          const effectiveMode = state.mode === 'system' ? getSystemPreference() : state.mode;
          applyCloudscapeMode(effectiveMode);
          // Update effectiveMode in case system preference changed
          if (state.effectiveMode !== effectiveMode) {
            state.effectiveMode = effectiveMode;
          }
        }
      },
    }
  )
);

// Listen for system preference changes
if (typeof window !== 'undefined' && window.matchMedia) {
  window.matchMedia('(prefers-color-scheme: dark)').addEventListener('change', (e) => {
    const state = useDarkMode.getState();
    if (state.mode === 'system') {
      const effectiveMode = e.matches ? 'dark' : 'light';
      applyCloudscapeMode(effectiveMode);
      useDarkMode.setState({ effectiveMode });
    }
  });
}
