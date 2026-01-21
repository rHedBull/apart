/**
 * TopNav - Shared header with branding and dark mode toggle
 */

import TopNavigation from '@cloudscape-design/components/top-navigation';
import { useDarkMode } from '../hooks/useDarkMode';

export function TopNav() {
  const { mode, setMode } = useDarkMode();

  const getModeLabel = () => {
    switch (mode) {
      case 'light': return 'Light mode';
      case 'dark': return 'Dark mode';
      case 'system': return 'System';
    }
  };

  return (
    <TopNavigation
      identity={{
        href: '/',
        title: 'Apart',
        logo: {
          src: '',
          alt: 'Apart'
        }
      }}
      utilities={[
        {
          type: 'menu-dropdown',
          text: getModeLabel(),
          iconName: mode === 'dark' ? 'status-negative' : mode === 'light' ? 'status-positive' : 'settings',
          items: [
            { id: 'light', text: 'Light mode' },
            { id: 'dark', text: 'Dark mode' },
            { id: 'system', text: 'System preference' },
          ],
          onItemClick: ({ detail }) => {
            setMode(detail.id as 'light' | 'dark' | 'system');
          },
        },
      ]}
    />
  );
}
