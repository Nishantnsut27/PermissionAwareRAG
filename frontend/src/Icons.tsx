import type { SVGProps } from 'react';

type IconName = 'assistant' | 'user' | 'send' | 'stop' | 'menu' | 'close' | 'shield' | 'chart' | 'grid' | 'refresh' | 'download';
export default function Icon({ name, ...props }: SVGProps<SVGSVGElement> & { name: IconName }) {
  return <svg width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.7" strokeLinecap="round" strokeLinejoin="round" aria-hidden="true" focusable="false" {...props}>
    {name === 'assistant' && <><path d="M12 8.5C9.2 6.8 6.2 6.4 3 7.2v12c3.2-.8 6.2-.4 9 1.3 2.8-1.7 5.8-2.1 9-1.3v-7.7M12 8.5v12M6.5 11l2 .5m-2 3 2 .5" /><path d="m18 2 1.1 3.4L22.5 6.5l-3.4 1.1L18 11l-1.1-3.4-3.4-1.1 3.4-1.1L18 2Z" /></>}
    {name === 'user' && <><circle cx="12" cy="8" r="3.5" /><path d="M5.5 20v-1.5a6.5 6.5 0 0 1 13 0V20" /></>}
    {name === 'send' && <path d="M12 19V5m-6 6 6-6 6 6" />}
    {name === 'stop' && <rect x="6.5" y="6.5" width="11" height="11" rx="2" fill="currentColor" stroke="none" />}
    {name === 'menu' && <path d="M4 6h16M4 12h16M4 18h16" />}
    {name === 'close' && <path d="m6 6 12 12M18 6 6 18" />}
    {name === 'shield' && <><path d="m12 3 8 3v5c0 5-4 8-8 10-4-2-8-5-8-10V6l8-3Z" /><path d="m8.5 12 2.5 2.5 4.5-5" /></>}
    {name === 'chart' && <><path d="M4 3v17h17" /><rect x="8" y="10" width="3" height="6" rx="1" /><rect x="15" y="5" width="3" height="11" rx="1" /></>}
    {name === 'grid' && <><rect x="3" y="3" width="18" height="18" rx="3" /><path d="M3 9h18M9 9v12M15 9v12M3 15h18" /></>}
    {name === 'refresh' && <><path d="M21 12A9 9 0 0 0 5.3 5.7L3 8M3 3v5h5M3 12a9 9 0 0 0 15.7 6.3L21 16m-5 0h5v5" /></>}
    {name === 'download' && <><path d="M12 3v12m-4-4 4 4 4-4M4 16v4a1 1 0 0 0 1 1h14a1 1 0 0 0 1-1v-4" /></>}
  </svg>;
}
