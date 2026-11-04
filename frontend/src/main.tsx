import ReactDOM from 'react-dom/client';
import { BrowserRouter } from 'react-router-dom';
import { QueryClient, QueryClientProvider } from '@tanstack/react-query';
import App, { ErrorBoundary } from './App';
import './styles.css';

const queryClient = new QueryClient({ defaultOptions: { queries: {
  refetchOnWindowFocus: false, refetchOnReconnect: false, retryDelay: attempt => Math.min(1000 * 2 ** attempt, 4000),
} } });
window.addEventListener('vite:preloadError', event => {
  event.preventDefault();
  // Never automatically reload or resubmit an active question after deployment.
  ReactDOM.createRoot(document.getElementById('recovery') || (() => {
    const element = document.createElement('div'); element.id = 'recovery'; document.body.appendChild(element); return element;
  })()).render(<div className="update-notice notice" role="alert">An application update interrupted loading.
    <button onClick={() => window.location.reload()}>Reload app</button></div>);
});
ReactDOM.createRoot(document.getElementById('root')!).render(
  <ErrorBoundary><QueryClientProvider client={queryClient}><BrowserRouter><App /></BrowserRouter></QueryClientProvider></ErrorBoundary>,
);
