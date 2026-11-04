import { Component, Suspense, lazy, useEffect, useLayoutEffect, useReducer, useRef, useState, type ReactNode } from 'react';
import { Navigate, Route, Routes, useLocation, useNavigate } from 'react-router-dom';
import { useQuery } from '@tanstack/react-query';
import * as api from './api';
import { cleanAnswer, prompts, SELLERS } from './content';
import { history, loadState, reducer, saveState, type State } from './state';
import type { Session } from './types';
import Icon from './Icons';

const AnswerContent = lazy(() => import('./AnswerContent'));
const Evaluation = lazy(() => import('./Evaluation'));
export class ErrorBoundary extends Component<{ children: ReactNode }, { failed: boolean }> {
  state = { failed: false };
  static getDerivedStateFromError() { return { failed: true }; }
  render() {
    return this.state.failed ? <div className="recovery panel" role="alert"><h1>The workspace could not load.</h1>
      <p>A connection or application update may have interrupted loading. Your saved tab session will be restored after reloading.</p>
      <button className="primary" onClick={() => window.location.reload()}>Reload app</button></div> : this.props.children;
  }
}
function Brand() {
  return <div className="brand"><img src="/knowledge-mark.svg" width="42" height="42" alt="" /><div><strong>Enterprise Knowledge AI</strong><span>Permission-aware assistant</span></div></div>;
}
function Hero({ login = false, name = '' }: { login?: boolean; name?: string }) {
  return <div className="hero"><div className="eyebrow">Permission-aware knowledge assistant</div>
    <h1>{login ? 'Knowledge answers, scoped to your access.' : `How can I help, ${name}?`}</h1>
    <p>{login ? 'Choose an account to explore the knowledge workspace.' : 'Ask about the information you are authorized to access.'}</p></div>;
}
function Login({ onLogin }: { onLogin: (session: Session) => void }) {
  const directory = useQuery({ queryKey: ['demo-users', api.API_BASE_URL], queryFn: ({ signal }) => api.listUsers(signal), staleTime: 300000, retry: 2 });
  const [selected, setSelected] = useState('');
  const [password, setPassword] = useState('');
  const [error, setError] = useState('');
  const [busy, setBusy] = useState(false);
  const selectedId = selected || directory.data?.[0]?.user_id || '';
  async function submit(event: React.FormEvent) {
    event.preventDefault();
    if (!selectedId || busy) return;
    setBusy(true); setError('');
    try { onLogin(await api.login(selectedId, password)); }
    catch (error) { setError(error instanceof Error ? error.message : 'Unable to sign in.'); }
    finally { setBusy(false); }
  }
  return <><Hero login /><section className="login-card panel" aria-labelledby="login-title">
    <div className="eyebrow">Demo identity</div><h2 id="login-title">Explore a sample account</h2>
    <p className="caption">Select a sample identity. The backend will apply its permissions to your questions.</p>
    {directory.isPending && <div className="notice" role="status">Connecting to the demo service… The service may be waking up.</div>}
    {directory.error && <div className="notice error" role="alert">{directory.error.message}</div>}
    {directory.error && <button onClick={() => void directory.refetch()}>Retry connection</button>}
    <form onSubmit={submit}>
      <label>Demo account<select value={selectedId} onChange={event => setSelected(event.target.value)} disabled={!directory.data || busy}>
        {!directory.data && <option value="">Loading accounts…</option>}
        {directory.data?.map(user => <option key={user.user_id} value={user.user_id}>{user.name} · {user.role}</option>)}
      </select></label>
      <label>Demo password<input type="password" autoComplete="current-password" value={password} onChange={event => setPassword(event.target.value)} placeholder="firstname@login" required disabled={busy} /></label>
      <p className="caption">Password example: aditya@login. Use the selected account's first name.</p>
      {error && <div className="notice error" role="alert">{error}</div>}
      <button className="primary full" disabled={!directory.data || busy}>{busy ? 'Signing in…' : 'Continue'}</button>
    </form>
    <details><summary>Demo IDs and password examples</summary>{directory.data?.map(user => <p className="caption" key={user.user_id}>{user.user_id} · {user.name} · {user.name.split(' ')[0].toLowerCase()}@login</p>)}</details>
  </section></>;
}
function ServiceStatus() {
  const [status, setStatus] = useState<{ ready: boolean; problems: unknown[] } | null>(null);
  const [busy, setBusy] = useState(false);
  const controller = useRef<AbortController | null>(null);
  useEffect(() => () => controller.current?.abort(), []);
  async function check() {
    setBusy(true); controller.current = new AbortController();
    try { setStatus(await api.ready(controller.current.signal)); }
    catch { if (!controller.current.signal.aborted) setStatus({ ready: false, problems: ['API cannot be reached.'] }); }
    finally { setBusy(false); }
  }
  return <section className="sidebar-section"><h2>Service status</h2><button onClick={check} disabled={busy}>{busy ? 'Checking…' : 'Check service'}</button>
    {status ? <><p className="caption" role="status"><span className={`dot ${status.ready ? 'success' : ''}`} />{status.ready ? 'Backend and knowledge index ready' : 'Service unavailable or not ready'}</p>
      {status.problems.map((problem, index) => <p className="caption" key={index}>{String(problem)}</p>)}<p className="caption">Last requested check.</p></>
      : <p className="caption">Check the connection and knowledge index when needed.</p>}</section>;
}
function Messages({ state, progress }: { state: State; progress: string }) {
  const bottom = useRef<HTMLDivElement>(null);
  useEffect(() => { bottom.current?.scrollIntoView?.({ block: 'end' }); }, [state.messages, progress]);
  return <div className="messages">{state.messages.map(message => <article className={`message ${message.role}`} key={message.id}>
    <div className={`avatar avatar-${message.role}`} aria-hidden="true"><Icon name={message.role === 'user' ? 'user' : 'assistant'} /></div>
    <div className="message-body"><div className="message-author">{message.role === 'user' ? 'You' : 'Knowledge assistant'}</div>
      {message.role === 'user' ? <p className="user-text">{message.content}</p>
        : message.outcome === 'pending' ? <div className="working" role="status"><span className="dot pulse" />{progress}</div>
        : message.outcome === 'refusal' ? <div className="notice refusal"><strong>▣ No authorized information found</strong><p>I couldn't find sufficient authorized information to answer this request.</p></div>
        : message.outcome === 'error' || message.outcome === 'interrupted' ? <div className="notice error" role="alert">{message.content}</div>
        : <Suspense fallback={<p>{message.content}</p>}><AnswerContent text={message.content} sources={message.sources} copy={message.outcome === 'answer'} /></Suspense>}
      {message.outcome === 'partial' && <p className="notice warning">Partial answer: generation ended before completion. This answer is not in conversation memory.</p>}
      {message.stats && <p className="stats">scope {message.effective_sellers?.join(', ') || 'none'} · candidates {message.stats.candidates ?? 0} · authorized {message.stats.authorized ?? 0} · used {message.stats.selected ?? 0} · withheld {message.stats.withheld ?? 0} · {message.stats.latency_ms ?? 0} ms</p>}
      {message.outcome === 'answer' && <p className="caption completion">Answer completed.</p>}
    </div></article>)}<div ref={bottom} /></div>;
}
export default function App() {
  const [state, dispatch] = useReducer(reducer, undefined, loadState);
  const [question, setQuestion] = useState('');
  const [progress, setProgress] = useState('Working on your request…');
  const [banner, setBanner] = useState('');
  const [restoreError, setRestoreError] = useState('');
  const [restoreAttempt, setRestoreAttempt] = useState(0);
  const [menu, setMenu] = useState(false);
  const request = useRef<AbortController | null>(null);
  const input = useRef<HTMLTextAreaElement>(null);
  const navigate = useNavigate();
  const location = useLocation();
  const token = state.session?.token;
  const busy = state.messages.some(message => message.outcome === 'pending');
  const canAsk = !!state.session && state.verified && !busy;
  useLayoutEffect(() => {
    if (!input.current) return;
    input.current.style.height = 'auto';
    input.current.style.height = `${Math.min(input.current.scrollHeight, 180)}px`;
  }, [question, location.pathname, !!state.session]);
  useEffect(() => { saveState(state); document.documentElement.dataset.theme = state.theme; }, [state]);
  useEffect(() => () => request.current?.abort(), []);
  useEffect(() => {
    if (!token || state.verified) return;
    const controller = new AbortController(); setRestoreError('');
    api.me(token, controller.signal).then(user => dispatch({ type: 'verified', user })).catch(error => {
      if (controller.signal.aborted) return;
      if (error instanceof api.ApiError && error.status === 401) {
        dispatch({ type: 'logout' }); setQuestion(''); setBanner('Your demo session has ended. Please sign in again.');
      } else setRestoreError(error instanceof Error ? error.message : 'Unable to restore your session.');
    });
    return () => controller.abort();
  }, [token, state.verified, restoreAttempt]);
  useEffect(() => {
    if (!state.session) return;
    const timer = setTimeout(() => { request.current?.abort(); request.current = null; dispatch({ type: 'logout' }); setQuestion(''); setMenu(false); setBanner('Your demo session has expired. Please sign in again.'); }, Math.max(0, state.session.expiresAt - Date.now()));
    return () => clearTimeout(timer);
  }, [token, state.session?.expiresAt]);
  function signOut() { request.current?.abort(); request.current = null; dispatch({ type: 'logout' }); setQuestion(''); setBanner(''); setMenu(false); navigate('/login'); }
  function reset() { request.current?.abort(); request.current = null; dispatch({ type: 'reset' }); setQuestion(''); setMenu(false); navigate('/chat'); }
  function stopAnswer() {
    request.current?.abort(); request.current = null;
    const pending = state.messages.find(message => message.outcome === 'pending');
    if (pending) dispatch({ type: 'finish', message: { ...pending, outcome: 'interrupted', content: 'You stopped this answer. You can ask again when ready.' } });
  }
  async function ask(text: string) {
    const clean = text.trim();
    if (!canAsk || !state.session || !clean || clean.length > 1000 || request.current) return;
    const controller = new AbortController(); request.current = controller;
    const id = crypto.randomUUID();
    dispatch({ type: 'start', question: clean, id }); setQuestion(''); setProgress('Working on your request…'); navigate('/chat');
    try {
      const answer = await api.ask(state.session.token, clean, history(state), state.seller, controller.signal,
        type => { if (!controller.signal.aborted) setProgress(type === 'meta' ? 'Generating an answer from authorized sources…' : 'Receiving the answer…'); });
      if (controller.signal.aborted) return;
      const cleaned = cleanAnswer(answer.answer);
      dispatch({ type: 'finish', message: { id, role: 'assistant', content: cleaned.text, outcome: answer.outcome,
        sources: cleaned.noEvidence ? [] : answer.sources, stats: answer.stats, effective_sellers: answer.effective_sellers } });
    } catch (error) {
      if (controller.signal.aborted) return;
      if (error instanceof api.ApiError && error.status === 401) { signOut(); setBanner('Your demo session has ended. Please sign in again.'); return; }
      const content = error instanceof Error ? error.message : 'Something went wrong while answering. Please try again.';
      dispatch({ type: 'finish', message: { id, role: 'assistant', content, outcome: /connection ended/i.test(content) ? 'interrupted' : 'error' } });
    } finally { if (request.current === controller) request.current = null; }
  }
  const user = state.session?.user;
  return <div className="workspace">
    <header className="mobile-header"><Brand /><button aria-label={menu ? 'Close navigation' : 'Open navigation'} aria-expanded={menu} onClick={() => setMenu(!menu)}><Icon name={menu ? 'close' : 'menu'} /></button></header>
    {menu && <button className="menu-backdrop" aria-label="Close navigation" onClick={() => setMenu(false)} />}
    <aside className={`sidebar ${menu ? 'open' : ''}`} aria-label="Workspace navigation"><Brand />
      {user ? <><section className="sidebar-section"><h2>Demo identity</h2><div className="identity panel"><strong>{user.name}</strong><span className="role">{user.role}</span><span className="caption">{user.department} · {user.user_id}</span></div></section>
        <section className="sidebar-section"><h2>Authorized sellers</h2><div className="chips">{user.seller_scope.length ? user.seller_scope.map(id => <span key={id}>✓ {id}</span>) : <span>Organization scope</span>}</div>
          <p className="caption">Organization-wide documents also follow role, clearance and department rules.</p></section>
        <section className="sidebar-section"><h2>Focus (optional)</h2><select aria-label="Seller filter" value={state.seller} disabled={!state.verified || busy || !user.seller_scope.length} onChange={event => dispatch({ type: 'seller', seller: event.target.value })}>
          <option value="">All authorized sellers</option>{user.seller_scope.map(id => <option key={id} value={id}>{id} - {SELLERS[id] || id}</option>)}</select></section>
        <section className="sidebar-section"><h2>Conversation</h2><label className="caption">Conversation memory <span className="range-value">{state.memory}</span><input type="range" min="2" max="6" value={state.memory} onChange={event => dispatch({ type: 'memory', memory: Number(event.target.value) })} /></label>
          <button className="primary full" onClick={reset}>New conversation</button><button className="full" onClick={signOut}>New User</button></section>
      </> : <section className="sidebar-section how"><h2>How it works</h2><p>Choose a demo identity.</p><p>Ask about information available to your role.</p><p>Inspect the authorized sources behind each answer.</p><p className="caption">Each account has its own role, department and seller permissions.</p></section>}
      <div className="theme-control"><span>Light</span><button className={`theme-switch ${state.theme}`} role="switch" aria-checked={state.theme === 'dark'} aria-label="Dark theme" onClick={() => dispatch({ type: 'theme' })}><span /></button><span>Dark</span></div>
      {user && <section className="sidebar-section"><h2>Evaluation</h2><p className="caption">Open saved evaluation results when needed.</p><button className="full" onClick={() => { navigate(location.pathname === '/evaluation' ? '/chat' : '/evaluation'); setMenu(false); }}>{location.pathname === '/evaluation' ? 'Back to chat' : 'Evaluation overview'}</button></section>}
      <ServiceStatus /><p className="guard">▣ Authorization is enforced server-side by the permission engine before any document reaches the model.</p>
    </aside>
    <main className={`main-content ${user && location.pathname === '/chat' ? 'chat-main' : ''}`} id="main-content">
      {banner && <div className="notice" role="status">{banner}</div>}
      {user && !state.verified && <div className={`notice ${restoreError ? 'error' : ''}`} role="status">{restoreError || 'Restoring your demo session…'}
        {restoreError && <button onClick={() => setRestoreAttempt(value => value + 1)}>Retry session</button>}</div>}
      <Routes>
        <Route path="/login" element={user ? <Navigate to="/chat" replace /> : <Login onLogin={session => { dispatch({ type: 'login', session }); setQuestion(''); setBanner(''); navigate('/chat'); }} />} />
        <Route path="/chat" element={!user ? <Navigate to="/login" replace /> : <>
          <div className={`chat-scroll ${state.messages.length ? '' : 'chat-welcome'}`}>
            {state.messages.length ? <Messages state={state} progress={progress} /> : <><div className="welcome-mark"><Icon name="assistant" /></div><Hero name={user.name.split(' ')[0]} /><section className="suggested"><h2>Suggested questions</h2><div className="question-grid">{prompts(user).map(text => <button key={text} disabled={!canAsk} onClick={() => void ask(text)}>{text}</button>)}</div></section></>}
          </div>
          <div className="composer-dock"><form className="composer" onSubmit={event => { event.preventDefault(); void ask(question); }}><label className="sr-only" htmlFor="question">Ask a question</label>
            <textarea ref={input} id="question" rows={1} maxLength={1000} value={question} onChange={event => setQuestion(event.target.value)} disabled={!state.verified || busy}
              aria-describedby="composer-hint" placeholder="Ask about your workspace…" onKeyDown={event => { if (event.key === 'Enter' && !event.shiftKey && !event.nativeEvent.isComposing) { event.preventDefault(); void ask(question); } }} />
            <div className="composer-footer"><span className="composer-scope"><Icon name="shield" />{state.seller || 'Authorized sources'}</span>
              {busy ? <button type="button" className="send stop" aria-label="Stop answer" onClick={stopAnswer}><Icon name="stop" /></button>
                : <button type="submit" className="send" aria-label="Send question" disabled={!canAsk || !question.trim()}><Icon name="send" /></button>}</div></form>
            <div className="composer-hint caption" id="composer-hint"><span>Enter to send · Shift + Enter for a new line</span><span>{question.length}/1000</span></div>
          </div>
        </>} />
        <Route path="/evaluation" element={!user ? <Navigate to="/login" replace /> : <Suspense fallback={<p role="status">Loading evaluation view…</p>}><Evaluation /></Suspense>} />
        <Route path="*" element={<Navigate to={user ? '/chat' : '/login'} replace />} />
      </Routes>
    </main>
  </div>;
}
