import { isUser, type Message, type Session, type UserProfile } from './types';

export const STORAGE_KEY = 'knowledge-workspace.v1';
export interface State {
  session: Session | null; verified: boolean; messages: Message[]; seller: string;
  memory: number; theme: 'light' | 'dark';
}
export const initialState: State = { session: null, verified: false, messages: [], seller: '', memory: 6, theme: 'light' };
export type Action =
  | { type: 'login'; session: Session } | { type: 'verified'; user: UserProfile }
  | { type: 'logout' } | { type: 'reset' } | { type: 'theme' }
  | { type: 'seller'; seller: string } | { type: 'memory'; memory: number }
  | { type: 'start'; question: string; id: string } | { type: 'finish'; message: Message };
export function reducer(state: State, action: Action): State {
  switch (action.type) {
    case 'login': return { ...state, session: action.session, verified: true, messages: [], seller: '' };
    case 'verified': return state.session ? { ...state, verified: true, session: { ...state.session, user: action.user },
      seller: action.user.seller_scope.includes(state.seller) ? state.seller : '' } : state;
    case 'logout': return { ...initialState, theme: state.theme, memory: state.memory };
    case 'reset': return { ...state, messages: [] };
    case 'theme': return { ...state, theme: state.theme === 'light' ? 'dark' : 'light' };
    case 'memory': return { ...state, memory: Math.max(2, Math.min(6, action.memory)) };
    case 'seller': return { ...state, seller: state.session?.user.seller_scope.includes(action.seller) ? action.seller : '' };
    case 'start': return { ...state, messages: [...state.messages,
      { id: `${action.id}-user`, role: 'user', content: action.question },
      { id: action.id, role: 'assistant', content: '', outcome: 'pending' }] };
    case 'finish': return { ...state, messages: state.messages.map(row => row.id === action.message.id ? action.message : row) };
  }
}
export function history(state: State) {
  return state.messages.filter(row => !row.outcome || row.outcome === 'answer')
    .slice(-state.memory).map(({ role, content }) => ({ role, content }));
}
export function loadState(): State {
  try {
    const raw = sessionStorage.getItem(STORAGE_KEY);
    if (!raw) return { ...initialState };
    const saved = JSON.parse(raw) as State;
    const theme = saved.theme === 'dark' ? 'dark' : 'light';
    const memory = Number.isInteger(saved.memory) ? Math.max(2, Math.min(6, saved.memory)) : 6;
    if (!saved.session || typeof saved.session.token !== 'string' || !saved.session.token
        || !Number.isFinite(saved.session.expiresAt) || saved.session.expiresAt <= Date.now() || !isUser(saved.session.user)) {
      return { ...initialState, theme, memory };
    }
    const messages = Array.isArray(saved.messages) ? saved.messages.filter(row => row && typeof row.id === 'string'
      && ['user', 'assistant'].includes(row.role) && typeof row.content === 'string'
      && (!row.outcome || ['answer', 'partial', 'refusal', 'error', 'interrupted', 'pending'].includes(row.outcome))) : [];
    return { session: saved.session, verified: false, theme, memory,
      seller: saved.session.user.seller_scope.includes(saved.seller) ? saved.seller : '',
      messages: messages.map(row => row.outcome === 'pending' ? { ...row, outcome: 'interrupted', content: 'The page refreshed before this answer completed. Please ask again.' } : row) };
  } catch { return { ...initialState }; }
}
export function saveState(state: State) {
  try { sessionStorage.setItem(STORAGE_KEY, JSON.stringify({ ...state, verified: false })); }
  catch { /* Storage can be unavailable; the workspace still works in memory. */ }
}
