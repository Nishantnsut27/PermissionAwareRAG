import { describe, expect, it } from 'vitest';
import { history, initialState, loadState, reducer, saveState, STORAGE_KEY, type State } from '../src/state';
const user = { user_id: 'U-001', name: 'Aditya Verma', role: 'Account Manager', department: 'Business', seller_scope: ['S001', 'S003'] };
function signed(): State { return reducer(initialState, { type: 'login', session: { token: 's', user, expiresAt: Date.now() + 3600000 } }); }
describe('tab session and conversation', () => {
  it('drops restored answers with malformed metadata before rendering or using memory', () => {
    sessionStorage.setItem(STORAGE_KEY, JSON.stringify({ ...signed(), messages: [
      { id: '1', role: 'assistant', outcome: 'answer', content: 'bad answer', stats: { selected: {} } },
      { id: '2', role: 'assistant', outcome: 'answer', content: 'bad source', sources: [{ pages: [{}] }] },
      { id: '3', role: 'assistant', outcome: 'answer', content: 'valid', stats: { selected: 2 } },
    ] }));
    expect(loadState().messages.map(m => m.content)).toEqual(['valid']);
    expect(history(loadState()).map(m => m.content)).toEqual(['valid']);
  });
  it('restores completed messages but requires backend verification and never replays pending work', () => {
    let state = reducer(signed(), { type: 'start', id: '1', question: 'question' });
    state = reducer(state, { type: 'finish', message: { id: '1', role: 'assistant', content: 'Canonical', outcome: 'answer' } });
    state = reducer(state, { type: 'start', id: '2', question: 'unfinished' });
    saveState(state); const restored = loadState();
    expect(restored.verified).toBe(false);
    expect(restored.messages[1].content).toBe('Canonical');
    expect(restored.messages[3].outcome).toBe('interrupted');
    expect(history(restored).some(row => row.content.includes('page refreshed'))).toBe(false);
  });
  it('discards expired and malformed sessions', () => {
    const state = signed(); state.session!.expiresAt = Date.now() - 1;
    saveState(state); expect(loadState().session).toBeNull();
    sessionStorage.setItem(STORAGE_KEY, '{broken'); expect(loadState()).toEqual(initialState);
  });
  it('only permits a seller returned by the backend and rechecks restored scope', () => {
    expect(reducer(signed(), { type: 'seller', seller: 'S002' }).seller).toBe('');
    const state = reducer(signed(), { type: 'seller', seller: 'S001' });
    expect(reducer(state, { type: 'verified', user: { ...user, seller_scope: [] } }).seller).toBe('');
  });
  it('switching users clears transcript, pending work and selected seller', () => {
    const state = reducer(reducer(signed(), { type: 'seller', seller: 'S001' }), { type: 'start', id: '1', question: 'private' });
    const cleared = reducer(state, { type: 'logout' });
    expect(cleared.session).toBeNull(); expect(cleared.messages).toEqual([]); expect(cleared.seller).toBe('');
  });
  it('uses only eligible history, capped by the selected memory', () => {
    const state = signed(); state.memory = 2;
    state.messages = [
      { id: '1', role: 'user', content: 'first' }, { id: '2', role: 'assistant', content: 'answer', outcome: 'answer' },
      { id: '3', role: 'assistant', content: 'partial', outcome: 'partial' }, { id: '4', role: 'assistant', content: 'denied', outcome: 'refusal' },
      { id: '5', role: 'assistant', content: 'failed', outcome: 'error' }, { id: '6', role: 'user', content: 'follow-up' },
    ];
    expect(history(state)).toEqual([{ role: 'assistant', content: 'answer' }, { role: 'user', content: 'follow-up' }]);
  });
});
