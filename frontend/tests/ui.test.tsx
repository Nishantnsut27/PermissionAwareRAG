import { act, fireEvent, render, screen, waitFor } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { QueryClient, QueryClientProvider } from '@tanstack/react-query';
import { MemoryRouter } from 'react-router-dom';
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import App from '../src/App';
import AnswerContent from '../src/AnswerContent';
import { STORAGE_KEY } from '../src/state';
import * as api from '../src/api';

const user = { user_id: 'U-001', name: 'Aditya Verma', role: 'Account Manager', department: 'Business', seller_scope: ['S001', 'S003'] };
const final = { answer: 'Canonical answer', outcome: 'answer' as const, completed: true, sources: [], stats: {}, effective_sellers: ['S001'] };
const session = () => ({ token: 'session', user, expiresAt: Date.now() + 3600000 });
function ui(route = '/login') {
  const client = new QueryClient({ defaultOptions: { queries: { retry: false, retryDelay: 0, gcTime: 0, refetchOnWindowFocus: false } } });
  const result = render(<QueryClientProvider client={client}><MemoryRouter initialEntries={[route]}><App /></MemoryRouter></QueryClientProvider>);
  return { ...result, client };
}
beforeEach(() => {
  vi.spyOn(api, 'listUsers').mockResolvedValue([user]); vi.spyOn(api, 'me').mockResolvedValue(user);
  vi.spyOn(api, 'login').mockImplementation(async () => session());
  vi.spyOn(api, 'ready').mockResolvedValue({ ready: true, problems: [] });
  vi.spyOn(api, 'evaluation').mockResolvedValue({ has_results: true, run_id: 'saved', total_cases: 4 });
  vi.spyOn(api, 'ask').mockResolvedValue(final);
});
afterEach(() => vi.restoreAllMocks());
async function signIn() {
  const interaction = userEvent.setup(); await screen.findByRole('option', { name: /Aditya Verma/ });
  await interaction.type(screen.getByLabelText('Demo password'), 'aditya@login');
  await interaction.click(screen.getByRole('button', { name: 'Continue' }));
  await screen.findByRole('heading', { name: 'How can I help, Aditya?' });
  return interaction;
}
describe('workspace behavior', () => {
  it('renders a login shell immediately while the directory is unavailable', async () => {
    vi.mocked(api.listUsers).mockImplementation(() => new Promise(() => {})); ui();
    expect(screen.getByRole('heading', { name: 'Explore a sample account' })).toBeVisible();
    expect(screen.getByRole('button', { name: 'Continue' })).toBeDisabled();
    expect(api.ready).not.toHaveBeenCalled(); expect(api.evaluation).not.toHaveBeenCalled();
  });
  it('reuses the directory across theme changes, and diagnostics/evaluations run only on demand', async () => {
    ui(); const interaction = await signIn();
    await interaction.click(screen.getByRole('switch', { name: 'Dark theme' }));
    await interaction.selectOptions(screen.getByLabelText('Seller filter'), 'S001');
    expect(api.listUsers).toHaveBeenCalledTimes(1); expect(api.ready).not.toHaveBeenCalled(); expect(api.evaluation).not.toHaveBeenCalled();
    await interaction.click(screen.getByRole('button', { name: 'Check service' }));
    await screen.findByText('Backend and knowledge index ready'); expect(api.ready).toHaveBeenCalledTimes(1);
    await interaction.click(screen.getByRole('button', { name: 'Evaluation overview' }));
    await screen.findByText('saved'); expect(api.evaluation).toHaveBeenCalledTimes(1);
    await interaction.click(screen.getByRole('button', { name: 'Refresh saved results' }));
    await waitFor(() => expect(api.evaluation).toHaveBeenCalledTimes(2));
  });
  it('renders canonical answers and clears the account when switching users', async () => {
    ui(); const interaction = await signIn();
    await interaction.type(screen.getByLabelText('Ask a question'), 'question');
    await interaction.click(screen.getByRole('button', { name: 'Send question' }));
    await screen.findByText('Canonical answer');
    expect(api.ask).toHaveBeenCalledTimes(1);
    await interaction.click(screen.getByRole('button', { name: 'New User' }));
    await screen.findByRole('heading', { name: 'Explore a sample account' });
    expect(screen.queryByText('Canonical answer')).not.toBeInTheDocument();
    expect(JSON.parse(sessionStorage.getItem(STORAGE_KEY)!).session).toBeNull();
  });
  it('shows login errors without retrying credentials', async () => {
    vi.mocked(api.login).mockRejectedValue(new api.ApiError('Incorrect password', 401)); ui();
    const interaction = userEvent.setup(); await screen.findByRole('option', { name: /Aditya/ });
    await interaction.type(screen.getByLabelText('Demo password'), 'wrong'); await interaction.click(screen.getByRole('button', { name: 'Continue' }));
    await screen.findByText('Incorrect password'); expect(api.login).toHaveBeenCalledTimes(1);
  });
  it('checks restored sessions and returns expired backend sessions to login', async () => {
    sessionStorage.setItem(STORAGE_KEY, JSON.stringify({ session: session(), messages: [], theme: 'light', memory: 6 }));
    vi.mocked(api.me).mockRejectedValue(new api.ApiError('Expired', 401)); ui('/chat');
    await screen.findByRole('heading', { name: 'Explore a sample account' });
    expect(api.me).toHaveBeenCalledTimes(1); expect(api.ask).not.toHaveBeenCalled();
  });
  it('aborts active requests when starting a new conversation', async () => {
    let active: AbortSignal | undefined;
    vi.mocked(api.ask).mockImplementation((_token, _question, _history, _seller, signal) => { active = signal; return new Promise(() => {}); });
    ui(); const interaction = await signIn();
    await interaction.click(screen.getByRole('button', { name: /What caused the settlement/ }));
    await screen.findByText('Working on your request…');
    await interaction.click(screen.getByRole('button', { name: 'New conversation' }));
    expect(active?.aborted).toBe(true); expect(api.ask).toHaveBeenCalledTimes(1);
    expect(screen.queryByText('Working on your request…')).not.toBeInTheDocument();
  });
  it('clears an unsent draft on expiry before another login', async () => {
    vi.useFakeTimers();
    try {
      sessionStorage.setItem(STORAGE_KEY, JSON.stringify({ session: { ...session(), expiresAt: Date.now() + 60000 }, messages: [], theme: 'light', memory: 6 }));
      await act(async () => { ui('/chat'); });
      fireEvent.change(screen.getByLabelText('Ask a question'), { target: { value: 'private unsent draft' } });
      await act(async () => { await vi.advanceTimersByTimeAsync(60001); });
      expect(screen.getByRole('heading', { name: 'Explore a sample account' })).toBeVisible();
      fireEvent.change(screen.getByLabelText('Demo password'), { target: { value: 'aditya@login' } });
      await act(async () => { fireEvent.submit(screen.getByRole('button', { name: 'Continue' }).closest('form')!); });
      expect(screen.getByLabelText('Ask a question')).toHaveValue('');
      expect(api.ask).not.toHaveBeenCalled();
    } finally { vi.useRealTimers(); }
  });
  it('stops an answer without accepting late results or replaying generation', async () => {
    let active: AbortSignal | undefined;
    let finish: ((answer: typeof final) => void) | undefined;
    vi.mocked(api.ask).mockImplementation((_token, _question, _history, _seller, signal) => {
      active = signal;
      return new Promise(resolve => { finish = resolve; });
    });
    ui(); const interaction = await signIn();
    await interaction.type(screen.getByLabelText('Ask a question'), 'question');
    await interaction.click(screen.getByRole('button', { name: 'Send question' }));
    await interaction.click(screen.getByRole('button', { name: 'Stop answer' }));
    expect(active?.aborted).toBe(true);
    await act(async () => { finish?.(final); });
    expect(screen.queryByText('Canonical answer')).not.toBeInTheDocument();
    expect(screen.getByText('You stopped this answer. You can ask again when ready.')).toBeVisible();
    expect(screen.getByLabelText('Ask a question')).toBeEnabled();
    expect(api.ask).toHaveBeenCalledTimes(1);
  });
  it('renders hostile metadata as text and blocks unsafe Markdown', () => {
    render(<AnswerContent text={'[click](javascript:alert(1)) <script>alert(2)</script>'} sources={[
      { label: '<img src=x onerror=alert(1)>', document_id: '<svg onload=alert(2)>', pages: ['<script>bad</script>'], relevance: NaN },
    ]} />);
    expect(screen.getByText('<img src=x onerror=alert(1)>')).toBeInTheDocument();
    expect(document.querySelector('script,svg,img')).toBeNull();
    expect(document.querySelector('a')?.getAttribute('href')).not.toMatch(/javascript:/);
    expect(screen.queryByText(/Relevance: NaN/)).not.toBeInTheDocument();
  });
});
