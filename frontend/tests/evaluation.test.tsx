import { act, fireEvent, render, screen, waitFor } from '@testing-library/react';
import { QueryClient, QueryClientProvider, focusManager } from '@tanstack/react-query';
import { MemoryRouter, useLocation } from 'react-router-dom';
import { afterEach, describe, expect, it, vi } from 'vitest';
import Evaluation, { metricText } from '../src/Evaluation';
import { parseEvaluation } from '../src/evaluationData';
import * as api from '../src/api';

const stored = { has_results: true, run_id: 'run-new', report_id: 'runs/new.json', status: 'Incomplete' as const,
  selected_cases: 90, total_cases: 3, dataset_version: '1.1', available_reports: [
    { report_id: 'runs/new.json', status: 'Incomplete' as const, selected_cases: 90, dataset_version: '1.1' },
    { report_id: 'runs/old.json', status: 'Legacy' as const, selected_cases: 60, dataset_version: '1.0' },
  ] };
function RouteLabel() { return <output data-testid="route">{useLocation().search}</output>; }
function ui(route = '/evaluation') {
  const client = new QueryClient({ defaultOptions: { queries: { retry: false, gcTime: 0, refetchOnWindowFocus: false } } });
  return render(<QueryClientProvider client={client}><MemoryRouter initialEntries={[route]}><Evaluation /><RouteLabel /></MemoryRouter></QueryClientProvider>);
}
afterEach(() => { focusManager.setFocused(undefined); vi.useRealTimers(); vi.restoreAllMocks(); });
describe('saved evaluation dashboard', () => {
  it('keeps historical route selection until Latest evaluation is selected', async () => {
    const read = vi.spyOn(api, 'evaluation').mockImplementation(async report => ({ ...stored, run_id: report === 'runs/old.json' ? 'run-old' : 'run-new' }));
    ui('/evaluation?report=runs%2Fold.json');
    await screen.findByText('run-old');
    expect(read.mock.calls[0][0]).toBe('runs/old.json');
    fireEvent.click(screen.getByText('Refresh saved results'));
    await waitFor(() => expect(read).toHaveBeenCalledTimes(2));
    expect(read.mock.calls[1][0]).toBe('runs/old.json');
    fireEvent.change(screen.getByLabelText('Saved evaluation run'), { target: { value: '' } });
    await screen.findByText('run-new');
    expect(screen.getByTestId('route')).toHaveTextContent('');
    expect(read.mock.calls.at(-1)?.[0]).toBeUndefined();
    expect(screen.queryByText(/operator account/)).not.toBeInTheDocument();
  });
  it('polls latest every 30 seconds while visible and never polls historical runs', async () => {
    vi.useFakeTimers(); focusManager.setFocused(true);
    const read = vi.spyOn(api, 'evaluation').mockResolvedValue(stored);
    await act(async () => { ui(); });
    await act(async () => { await vi.advanceTimersByTimeAsync(30001); });
    expect(read).toHaveBeenCalledTimes(2);
    focusManager.setFocused(false);
    await act(async () => { await vi.advanceTimersByTimeAsync(30001); });
    expect(read).toHaveBeenCalledTimes(2);
    focusManager.setFocused(true);
    await act(async () => { fireEvent.change(screen.getByLabelText('Saved evaluation run'), { target: { value: 'runs/old.json' } }); });
    const before = read.mock.calls.length;
    await act(async () => { await vi.advanceTimersByTimeAsync(60001); });
    expect(read).toHaveBeenCalledTimes(before);
  });
  it('validates metrics and strips private fields before summary download', () => {
    const m = { value: .8, measured: 4, applicable: 7, total: 9 };
    const parsed = parseEvaluation({ ...stored, results: [{ answer: 'secret' }], measurements: { quality: { groundedness: m }, private: 'secret' } });
    expect(JSON.stringify(parsed)).not.toContain('secret');
    expect(parsed.measurements?.quality.groundedness).toEqual(m);
    expect(() => parseEvaluation({ has_results: 'yes' })).toThrow();
    expect(() => parseEvaluation({ has_results: true, measurements: { quality: { groundedness: { ...m, measured: 12 } } } })).toThrow();
    expect(metricText({ ...m, value: null, measured: 0 })).toBe('Not measured');
    expect(metricText({ ...m, value: null, measured: 0, applicable: 0 })).toBe('Not applicable');
    expect(parseEvaluation({ has_results: true }).measurements).toBeUndefined();
  });
});
