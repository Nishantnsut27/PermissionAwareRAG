import { describe, expect, it, vi } from 'vitest';
import { ask, streamEvents, validateEvent } from '../src/api';

const done = { type: 'done', answer: 'Canonical answer', outcome: 'answer', completed: true, sources: [], stats: {}, effective_sellers: ['S001'] };
function bytes(chunks: string[]) {
  return new ReadableStream<Uint8Array>({ start(controller) {
    for (const chunk of chunks) controller.enqueue(new TextEncoder().encode(chunk)); controller.close();
  } });
}
async function events(chunks: string[]) { const result = []; for await (const event of streamEvents(bytes(chunks))) result.push(event); return result; }
describe('POST SSE transport', () => {
  it.each([
    { stats: [] }, { stats: { candidates: {} } }, { stats: { selected: -1 } },
    { stats: { latency_ms: Infinity } }, { stats: { model: [] } }, { stats: { reranked: 'true' } },
    { sources: [null] }, { sources: [{ pages: [{}] }] }, { sources: [{ label: {} }] },
    { sources: [{ document_id: [] }] }, { sources: [{ relevance: NaN }] },
    { effective_sellers: [{}] }, { request_id: {} },
  ])('rejects malformed metadata before accepting an answer: %j', async invalid => {
    expect(() => validateEvent({ ...done, ...invalid })).toThrow('invalid final answer');
  });
  it('accepts valid partial answers and nullable source relevance', () => {
    expect(validateEvent({ ...done, outcome: 'partial', completed: false, sources: [{ relevance: null }] }).type).toBe('done');
  });
  it('does not retry generation on invalid metadata', async () => {
    const fetcher = vi.spyOn(globalThis, 'fetch').mockResolvedValue(new Response(bytes([
      `data: ${JSON.stringify({ ...done, stats: { selected: {} } })}\n\n`,
    ]), { headers: { 'Content-Type': 'text/event-stream' } }));
    await expect(ask('s', 'q', [], '', new AbortController().signal, vi.fn())).rejects.toThrow('invalid final answer');
    expect(fetcher).toHaveBeenCalledTimes(1); fetcher.mockRestore();
  });
  it('parses fragmented frames, CRLF and multibyte characters', async () => {
    const body = new TextEncoder().encode(`:heartbeat\r\n\r\ndata: {"type":"delta","text":"café"}\r\n\r\ndata: ${JSON.stringify(done)}\r\n\r\n`);
    const stream = new ReadableStream<Uint8Array>({ start(controller) {
      for (const byte of body) controller.enqueue(new Uint8Array([byte])); controller.close();
    } });
    const result = []; for await (const event of streamEvents(stream)) result.push(event);
    expect(result).toEqual([{ type: 'delta', text: 'café' }, done]);
  });
  it.each([
    ['malformed JSON', ['data: {broken}\n\n']],
    ['missing terminal event', ['data: {"type":"delta","text":"unfinished"}\n\n']],
    ['invalid event type', ['data: {"type":"unexpected"}\n\n']],
    ['invalid terminal outcome', [`data: ${JSON.stringify({ ...done, outcome: 'complete' })}\n\n`]],
    ['invalid completion flag', [`data: ${JSON.stringify({ ...done, outcome: 'partial' })}\n\n`]],
  ])('rejects %s', async (_name, chunks) => { await expect(events(chunks)).rejects.toThrow(); });
  it('returns only the canonical answer, without client identity fields or generation retries', async () => {
    const fetcher = vi.spyOn(globalThis, 'fetch').mockResolvedValue(new Response(bytes([
      'data: {"type":"delta","text":"unverified draft"}\n\n', `data: ${JSON.stringify(done)}\n\n`,
    ]), { headers: { 'Content-Type': 'text/event-stream' } }));
    const progress = vi.fn();
    const result = await ask('session', 'question', [], 'S001', new AbortController().signal, progress);
    expect(result.answer).toBe('Canonical answer');
    expect(fetcher).toHaveBeenCalledTimes(1);
    const init = fetcher.mock.calls[0][1]!;
    expect(init.method).toBe('POST');
    expect(init.headers).toMatchObject({ Authorization: 'Bearer session' });
    expect(JSON.parse(init.body as string)).toEqual({ query: 'question', seller_filter: ['S001'] });
    expect(progress).toHaveBeenCalledWith('delta'); fetcher.mockRestore();
  });
  it('reports provider errors and rate limits without replaying requests', async () => {
    const fetcher = vi.spyOn(globalThis, 'fetch').mockResolvedValue(new Response(JSON.stringify({ message: 'Budget exhausted' }), { status: 429 }));
    await expect(ask('s', 'q', [], '', new AbortController().signal, vi.fn())).rejects.toMatchObject({ status: 429 });
    expect(fetcher).toHaveBeenCalledTimes(1); fetcher.mockRestore();
    await expect(events(['data: {"type":"error","message":"Provider failed"}\n\n'])).resolves.toEqual([{ type: 'error', message: 'Provider failed' }]);
  });
});
