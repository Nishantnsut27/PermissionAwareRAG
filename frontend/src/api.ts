import { isAnswer, isRecord, isUser, type Answer, type StreamEvent, type UserProfile } from './types';
import { parseEvaluation, type EvaluationSummary } from './evaluationData';

export const API_BASE_URL = (import.meta.env.VITE_API_BASE_URL || 'http://127.0.0.1:8000').replace(/\/+$/, '');
export class ApiError extends Error {
  constructor(message: string, public status?: number) { super(message); this.name = 'ApiError'; }
}
async function request(path: string, init: RequestInit = {}, timeout = 15000): Promise<Response> {
  try {
    return await fetch(`${API_BASE_URL}${path}`, {
      ...init, credentials: 'omit', cache: 'no-store',
      signal: init.signal ? AbortSignal.any([init.signal, AbortSignal.timeout(timeout)]) : AbortSignal.timeout(timeout),
    });
  } catch (error) {
    if (init.signal?.aborted) throw error;
    throw new ApiError('The service is starting or unavailable. Please try again shortly.');
  }
}
async function payload(response: Response): Promise<Record<string, unknown>> {
  let body: Record<string, unknown>;
  try { body = await response.json(); }
  catch { throw new ApiError('The service is starting or unavailable. Please try again shortly.', response.status); }
  if (!body || typeof body !== 'object' || Array.isArray(body)) throw new ApiError('The service returned an invalid response.');
  if (!response.ok) throw new ApiError(typeof body.message === 'string' ? body.message : 'The request failed.', response.status);
  return body;
}
export async function listUsers(signal?: AbortSignal): Promise<UserProfile[]> {
  const body = await payload(await request('/demo/users', { signal }));
  if (!Array.isArray(body.users) || !body.users.every(isUser) || !body.users.length) throw new ApiError('The demo account directory is unavailable.');
  return body.users;
}
export async function login(user_id: string, password: string) {
  const body = await payload(await request('/demo/login', {
    method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ user_id, password }),
  }));
  if (typeof body.access_token !== 'string' || !body.access_token || !isUser(body.user)
      || typeof body.expires_in !== 'number' || body.expires_in <= 0) throw new ApiError('The login response was invalid.');
  return { token: body.access_token, user: body.user, expiresAt: Date.now() + Math.min(body.expires_in, 3600) * 1000 };
}
export async function me(token: string, signal?: AbortSignal): Promise<UserProfile> {
  const body = await payload(await request('/demo/me', { signal, headers: { Authorization: `Bearer ${token}` } }));
  if (!isUser(body.user)) throw new ApiError('The restored session was invalid.');
  return body.user;
}
export async function ready(signal?: AbortSignal) {
  const response = await request('/ready', { signal }, 10000);
  const body = await response.json();
  return { ready: response.ok && body.status === 'ready', problems: Array.isArray(body.problems) ? body.problems : [] };
}
export async function evaluation(report?: string, signal?: AbortSignal): Promise<EvaluationSummary> {
  return parseEvaluation(await payload(await request(`/evaluation${report ? `?report=${encodeURIComponent(report)}` : ''}`, { signal })));
}
export function validateEvent(value: unknown): StreamEvent {
  if (!isRecord(value)) throw new ApiError('The answer stream contained an invalid event.');
  const event = value as StreamEvent;
  if (!['meta', 'delta', 'done', 'error'].includes(event.type)) throw new ApiError('The answer stream contained an invalid event.');
  if (event.type === 'delta' && typeof event.text !== 'string') throw new ApiError('The answer stream contained an invalid delta.');
  if (event.type === 'done' && !isAnswer(value)) throw new ApiError('The service returned an invalid final answer.');
  if (event.type === 'error' && event.message !== undefined && typeof event.message !== 'string') throw new ApiError('The answer stream contained an invalid error.');
  return event;
}
// SSE is a framing protocol: a network chunk may split a JSON value or UTF-8 character.
export async function* streamEvents(stream: ReadableStream<Uint8Array>): AsyncGenerator<StreamEvent> {
  const reader = stream.getReader();
  const decoder = new TextDecoder();
  let lineBuffer = '', data: string[] = [], terminal = false;
  function frame(): StreamEvent | undefined {
    if (!data.length) return;
    let value: unknown;
    try { value = JSON.parse(data.join('\n')); } catch { throw new ApiError('The answer stream contained malformed data.'); }
    data = [];
    return validateEvent(value);
  }
  try {
    while (true) {
      const { value, done } = await reader.read();
      lineBuffer += decoder.decode(value, { stream: !done });
      if (lineBuffer.length > 2_000_000) throw new ApiError('The answer stream exceeded its size limit.');
      let newline: number;
      while ((newline = lineBuffer.indexOf('\n')) !== -1) {
        const line = lineBuffer.slice(0, newline).replace(/\r$/, '');
        lineBuffer = lineBuffer.slice(newline + 1);
        if (line.startsWith('data:')) data.push(line.slice(5).replace(/^ /, ''));
        else if (!line) {
          const event = frame();
          if (event) {
            yield event;
            if (event.type === 'done' || event.type === 'error') { terminal = true; return; }
          }
        }
      }
      if (done) break;
    }
    if (!terminal) throw new ApiError('The connection ended before the answer completed. Please try again.');
  } finally {
    await reader.cancel().catch(() => {});
    reader.releaseLock();
  }
}
export async function ask(token: string, query: string, conversation: { role: string; content: string }[],
  seller: string, signal: AbortSignal, onProgress: (type: 'meta' | 'delta') => void): Promise<Answer> {
  const response = await request('/demo/query/stream', {
    method: 'POST', signal,
    headers: { Authorization: `Bearer ${token}`, 'Content-Type': 'application/json', Accept: 'text/event-stream' },
    body: JSON.stringify({ query, ...(conversation.length ? { conversation } : {}), ...(seller ? { seller_filter: [seller] } : {}) }),
  }, 110000);
  if (!response.ok) await payload(response);
  if (!response.headers.get('Content-Type')?.startsWith('text/event-stream') || !response.body) throw new ApiError('The service did not return an answer stream.');
  try {
    for await (const event of streamEvents(response.body)) {
      if (event.type === 'meta' || event.type === 'delta') onProgress(event.type);
      else if (event.type === 'error') throw new ApiError(event.message || 'The answer could not be completed.');
      else if (event.type === 'done') return event;
    }
  } catch (error) {
    if (error instanceof ApiError || signal.aborted) throw error;
    throw new ApiError('The connection ended before the answer completed. Please try again.');
  }
  throw new ApiError('The connection ended before the answer completed. Please try again.');
}
