export interface UserProfile {
  user_id: string; name: string; role: string; department: string; seller_scope: string[];
}
export type Outcome = 'answer' | 'partial' | 'refusal' | 'error' | 'interrupted' | 'pending';
export interface Source {
  label?: string; document_id?: string; seller_id?: string; pages?: (number | string)[]; relevance?: number | null;
}
export interface Stats {
  candidates?: number; authorized?: number; selected?: number; withheld?: number;
  latency_ms?: number; reranked?: boolean; model?: string;
}
export interface Answer {
  answer: string; outcome: 'answer' | 'partial' | 'refusal'; completed: boolean;
  sources: Source[]; stats: Stats; effective_sellers: string[]; request_id?: string;
}
export type StreamEvent =
  | ({ type: 'meta'; outcome?: string } & Partial<Answer>)
  | { type: 'delta'; text: string }
  | ({ type: 'done' } & Answer)
  | { type: 'error'; message?: string };
export interface Message {
  id: string; role: 'user' | 'assistant'; content: string; outcome?: Outcome;
  sources?: Source[]; stats?: Stats; effective_sellers?: string[];
}
export interface Session { token: string; expiresAt: number; user: UserProfile }
export function isRecord(value: unknown): value is Record<string, unknown> {
  return !!value && typeof value === 'object' && !Array.isArray(value);
}
const optional = (value: Record<string, unknown>, key: string, check: (x: unknown) => boolean) =>
  value[key] === undefined || check(value[key]);
const text = (x: unknown) => typeof x === 'string';
const finite = (x: unknown): x is number => typeof x === 'number' && Number.isFinite(x);
export const isStrings = (x: unknown): x is string[] => Array.isArray(x) && x.every(text);
export function isStats(value: unknown): value is Stats {
  return isRecord(value)
    && ['candidates', 'authorized', 'selected', 'withheld'].every(key => optional(value, key, x => finite(x) && Number.isInteger(x) && x >= 0))
    && optional(value, 'latency_ms', x => finite(x) && x >= 0)
    && optional(value, 'reranked', x => typeof x === 'boolean') && optional(value, 'model', text);
}
export function isSource(value: unknown): value is Source {
  return isRecord(value) && ['label', 'document_id', 'seller_id', 'citation'].every(key => optional(value, key, text))
    && ['document_type', 'department'].every(key => optional(value, key, x => x === null || text(x)))
    && ['chunk_ids', 'evidence_ids'].every(key => optional(value, key, isStrings))
    && optional(value, 'relevance', x => x === null || finite(x))
    && optional(value, 'pages', x => Array.isArray(x) && x.every(p => typeof p === 'string' || (finite(p) && Number.isInteger(p) && p >= 1)));
}
export function isAnswer(value: unknown): value is Answer {
  return isRecord(value) && ['answer', 'partial', 'refusal'].includes(String(value.outcome))
    && typeof value.answer === 'string' && (value.outcome === 'refusal' || !!value.answer.trim())
    && value.completed === (value.outcome !== 'partial') && Array.isArray(value.sources) && value.sources.every(isSource)
    && isStats(value.stats) && isStrings(value.effective_sellers) && optional(value, 'request_id', text);
}
export function isMessage(value: unknown): value is Message {
  return isRecord(value) && typeof value.id === 'string' && ['user', 'assistant'].includes(String(value.role))
    && typeof value.content === 'string'
    && (value.role === 'user' ? value.outcome === undefined : ['answer', 'partial', 'refusal', 'error', 'interrupted', 'pending'].includes(String(value.outcome)))
    && optional(value, 'sources', x => Array.isArray(x) && x.every(isSource))
    && optional(value, 'stats', isStats) && optional(value, 'effective_sellers', isStrings);
}
export function isUser(value: unknown): value is UserProfile {
  if (!value || typeof value !== 'object') return false;
  const user = value as UserProfile;
  return ['user_id', 'name', 'role', 'department'].every(key => typeof user[key as keyof UserProfile] === 'string')
    && Array.isArray(user.seller_scope) && user.seller_scope.every(id => typeof id === 'string');
}
