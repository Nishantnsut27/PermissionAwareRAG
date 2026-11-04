export interface UserProfile {
  user_id: string; name: string; role: string; department: string; seller_scope: string[];
}
export type Outcome = 'answer' | 'partial' | 'refusal' | 'error' | 'interrupted' | 'pending';
export interface Source {
  label?: string; document_id?: string; seller_id?: string; pages?: unknown[]; relevance?: number;
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
export function isUser(value: unknown): value is UserProfile {
  if (!value || typeof value !== 'object') return false;
  const user = value as UserProfile;
  return ['user_id', 'name', 'role', 'department'].every(key => typeof user[key as keyof UserProfile] === 'string')
    && Array.isArray(user.seller_scope) && user.seller_scope.every(id => typeof id === 'string');
}
