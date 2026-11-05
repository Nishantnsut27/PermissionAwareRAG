import { useState } from 'react';
import Markdown from 'react-markdown';
import remarkGfm from 'remark-gfm';
import type { Source } from './types';

export function Sources({ sources }: { sources: Source[] }) {
  const rows = sources.filter(source => source && typeof source === 'object');
  if (!rows.length) return null;
  return <details className="sources"><summary>▤ Sources ({rows.length})</summary>{rows.map((source, index) =>
    <div className="source-item" key={index}>
      <strong>{String(source.label || 'Untitled source')}</strong>
      <div className="caption">Document ID: {String(source.document_id || 'unknown')}
        {source.seller_id ? ` · Seller: ${String(source.seller_id)}` : ''}
        {Array.isArray(source.pages) && source.pages.length ? ` · ${source.pages.length === 1 ? 'Page' : 'Pages'}: ${source.pages.map(String).join(', ')}` : ''}
        {typeof source.relevance === 'number' && Number.isFinite(source.relevance) ? ` · Relevance: ${source.relevance.toFixed(2)}` : ''}
      </div>
    </div>)}</details>;
}
export default function AnswerContent({ text, sources = [], copy = false }: { text: string; sources?: Source[]; copy?: boolean }) {
  const [copied, setCopied] = useState('Copy answer');
  async function copyAnswer() {
    try { await navigator.clipboard.writeText(text); setCopied('Copied'); }
    catch { setCopied('Copy unavailable — select the answer text'); }
    setTimeout(() => setCopied('Copy answer'), 2500);
  }
  return <>
    <div className="markdown"><Markdown remarkPlugins={[remarkGfm]} skipHtml
      urlTransform={url => /^https?:\/\//i.test(url) ? url : ''}
      components={{ a: props => <a {...props} target="_blank" rel="noopener noreferrer" /> }}>
      {text}
    </Markdown></div>
    {copy && <button className="copy-button" onClick={copyAnswer}>{copied}</button>}
    <Sources sources={sources} />
  </>;
}
