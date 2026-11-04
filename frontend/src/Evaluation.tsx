import { useState } from 'react';
import { useQuery } from '@tanstack/react-query';
import { evaluation } from './api';

export default function Evaluation() {
  const [report, setReport] = useState('');
  const query = useQuery({ queryKey: ['evaluation', report], queryFn: ({ signal }) => evaluation(report || undefined, signal), staleTime: 60000, retry: 2 });
  const data = query.data;
  return <section className="evaluation" aria-labelledby="evaluation-title">
    <div className="hero"><div className="eyebrow">Saved results · No automatic runs</div>
      <h1 id="evaluation-title">Evaluation overview</h1><p>Every case is executed through the same permission-aware pipeline as the chat interface.</p></div>
    <div className="panel">
      {!!data?.available_reports?.length && <label>Saved evaluation run<select value={report} onChange={event => setReport(event.target.value)}>
        <option value="">Latest saved report</option>{data.available_reports.map(row => <option key={row.report_id} value={row.report_id}>{row.report_id}{row.generated_at ? ` · ${row.generated_at}` : ''}</option>)}
      </select></label>}
      <button className="primary" onClick={() => void query.refetch()} disabled={query.isFetching}>Refresh saved results</button>
      {query.isPending && <p role="status">Loading saved results…</p>}
      {query.error && <div className="notice error" role="alert">{query.error.message}</div>}
      {data && (data.has_results ? <><p>Saved report <strong>{data.run_id || data.report_id || '(checkpoint)'}</strong> contains <strong>{data.total_cases ?? 0}</strong> recorded cases.</p>
        {data.generated_at && <p className="caption">Generated {data.generated_at}</p>}
        <div className="notice">Detailed metrics, answers and evidence require an authenticated operator account.</div></>
        : <p>No evaluation results are available.</p>)}
      <p className="caption">Refreshing this page only reads saved results. Evaluation runs are started explicitly from the evaluation CLI.</p>
    </div>
  </section>;
}
