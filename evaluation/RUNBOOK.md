# Run and resume the evaluation

To retry **ERROR cases (generation, judging, retrieval, or other errors)** and
**FAIL cases with two or fewer total attempts** in the existing full report:

```powershell
.\.venv\Scripts\python.exe scripts/run_evaluation.py --name baseline-v11-20261104 --phase full --retry-errors-and-low-attempt-failures
```

Append `--plan` for an offline preview. The first execution freezes the selected
IDs and grants each one additional attempt, including error cases at their old
cap. Repeating this command continues the remaining frozen batch; it does not
grant more attempts to cases already attempted in this batch. Provider errors
still pause execution. An interrupted in-flight attempt counts as an attempt.
Passing cases and FAIL cases above two attempts are not executed. A changed
parent answer still invalidates dependent scores so stale conversation results
are never presented as current; only selected cases can execute.

All 90 cases, original scores, prior and new attempt details, and the retry policy
remain together in `evaluation/runs/baseline-v11-20261104-full.json`. The original
response log is rebuilt with **all** attempts at
`.cache/evaluation-responses/run-20261104-101037-414954.jsonl`. This is an
evaluation rerun; it does not train or change model weights. The scheduler-only
checkpoint migration verifies the archived runner and all other evaluation
inputs; previous attempt fingerprints are preserved.

To retry the queued cases in the existing **E001–E090 full report** once, without revisiting the pilot:

```powershell
.\.venv\Scripts\python.exe scripts/run_evaluation.py --name baseline-v11-20261104 --phase full
```

This keeps `evaluation/runs/baseline-v11-20261104-full.json`, its original run ID, and `.cache/evaluation-responses/run-20261104-101037-414954.jsonl`. Passing cases and every previous attempt stay in that report. Each eligible queued case receives at most one new attempt. **E001, E010, E012, and E025 have one explicitly approved fifth attempt in this existing full report only. Every other case keeps the four-attempt cap; newly exhausted cases do not gain an extra attempt.** The exception does not apply to a new report or the pilot. A changed prerequisite answer invalidates its dependent follow-up as before. Append `--plan` to see the exact queue without provider calls or checkpoint writes.

Groq generation and judging share one pool of unique `GROQ_API_KEY_1` through `GROQ_API_KEY_20` values (plus the legacy `GROQ_API_KEY`). New requests rotate across usable keys. A 429 skips that key until its provider-supplied `Retry-After`, reset duration, or fallback cooldown expires; 401/403 disables it for the process. Transient network/5xx retries stay on the same key. Updating credentials takes effect when the command restarts.

The Groq **tokens per day (TPD)** limit can be shared by all keys in one organization. Five keys do not provide five daily allowances when they share that quota. Wait for the provider's reset before rerunning after all keys are exhausted. The runner saves its checkpoint and stops at another quota error, rather than repeatedly consuming case attempts. The reviewed rotation update has an exact fingerprint migration for these existing checkpoints; original attempt configurations and scores remain unchanged, and unrelated model/data/code changes still reject resume.

To run both phases:

From the repository root in PowerShell:

```powershell
.\.venv\Scripts\python.exe scripts/run_evaluation.py
```

The command runs a 16-case pilot (the 11-case category/security pilot plus
E061, E073, E079, E083, E087 and any prerequisites), then the full 90-case
dataset. The pilot and full baseline have separate reports. A completed pilot
with genuine quality failures may proceed; an API or judging error pauses it.

Press **Ctrl+C once** to pause. Run the identical command to resume. Within
each phase, fresh cases run first, then cases that failed or were interrupted
on an earlier invocation. Passing cases are kept. Each case gets at most one
attempt per invocation and **four total attempts per report** (the initial
attempt plus three retries). A new failure waits for the next explicit restart;
it is never automatically retried in that invocation. The terminal prints
`Why:` with failed thresholds, security findings, or API/judging errors, plus
the attempt count and whether another attempt is available.

Follow-up cases wait for their prerequisites. If a retried prerequisite
changes its answer, affected follow-ups need a new attempt too, within the
same cap. At the limit, failed or unresolved cases stay in the report. An
exhausted API/judging/dependency error leaves the report explicitly incomplete.
The command visits the pilot before the full run on every restart; a scored
pilot with quality failures can still advance to the full run.

Every attempt and its original scores remain in the report and response log.
Headline metrics describe the latest attempt; `first_attempt_summary` retains
the original baseline scores. Resume works at case boundaries, not halfway
through a provider request. An in-flight attempt is reserved in an atomic
checkpoint before provider calls, so closing the terminal still counts that
attempt toward the cap. Do not change code, dataset, models, thresholds, or
index between attempts. Mismatched checkpoints are rejected. The scheduler
upgrade accepts the existing paused checkpoint only after verifying the archived
source fingerprint and the exact reviewed new source hashes. The Jina request
policy and four-attempt limit migration is recorded in `policy_migrations`;
all earlier attempt configurations and scores remain intact. Other changes to
code, dataset, models, thresholds or index still prevent resuming.

Jina uses `JINA_API_KEY_1` and `JINA_API_KEY_2` in strict alternating allocation
order across embedding and reranking requests within each process. Restarting
the command starts the sequence at key 1. Numbered keys take precedence over
the legacy `JINA_API_KEY`; blank and duplicate keys are ignored. Updating only
credential values and restarting keeps checkpoint compatibility.

During evaluation, the first failed Jina request stops the case without an
automatic retry, key failover, or reranking fallback. The terminal names the
operation, key variable, and actual HTTP/network/response error and asks you
to check/change that key. Already-running requests may finish, but queued Jina
requests are blocked and the failed case is not generated or judged further.
The reserved case attempt still counts toward the four-attempt cap. Safe
`jina_calls` diagnostics record sequence, operation, key label, timing and
outcome; no credential values or authorization headers are stored. Ordinary
application requests retain their existing retry/fallback behavior.

- `evaluation/runs/baseline-v11-20261104-pilot.json`
- `evaluation/runs/baseline-v11-20261104-full.json`
- `.cache/evaluation-responses/<run-id>.jsonl`

The response log includes the actual question/history, answer, source context,
stage trace, expected facts, scores, raw judge requests/responses (including
malformed responses), errors, timestamps, and completed/interrupted attempts. It
contains no API keys and stays local. The dashboard and summary download
expose aggregate metrics only. Historical reports are preserved.

The full baseline requires 133.5 minutes of pacing alone, plus generation and
judging. The pilot requires 22.5 minutes of pacing. Current provider settings,
90-second spacing and zero automatic runner retries are retained. Quota,
provider, judging and index errors pause the run; inspect the report/log and
restart explicitly after fixing the cause.

For an offline preview, append `--plan`. For an intentional new baseline,
use a new name, for example `--name baseline-v11-next`. Never change the name
just to resume. Exit code 0 means completed without failed cases, 1 means
completed with genuine failures, 2 means incomplete/error, and 130 means paused.

Dataset v1.1 preserves E001-E060 except the documented E049 correction:
POL-IT-002 page 2 specifies security incident retention of 3 active plus 4
archive years (7 total). The correction and previous text are retained in
`fixtures/expectation_corrections.json`. New cases carry corpus excerpts and
hashes checked against the permission matrix. Original and expanded cohorts
are reported separately.

The earlier `pilot-20261104-v11.json` remains an incomplete historical attempt.
It used the previous judge context and cannot resume under the updated logging
and judging code. The command above uses a distinct checkpoint name.
