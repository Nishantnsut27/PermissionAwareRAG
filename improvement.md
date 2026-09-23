# RAG System: Problems, Causes, Fixes, and an Honest Assessment

Date: 2026-09-23
Scope: permission-aware RAG for Nexora Commerce (60-case golden evaluation)
Latest run: `evaluation/runs/full-20260923-fixed.json` (47 PASS / 6 FAIL / 1 ERROR)

This document is deliberately blunt. It records what is wrong, why, how to fix it,
and whether the system is actually good. It is written for the next engineer, not
for a demo.

---

## 1. Executive assessment: is this RAG "the best"? No. It is a strong retriever with a security architecture that is not production-ready.

Split the verdict by subsystem, because a single number hides the truth:

| Subsystem | Verdict | Evidence |
| --- | --- | --- |
| Retrieval | Strong | recall 95.5%, MRR 0.853, precision efficiency 0.95 |
| Generation | Strong | groundedness 98.4%, relevancy 100%, correctness 92.6% |
| Access control (retrieval level) | Correct | unauthorized_retrieval_rate 0.0% - nothing forbidden is ever fetched |
| Access control (answer level) | **Not production-ready** | answer_leakage_rate 12.5% - the model discloses restricted data that lives inside authorized documents |

The headline pass rate is 78% (47/60). That is good for a prototype and misleading
for a security product. **The entire value proposition of this system is the word
"permission-aware", and that is exactly the axis where it fails.** A RAG that answers
well but leaks is worse than a mediocre RAG that never leaks, because the failure is
silent and high-consequence.

So: excellent plumbing, unfinished security model. Not the best. Fixable.

---

## 2. The single most important problem: cross-boundary data inside an authorized document

### 2.1 What happens

Both answer-level leaks in the latest run (E035, E048) trace to **one document:
`ORG-001`, the "Organization Reference".**

- E035 (expected: refuse). User asks about StrideOne Footwear (seller S004), which
  they are not scoped to. The forbidden document `S004-ACC-001` is correctly never
  retrieved. But the model answers `Account tier: Gold` anyway, citing ORG-001.
- E048 (expected: partial answer). User may read S001 support tickets but not the
  S001 account overview (`S001-ACC-001`, which is in `forbidden_sources`). That
  document is never retrieved either. Yet the model assembles a full "Account
  Overview" table (Seller ID, tier, onboarding date) - built from ORG-001.

### 2.2 The cause (verified, not guessed)

`ORG-001` is tagged `seller_id = "Organization-wide"` and `classification = INTERNAL`,
so the permission engine authorizes it for every user. But its **text contains
seller-specific data for all four sellers** - tiers, staff coverage, brand names
(confirmed: the document mentions S001-S004, Aurelia, GreenCart, StrideOne).

The access model is enforced at **document granularity**. ORG-001 is an **aggregate
document that crosses seller boundaries**. The seller-scope filter that correctly
blocks `S004-ACC-001` is structurally incapable of blocking S004 facts that are
embedded inside a document labelled "Organization-wide". The security boundary and
the data layout disagree.

This is not a tuning bug. It is a data-modelling gap: **a document is the unit of
authorization, but it is not the unit of confidentiality.**

### 2.3 Did the recall-first change cause this? Partly - it widened the blast radius.

- E035 was already failing before any change (`was_leak = True` in the baseline).
  Only ORG-001 is shown; breadth is irrelevant. **Pre-existing.**
- E048 regressed PASS -> FAIL. ORG-001 was ranked 4th of the 5 documents shown. At
  the old effective breadth (~3 docs) it would not have entered the context, so the
  model had nothing to build the overview from. **Recall-first exposed a latent
  vulnerability that was always there.**

Honest conclusion: widening the context did not create the hole, it made a
pre-existing hole fire more often. Narrowing breadth would hide it again without
fixing it. Do not treat reverting breadth as a fix.

### 2.4 How to fix it (in priority order)

1. **Structural, correct fix - sub-document authorization on aggregate documents.**
   Re-ingest ORG-001 (and any similar reference/aggregate doc) with section-aware
   chunking, and tag each chunk with the seller it actually describes rather than
   "Organization-wide". A paragraph about StrideOne becomes `seller_id = S004`; the
   genuinely org-wide sections (company overview, role definitions) stay
   Organization-wide. The existing Qdrant seller-scope filter then excludes
   out-of-scope rows for free, with no new enforcement code. This is the right fix
   because it makes the unit of authorization match the unit of confidentiality.

2. **Interim, defensive fix - scope-aware redaction at retrieval time.** Before
   context assembly, mask lines in any retrieved chunk that name a seller outside the
   caller's authorized scope. Cheaper, brittle (depends on recognising seller
   identifiers), but deployable today and independent of re-ingestion.

3. **Defense in depth - generation discipline.** Strengthen the system prompt: never
   assemble a profile/overview of an entity from reference material; when a request
   mixes authorized and unauthorized scope, answer only the authorized part and
   explicitly decline the rest. This alone is not sufficient (it trusts the model),
   but it should always be present behind 1 or 2.

Recommended: do 1 as the real fix and keep 3 permanently. Use 2 only if a fix is
needed before re-ingestion is possible.

---

## 3. The precision/recall tradeoff and a weak metric

### 3.1 What happens

Precision@K fell 51.0% -> 41.9% when we moved to recall-first. This looks like a
regression and mostly is not one.

### 3.2 The cause

`precision_at_k` divides by the number of documents placed in the context, and that
K is variable (0-5), not fixed. The cases need 1.7 expected documents on average, so
`Precision@K is bounded by 1.7 / documents_shown`. Showing more documents to raise
recall mechanically lowers precision even when every relevant document is found -
`precision_efficiency` is 0.95, meaning retrieval is near the mathematical ceiling
for the breadth shown.

So the metric conflates two different things: "did we retrieve the wrong documents"
(a real defect) and "did we show more documents than the question strictly needed"
(a breadth policy). Only the first is a quality problem.

### 3.3 How to fix it

1. **Report a fixed-cutoff precision alongside the context-size one.** Precision@5
   over the ranked candidate list (independent of how many documents the context
   budget admitted) is comparable across runs and configurations. Keep the current
   context-size precision for answer-faithfulness reasoning.
2. **Adopt nDCG@k** for retrieval. It rewards putting relevant documents early and
   handles variable relevant-set sizes far better than raw precision. MRR (already
   reported) covers first-hit rank but not the full ordering.
3. Keep `precision_ceiling` / `precision_efficiency` (added in this work) as the
   interpretation aid so a breadth choice is never misread as a retrieval failure.

To genuinely raise precision without losing recall you need a better *ranker*, not a
tighter budget - see section 5.

---

## 4. The correctness = 0.50 cluster (answer completeness)

### 4.1 What happens

Three of the six failures fail at exactly the same value: `correctness 0.50 below
0.60` - E007 (authorized_factual), E018 (multi_document), E049 (policy_governance).
A cluster at a round 0.50 is a signal, not a coincidence.

### 4.2 The likely cause

0.50 is half credit. These are multi-part questions ("what do policy X and runbook Y
*together* require", "explain A and B") where the model answers one part well and the
other thinly, so the judge awards half. Contributing factors:

- `GROQ_REASONING_EFFORT = low` and `max_tokens = 1200` bias toward concise answers
  that can drop a sub-question.
- The judge is a single LLM sample at temperature 0; on borderline answers it lands
  on a coarse 0.5 rather than a calibrated value. E018 flipped PASS -> FAIL between
  runs on judge variance alone, which confirms borderline instability.

### 4.3 How to fix it

1. Manually read the three answers against their reference answers before changing
   anything - confirm whether the model or the judge is at fault. This is one hour of
   work and decides the direction.
2. If the model under-answers: raise `max_tokens`, lift reasoning effort to medium
   for multi-part categories, and add a prompt instruction to address every clause of
   a compound question explicitly.
3. If the judge is noisy: sample the correctness judge 3x and take the median, or add
   an explicit per-sub-question rubric. The deterministic `fact_coverage` (88.8%)
   already provides a more stable secondary signal - consider gating on it too.

---

## 5. Retrieval quality issues that limit the ceiling

### 5.1 Sparse BM25 applies IDF twice

`indexing/src/sparse_search.py` uses the same `encode()` for documents and queries,
then Qdrant takes their dot product, so each term's contribution is scaled by
`idf^2`. This over-weights rare terms and distorts lexical scoring. Exact-identifier
queries are most affected - and note that E025 missed exactly `INC-2026-0519` and
`INC-2026-0312`, ID-style documents that lexical search should nail.

Fix: apply IDF once (standard BM25 weights the query side with raw term frequency, or
apply IDF only to the document vector). Re-fit the vocabulary and re-index. Add a unit
test asserting an exact ID query ranks its document first.

### 5.2 The reranker is the precision ceiling

Precision cannot beat the reranker's ordering. On E001 the account-overview chunk
out-scored the actual settlement documents - the reranker's relevance judgement, not
selection, is the limit. Options, cheapest first:

1. Feed the reranker a cleaner query (done - conversation-resolved) and full chunks
   (done - 4800 char window).
2. Try a stronger rerank model, or a cross-encoder fine-tuned on a few dozen
   domain pairs from the golden set.
3. Add a light lexical prior for exact identifiers so ID-bearing documents are not
   buried by semantically fluent but less relevant prose.

### 5.3 Reranker token budget vs Jina rate limit

Widening the window to 4800 chars pushed each rerank call to ~24k tokens; at
`candidate_pool = 20` this hits Jina's 100k tokens/minute limit and, before it was
fixed, silently degraded to unranked fusion order (corrupted 13/45 cases in one
tuning pass). Retry-with-backoff is now in place, but the underlying pressure remains.

Fix options: reduce `candidate_pool` to 12-15 (little recall loss, the pool is
already over-provisioned), batch reranking, or raise the plan limit. Keep the
degrade-visible instrumentation (`stats.selection.rerank_applied/attempts/error`) so
a throttled run can never be mistaken for a healthy one.

---

## 6. Evaluation harness weaknesses (why the numbers are hard to trust run-to-run)

### 6.1 The judge is a single point of failure and non-deterministic

- E054 did not fail on quality; it errored because all five Groq keys were
  rate-limited during the judge call (`--delay 12` was too aggressive). The judge has
  no retry/backoff of its own, unlike the reranker now does.
- The judge is one LLM sample; borderline cases (the 0.50 cluster, E018's flip) show
  it is not perfectly stable even at temperature 0.

Fix: give the judge the same retry-with-backoff treatment as the reranker; pace judge
calls independently; for gate-adjacent scores, sample 3x and take the median.

### 6.2 Unstable denominators make rates non-comparable

`answer_leakage_rate` moved 4.5% -> 12.5%, but the leakage-checked denominator also
moved (22 -> 16) because it depends on how many cases were answered vs refused. A rate
whose denominator shifts between runs cannot be compared directly. Report leaked-case
**counts** next to rates, and hold the denominator fixed to the case definition
(forbidden_sources present, or refuse expected) rather than the runtime outcome.

### 6.3 Reproducibility

The run fingerprint changes on any source edit, and the judge is non-deterministic, so
two runs are never bit-identical. This is acceptable but should be stated whenever a
delta is reported. Prefer deltas computed on the intersection of completed cases (as
was done for the before/after table here) rather than whole-run aggregates.

---

## 7. Smaller issues worth tracking

- **LLM over-generation.** The model emits elaborate markdown tables even when a
  sentence would do (E048's "Account Overview" table is the leak vector). Prompt it to
  match answer shape to the question and to avoid reconstructing entity profiles.
- **Chunk-to-page provenance is heuristic.** `ingestion/src/chunking.py` recovers
  character offsets with `str.find` + a cursor; pathological repeated text could
  misattribute page numbers in a citation. Low impact, but citations are a trust
  surface in a permission product.
- **`final_top_k = 10` in `RetrievalConfig` is dead code** in the live path; the
  answering layer passes explicit values. Remove it to avoid future confusion.
- **`retrieval_method` labelling is asymmetric** in RRF (dense-only hits keep
  `"dense"`); cosmetic but misleading in traces.

---

## 8. Why do these problems still appear, at a deeper level?

Four systemic reasons, not four unlucky cases:

1. **Authorization is modelled on documents; confidentiality lives in content.** As
   long as a single document can contain data from multiple trust domains (ORG-001),
   a document-level filter will leak. Every answer-level leak in this system is a
   symptom of that one mismatch. This is the root cause behind the root causes.

2. **The last line of defense is a language model.** Retrieval-level security is
   deterministic and provably correct here (unauthorized_retrieval = 0.0%). But once
   authorized-but-sensitive text is in the context, the only thing between it and the
   user is the model's judgement, which is probabilistic. A security guarantee cannot
   rest on an LLM's discretion. The context that reaches the model must already be
   safe.

3. **Metrics that move against each other were treated as one score.** Recall and
   precision trade off; groundedness and leakage trade off (more faithful use of a
   broad context can mean more disclosure). Optimising one number moved another. The
   fix is to measure them separately and decide the operating point deliberately -
   which is now done, but the tension is inherent and must stay visible.

4. **The judge defines truth and the judge is fallible.** Correctness, relevancy and
   leakage are all LLM verdicts. Judge noise (the 0.50 cluster, E018's flip) and judge
   availability (E054) show up directly as "system" failures. Some reported failures
   are measurement failures. Until the judge is hardened and calibrated, treat scores
   within ~0.1 of a gate as ties, not verdicts.

---

## 9. Are we producing "too many errors"?

No, but the errors are concentrated where they matter most. Of 13 non-passing slots:

- 1 is infrastructure (E054, rate-limited judge) - not a system defect.
- 3 are answer-completeness on compound questions (E007, E018, E049), partly judge
  noise.
- 1 is a hard cross-seller aggregation recall miss (E025), tied to the sparse-IDF and
  reranker-ceiling issues.
- 2 are security leaks (E035, E048), both the ORG-001 architectural gap.

The correctness/recall misses are ordinary RAG tuning work. The two leaks are the ones
that should block a "permission-aware" claim. Quantity is fine; one *class* of error
(cross-boundary leakage) is disqualifying for the stated purpose until fixed.

---

## 10. Prioritised roadmap

| Priority | Item | Effort | Payoff |
| --- | --- | --- | --- |
| P0 | Sub-document seller tagging for ORG-001 and aggregate docs (section 2.4.1) | Medium | Closes both answer-level leaks; makes "permission-aware" true |
| P0 | Permanent generation guardrail for mixed/again-scope requests (section 2.4.3) | Low | Defense in depth behind P0 |
| P1 | Judge retry/backoff + independent pacing (section 6.1) | Low | Removes false ERRORs like E054 |
| P1 | Manual review of the 0.50 correctness cluster, then prompt/token fix (section 4) | Low | Recovers ~2-3 cases |
| P1 | Fix sparse BM25 double-IDF and re-index (section 5.1) | Medium | Helps exact-ID recall (E025 class) |
| P2 | Report fixed-K precision and nDCG (section 3.3) | Low | Trustworthy, comparable retrieval metrics |
| P2 | Reduce candidate_pool or batch rerank to ease the token limit (section 5.3) | Low | Faster, cheaper, fewer throttles |
| P2 | Stabilise leakage-rate denominator; report counts (section 6.2) | Low | Comparable security numbers |
| P3 | Median-of-3 judge for gate-adjacent scores (section 6.1) | Medium | Less metric noise |
| P3 | Reranker upgrade/fine-tune; lexical prior for IDs (section 5.2) | High | Raises the precision ceiling |

**Definition of done for the security claim:** re-run the golden set after P0 and
require `answer_leakage_rate = 0%` on E035 and E048 with `unauthorized_retrieval_rate`
still 0.0%, and no recall regression on the authorized cases. Only then is
"permission-aware" defensible.

---

## 11. What was already fixed in this iteration (for context)

- Reranker scored the bare question instead of the conversation-resolved query - fixed
  (recovered E027, which returned an empty context before).
- Reranker truncated passages at 2000 chars, hiding 68% of the corpus - window raised
  to 4800.
- Reranker treated rate limits (429) as permanent and silently shipped unranked
  context - now retries with backoff and reports degradation.
- Selection spent the budget one chunk per document - replaced with document-aware
  selection (relative cut + bounded depth), then fixed a depth-first context-packing
  bug that was silently dropping documents 3-5.
- Added `precision_ceiling` / `precision_efficiency` diagnostics without changing the
  headline formulas.

Net effect on the golden set: recall 86.7 -> 95.5%, MRR 0.746 -> 0.853, correctness
83.0 -> 92.6%, failures 12 -> 6. The remaining work is in this document, and the most
important item is section 2.
