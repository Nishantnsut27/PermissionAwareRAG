# Integrating Jev (TypeSafe AI, System One) into the Permission-Aware RAG

Date: 2026-09-23
Audience: the next engineer on this system
Prerequisite reading: `improvement.md` (this document proposes concrete fixes for the problems catalogued there)

---

## 1. What Jev is, in one paragraph

Jev is TypeSafe AI's flagship **System One** model. Unlike an LLM, it does not
generate text. You send it a **state** (the content to judge) and one or more typed
**questions**, and it returns **typed, calibrated answers with probabilities** that
code can branch on directly - no prompt engineering to force JSON, no parsing, no
"respond only with a number" begging. It is designed for fast, focused judgments
(TypeSafe quotes 70-500 ms) and every question in a request is evaluated in parallel
and in isolation, so asking more questions barely changes latency.

It exposes three primitives:

| Primitive | Question | Returns |
| --- | --- | --- |
| **Noul** | Is this statement true? (yes/no) | `noul` in 0-1 (probability of yes) |
| **Score** | Where on this ordered rubric? | `score`, `probabilities`, `confidence`, `legend` |
| **Choice** | Which option from a list? | `choice`, `probabilities`, `confidence` |

`state` may be a string, a JSON object, or an array of text (text only today - no
images/audio). Endpoint: `POST https://api.typesafe.ai/v1/systemone`, bearer auth with
`TYPESAFE_API_KEY`, model `jev-latest`. Python SDK: `pip install typesafe-sdk`.

### Why this matters *here specifically*

Read section 8 of `improvement.md` again: our four systemic problems are (1) the last
line of defense is a text-LLM's discretion, (2) the judge that defines truth is
non-deterministic and rate-limited, (3) opposing metrics were treated as one score,
(4) free-text judgments must be parsed and clustered at coarse values (the 0.50
cluster). Jev is almost purpose-built to attack (1), (2) and (4): it turns a
"convince an LLM to emit clean structured judgment" problem into a typed API call with
a calibrated probability and an explicit confidence you can route on.

**Jev does not replace the generator (Groq).** The RAG still needs an LLM to write the
answer. Jev replaces the places where we are currently *abusing* an LLM to make a
structured decision: the evaluation judge, the leakage detector, the relevance
threshold, and the answerability/refusal gate.

---

## 2. Where Jev fits this codebase (map of opportunities)

Ordered by value-to-effort. Each is expanded in section 3.

| # | Integration point | Problem it fixes (from improvement.md) | Primitive | Effort |
| --- | --- | --- | --- | --- |
| A | Evaluation judge replacement (`evaluation/judge.py`) | §6.1 non-deterministic, rate-limited judge; §4 the 0.50 cluster; E054 error | Score + Noul | Medium |
| B | **Inline leakage / cross-boundary guardrail** at answer time | §2 the ORG-001 leak (E035, E048) - the disqualifying bug | Noul | Medium |
| C | Chunk-level cross-seller redaction gate before context assembly | §2.4 structural leak fix (defense in depth) | Noul | Medium |
| D | Relevance reranking / selection scoring (`answering/src/reranker.py`, `selection.py`) | §5.2 reranker ceiling; §5.3 Jina token/rate limits; arbitrary 0.20/0.40 thresholds | Noul | Medium |
| E | Answerability / refusal gate (`answering/src/service.py`) | over-refusal vs leak balance; cleaner than "no chunk above threshold" | Noul | Low |
| F | Adaptive breadth via confidence | §3 the precision/recall tradeoff we were forced to pick by hand | Score/Noul confidence | Low |

The single highest-value item is **B** (and its structural sibling **C**): it directly
attacks the one class of failure that blocks the "permission-aware" claim. **A** is the
highest-confidence win because our judge is already the shakiest measurement surface.

---

## 3. The integrations in detail

### A. Replace the LLM-as-judge in evaluation

**Today** (`evaluation/judge.py`): we call Groq `gpt-oss-120b`, force it to emit a JSON
blob (`faithfulness`, `relevancy`, `correctness`, `fact_support[]`, ...), parse it, and
validate it. Problems observed in the real runs: it clusters correctness at exactly
0.50 on compound questions (E007, E018, E049), it flips PASS/FAIL on re-run (E018), and
it errored the whole case when all five Groq keys were rate-limited (E054).

**With Jev**, each metric becomes a typed question. The key move is the one TypeSafe
recommends: **decompose a complex judgment into atomic questions and combine in code.**

Groundedness / faithfulness - today it is one fuzzy 0-1 number. Decompose into one Noul
per claim in the answer, exactly as our `fact_coverage` already does per expected fact,
but calibrated and deterministic:

```python
# evaluation/jev_judge.py (sketch)
from typesafe_sdk import Noul, Score, TypeSafeClient

client = TypeSafeClient()  # reads TYPESAFE_API_KEY

def grade(question, answer, context, reference_answer, expected_facts):
    state = {
        "question": question,
        "context": context,          # what retrieval showed the model
        "answer": answer,            # what the model produced
        "reference_answer": reference_answer,
    }
    questions = {
        "relevancy": Score(
            instructions="How well does ANSWER address QUESTION, without drift or padding?",
            criteria=[
                "Does not address the question",
                "Addresses part of the question",
                "Fully addresses every part of the question",
            ],
        ),
        # One Noul per expected fact -> deterministic fact_coverage, calibrated.
        **{
            f"fact_{i}": Noul(
                instructions={
                    "fact": fact,
                    "question": "Does ANSWER assert `fact`, and is it supported by CONTEXT?",
                },
            )
            for i, fact in enumerate(expected_facts)
        },
    }
    resp = client.system_one(state=state, questions=questions)
    fact_ids = [f"fact_{i}" for i in range(len(expected_facts))]
    supported = [resp.answers[fid].noul > 0.5 for fid in fact_ids]
    relevancy_top = 2  # len(criteria) - 1
    return {
        "relevancy": resp.answers["relevancy"].score / relevancy_top,
        "fact_coverage": (sum(supported) / len(supported)) if supported else None,
        "supported_facts": sum(supported),
        "relevancy_confidence": resp.answers["relevancy"].confidence,
    }
```

Faithfulness can be one Noul per *sentence in the answer* ("Is this sentence supported
by CONTEXT?") averaged - the same decomposition. Correctness can be one Noul per
expected fact ("Does ANSWER agree with `fact`?") which is exactly what today's
`fact_support` array tries to be, minus the parsing and the 0.50 clustering.

Why this is strictly better for us:
- **Deterministic and calibrated.** No more PASS/FAIL flips on re-run; the 0.50 cluster
  disappears because each sub-fact is judged in isolation and combined in *our* code
  with *our* weights.
- **Confidence-gated escalation.** When Jev's confidence on a Score is low, route that
  case to a human or to the slow LLM judge - the three-path pattern TypeSafe describes.
  This is exactly what `improvement.md §6.1` asked for (treat near-gate scores as ties).
- **Latency and rate limits.** One parallel call at ~sub-second instead of a multi-second
  Groq generation, and it removes the judge from the Groq key pool that E054 exhausted.
- **No JSON contract.** Deletes `_parse_json`, the fenced-block regex, and the
  "unparseable output" failure mode in `judge.py`.

Keep the LLM judge behind a flag during migration and run both on the golden set to
measure agreement before switching (section 6).

### B. Inline leakage guardrail at answer time (the important one)

**This is the defense-in-depth fix for the ORG-001 leak (E035, E048).** Recall the root
cause from `improvement.md §2`: authorized documents (ORG-001) contain data about
sellers outside the caller's scope, and document-level ACL cannot stop it. The
structural fix is sub-document tagging, but a guardrail that runs on the *generated
answer* catches leaks regardless of where they came from.

Place it in `answering/src/service.py`, after the answer is produced and before it is
returned, using the caller's authorized scope:

```python
# answering/src/guardrail.py (sketch)
from typesafe_sdk import Noul, TypeSafeClient

client = TypeSafeClient()

def answer_leaks(answer: str, authorized_sellers: list[str],
                 all_sellers: list[str]) -> tuple[bool, float]:
    """True if the answer discloses data about a seller outside scope."""
    out_of_scope = [s for s in all_sellers if s not in authorized_sellers]
    if not out_of_scope:
        return False, 0.0
    resp = client.system_one(
        state={"answer": answer, "authorized_sellers": authorized_sellers},
        questions={
            "discloses_out_of_scope": Noul(instructions={
                "out_of_scope_sellers": out_of_scope,
                "question": ("Does ANSWER disclose substantive account, tier, "
                             "settlement, catalogue or profile data about any seller "
                             "in `out_of_scope_sellers`?"),
            }),
        },
    )
    p = resp.answers["discloses_out_of_scope"].noul
    return p > 0.5, p          # threshold tunable per risk appetite
```

Wire it into the pipeline (buffered and streaming both go through `_finalize` /
`_prepare` in `service.py`):

```python
leaked, p = guardrail.answer_leaks(answer_text, prepared.sellers, ALL_SELLERS)
if leaked:
    logger.warning("request=%s blocked by leakage guardrail p=%.2f", request_id, p)
    return _refusal(...)   # or redact the offending lines and re-answer
```

Notes:
- Thresholding is a product decision (Noul docs, "where to set the threshold depends on
  the cost of being wrong"). For a security product, set it low (block at p > 0.3) and
  send the middle band to a stricter check or a human.
- This runs on the answer, so it is model-agnostic and catches the E048/E035 pattern
  even when the offending data came from an authorized aggregate document.
- It is fast enough to run inline. For streaming, run it on the completed buffer before
  the final `done` event, or on a debounced running buffer.

### C. Chunk-level cross-boundary gate before context assembly

Better than blocking the answer is never letting the tainted content reach the model.
After authorization and before `context.build`, screen each candidate chunk that comes
from an org-wide / aggregate document for out-of-scope seller content:

```python
# in answering/src/service.py _prepare, after authorize_chunks, before rerank/context
def scrub_cross_boundary(chunks, authorized_sellers, all_sellers):
    out_of_scope = [s for s in all_sellers if s not in authorized_sellers]
    if not out_of_scope:
        return chunks, []
    questions = {
        f"leak_{i}": Noul(instructions={
            "out_of_scope_sellers": out_of_scope,
            "question": "Does this passage contain seller-specific data about any "
                        "seller in `out_of_scope_sellers`?",
        })
        for i, _ in enumerate(chunks)
    }
    # State carries all chunk texts; each question is judged in isolation, in parallel.
    resp = client.system_one(
        state={f"passage_{i}": c.text for i, c in enumerate(chunks)},
        questions=questions,
    )
    kept, dropped = [], []
    for i, c in enumerate(chunks):
        (dropped if resp.answers[f"leak_{i}"].noul > 0.5 else kept).append(c)
    return kept, dropped
```

This is the interim (defensive) fix from `improvement.md §2.4.2` made concrete, and it
composes with the permanent structural fix (sub-document tagging). Because Jev evaluates
all passages in one parallel request, screening the ~20-chunk candidate pool is a single
sub-second call. Keep it strictly *subtractive* (it can only drop chunks), so like the
selection stage it can never widen access.

### D. Relevance reranking and calibrated selection thresholds

TypeSafe publishes a re-ranking cookbook: one Noul per (query, candidate) pair - "Is
this passage relevant to the question?" - then sort candidates by the probability. This
maps onto `answering/src/reranker.py` and the thresholds in `answering/src/selection.py`:

```python
def jev_relevance(query: str, chunks: list) -> list[tuple[object, float]]:
    resp = client.system_one(
        state={"question": query},
        questions={
            f"rel_{i}": Noul(instructions={
                "passage": c.text,
                "question": "Is `passage` relevant and useful for answering QUESTION?",
            })
            for i, c in enumerate(chunks)
        },
    )
    scored = [(c, resp.answers[f"rel_{i}"].noul) for i, c in enumerate(chunks)]
    return sorted(scored, key=lambda x: x[1], reverse=True)
```

Why consider it:
- The relevance score becomes a **calibrated probability**, so the arbitrary `0.20`
  absolute and `0.40` relative thresholds in `config.py` gain real meaning, and the
  `precision_ceiling`/`efficiency` reasoning from `improvement.md §3` gets a principled
  cutoff.
- It sidesteps the Jina 100k-tokens/min pressure documented in `improvement.md §5.3` and
  the silent-degrade bug we had to fix - if Jev's throughput/limits suit us (check
  `/models` and `/api` for current limits and price).
- It is a drop-in for the existing `RerankOutcome` contract; keep Jina as a fallback, or
  A/B the two on the golden set.

Caveat: this is an addition to evaluate, not a slam-dunk replacement. Jina's
cross-encoder is purpose-built for ranking; measure MRR/recall of Jev-rerank vs
Jina-rerank on the 45 scored cases before switching. The reranking cookbook uses the
Noul probability directly rather than a boolean threshold, which is the right approach
for ordering.

### E. Answerability / refusal gate

Today refusal is implicit: if no chunk clears the rerank threshold, we refuse. A Noul
makes it explicit and calibrated:

```python
resp = client.system_one(
    state={"question": question, "context": context_block},
    questions={"answerable": Noul(
        instructions="Does CONTEXT contain enough information to answer QUESTION?")},
)
if resp.answers["answerable"].noul < 0.3:
    return _refusal(...)
```

This gives a cleaner, tunable boundary between "refuse" and "partial answer" (the
E048/E054 `partial_answer` behavior), independent of the reranker's score scale.

### F. Adaptive breadth via confidence (dissolves the precision/recall dilemma)

In section 3 of `improvement.md` we were forced to *hand-pick* an operating point
(recall-first) because precision and recall trade off with a fixed `max_documents`. Jev
confidence lets that be adaptive: ask a Score for the top document's sufficiency and use
its `confidence` to decide breadth.

- High confidence that the top 1-2 documents fully answer -> narrow the context
  (precision up, no recall cost because the evidence really is concentrated).
- Low confidence -> widen to `max_documents` (recall protected where it is actually
  needed).

This replaces one global knob with a per-query decision, which is the correct shape for
the tradeoff and something the current stack cannot express.

---

## 4. Architecture and how to wire it in

### Client and config

Add a thin, shared client (mirror how `answering/src/config.py` centralizes Groq/Jina):

```python
# answering/src/typesafe_client.py
import os
from typesafe_sdk import TypeSafeClient

TYPESAFE_API_KEY = os.getenv("TYPESAFE_API_KEY", "")
JEV_MODEL = os.getenv("JEV_MODEL", "jev-latest")
JEV_ENABLED = os.getenv("JEV_ENABLED", "false").strip().lower() == "true"

_client = TypeSafeClient() if TYPESAFE_API_KEY else None

def client():
    if _client is None:
        raise RuntimeError("TYPESAFE_API_KEY not set")
    return _client
```

- Add `TYPESAFE_API_KEY`, `JEV_MODEL`, `JEV_ENABLED` to `.env` and to
  `config.validate_env()` (only when `JEV_ENABLED`).
- Add `typesafe-sdk` to `requirements.txt`.
- **Fail closed, consistently with the existing design.** For the guardrails (B, C):
  if Jev is unavailable, prefer to refuse or fall back to the structural filter, never
  to silently pass tainted content. This mirrors the reranker rule ("fail open on
  relevance, closed on security") in `answering/src/reranker.py`.
- **Retry/backoff.** Reuse the pattern we added to `reranker.py` (retry 408/429/5xx with
  backoff, surface degradation in stats) so a Jev rate limit is never a silent pass.

### Where each call sits

```
_prepare (service.py)
  authorize_chunks
    -> [C] scrub_cross_boundary(chunks, scope)          # Jev Noul, subtractive
  rerank
    -> [D] jev_relevance(query, chunks)                 # Jev Noul, optional replacement
  select / context.build
    -> [F] adaptive breadth from Jev confidence          # Jev Score confidence
  build_messages -> llm.generate (Groq, unchanged)
_finalize
  -> [B] answer_leaks(answer, scope)                    # Jev Noul, output guardrail

evaluation/evaluator.py
  -> [A] jev_judge.grade(...) / leakage(...)            # Jev Score+Noul, replaces judge.py
```

### Determinism note for evaluation

`evaluation/runner.py:fingerprint` hashes the source of the pipeline. Adding Jev changes
the fingerprint (correct - it is a pipeline change). Jev's calibrated outputs are far
more stable run-to-run than the current LLM judge, which improves the reproducibility
complaint in `improvement.md §6.3`, but calibration is an aggregate property, not a
per-answer guarantee (their docs are explicit about this) - so keep confidence-gated
escalation for the borderline band.

---

## 5. Phased rollout

**Phase 1 - shadow the judge (no behavior change).** Implement `evaluation/jev_judge.py`.
Run it alongside the existing judge on `evaluation/results.json` cases; log both. No
gate changes. Goal: measure agreement and calibration.

**Phase 2 - leakage guardrail in shadow.** Add B and C behind `JEV_ENABLED`, in
"report-only" mode: log when they *would* block, do not actually block. Re-run the
golden set and confirm they fire on E035/E048 and stay quiet on the 9 authorized-pass
cases. This is the key validation for the security claim.

**Phase 3 - enforce guardrails.** Flip B/C to actually block/redact. Require
`answer_leakage_rate = 0%` on E035 and E048 with `unauthorized_retrieval_rate` still
0.0% and no recall regression (the definition of done from `improvement.md §10`).

**Phase 4 - relevance and breadth (optional, measured).** A/B Jev rerank (D) and adaptive
breadth (F) against the current recall-first defaults on the 45 scored cases. Keep
whichever wins on recall/MRR without reintroducing leaks.

**Phase 5 - retire the LLM judge** for the metrics where Jev agrees within tolerance;
keep the LLM judge only as the escalation target for low-confidence cases.

---

## 6. How to verify it (do not trust, measure)

- **Judge agreement.** On the completed golden run, compute correlation and
  disagreement between Jev Score/Noul verdicts and the current LLM judge. Investigate
  every case where they disagree by more than one band. Only migrate a metric once you
  trust the disagreements are Jev being *more* right (e.g. the 0.50 cluster).
- **Guardrail precision/recall.** Treat leakage detection as its own classification
  problem: on the golden set, does the guardrail flag exactly the cases with
  `forbidden`/leak expectations and not the clean ones? Tune the Noul threshold on that.
- **Latency and cost.** Record Jev `usage.input_tokens`/`output_tokens` and wall time;
  compare against the Groq judge time and the Jina rerank time it might replace. Confirm
  the inline guardrails keep answer latency acceptable (they should, at sub-second).
- **No security regression.** The hard gate: `unauthorized_retrieval_rate` must stay
  0.0% and answer-level leakage must fall. If either worsens, stop.

---

## 7. Risks, limits, and what NOT to use Jev for

- **It is not a generator.** It cannot write the answer, cite sources, or stream prose.
  Groq stays as the answering model. Jev only makes decisions about text.
- **Text only, today.** Fine for this corpus (PDF/TXT already extracted to text).
- **Calibration is aggregate, not per-answer.** A single high-confidence answer can
  still be wrong; their docs say so plainly. Keep confidence-gated human/LLM escalation
  for anything security-critical rather than trusting a lone Noul absolutely.
- **Atomic questions only.** Do not ask "is this relevant and safe and complete?" in one
  question - decompose and combine in code. This is a feature (auditable, tunable
  weights) but it is a design discipline the team must follow.
- **New third-party dependency and data-egress.** Answer text, context, and chunk text
  would be sent to TypeSafe. For a permission-aware/enterprise product, confirm the data
  processing terms and whether restricted content may leave the boundary - ironically,
  sending sensitive context to a leak-detector is itself a data-governance decision.
- **A guardrail is not the structural fix.** B and C reduce the blast radius of the
  ORG-001 problem but the correct fix remains sub-document seller tagging
  (`improvement.md §2.4.1`). Use Jev as defense in depth, not as a reason to skip the
  data-model fix.

---

## 8. Concrete first step

Smallest useful, lowest-risk slice that proves value on the exact failures we have:

1. `pip install typesafe-sdk`, add `TYPESAFE_API_KEY` and `JEV_ENABLED` to `.env` /
   `config.validate_env()`.
2. Implement `answering/src/guardrail.py` (integration B) in report-only mode.
3. Re-run only the leak cases: `--only E035,E048,E010,E005` (two leaks, two clean
   authorized cases) and confirm the guardrail flags E035/E048 and stays silent on the
   clean ones.
4. If it separates them cleanly, move to Phase 3 and enforce; then implement the judge
   replacement (A) in shadow.

Reference: TypeSafe docs - Introduction, Concepts/System One, Primitives (Choice, Score,
Noul), Confidence, API, and the Re-ranking / Parallel-questions cookbooks at
`https://docs.typesafe.ai`. The machine-readable index is `https://docs.typesafe.ai/llms.txt`.
