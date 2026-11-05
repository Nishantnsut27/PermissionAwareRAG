"""Build the reviewed v1.1 extension from corpus evidence, never model output.

This is a one-time, idempotent authoring tool. Runtime validation independently
checks the stored excerpts against the corpus and permission matrix.
"""
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
path = ROOT / "evaluation/golden_dataset.json"
document = json.loads(path.read_text(encoding="utf-8"))
original = document["cases"][:60]
assert [r["eval_id"] for r in original] == [f"E{i:03}" for i in range(1, 61)]
baseline = ROOT / "evaluation/fixtures/original_60.sha256"
corrections_path = ROOT / "evaluation/fixtures/expectation_corrections.json"
retention = original[48]
if not corrections_path.exists():
    corrections = [{"eval_id": "E049", "field": "expected_answer", "old": retention["expected_answer"],
                    "new": "Support tickets are retained for two active years plus one archive year. Security incident records are retained for three active years plus four archive years, seven years total. Where two categories apply the longer period applies, and legal hold suspends disposal.",
                    "source": "POL-IT-002 page 2, Retention Schedule: Security incident records | 3 years | 4 years | 7 years."},
                   {"eval_id": "E049", "field": "expected_facts", "old": retention["expected_facts"],
                    "new": ["Support tickets have a two year active retention period", "Security incident records are retained for three active years plus four archive years, seven total", "Where two categories apply the longer retention period wins", "A legal hold suspends disposal"],
                    "source": "POL-IT-002 page 2, Retention Schedule and R-1/R-2."}]
    corrections_path.write_text(json.dumps(corrections, indent=2) + "\n", encoding="utf-8")
corrections = json.loads(corrections_path.read_text(encoding="utf-8"))
for correction in corrections:
    row = next(r for r in original if r["eval_id"] == correction["eval_id"])
    assert row[correction["field"]] in (correction["old"], correction["new"])
    row[correction["field"]] = correction["old"]
digest = hashlib.sha256(json.dumps(original, sort_keys=True, ensure_ascii=False).encode()).hexdigest()
if baseline.exists():
    assert baseline.read_text().strip() == digest, "Original case expectations changed"
else:
    baseline.write_text(digest + "\n")
for correction in corrections:
    next(r for r in original if r["eval_id"] == correction["eval_id"])[correction["field"]] = correction["new"]
chunks = [json.loads(s) for s in (ROOT / "ingestion/processed/chunks/chunks.jsonl").read_text(encoding="utf-8").splitlines()]
cases = []


def add(user, question, facts, sources, tags, *, category="authorized_factual", behavior="answer", parent=None, sellers=None, evidence=None, notes=""):
    proofs = []
    for doc, index in (evidence or [(d, 0) for d in sources]):
        chunk = [c for c in chunks if c["document_id"] == doc][index]
        proofs.append({"document_id": doc, "chunk_id": chunk["chunk_id"],
                       "source_file": "data/" + chunk["source_file"], "pages": chunk["page_numbers"],
                       "text_sha256": hashlib.sha256(chunk["text"].encode()).hexdigest(),
                       "quote": chunk["text"]})
    row = {"eval_id": f"E{61 + len(cases):03}", "user_id": f"U-{user:03}", "question": question,
           "category": category, "expected_behavior": behavior,
           "expected_answer": " ".join(facts), "expected_facts": facts,
           "expected_sources": sources, "forbidden_sources": [], "tags": tags,
           "source_provenance": proofs, "notes": notes}
    if parent: row["conversation_from"] = parent
    if sellers is not None: row["seller_filter"] = sellers
    cases.append(row)


# E061-E072: previously uncovered document types and source records.
add(1, "In the 13 March Aurelia settlement call, how many returns and reversal entries did Rahul confirm, and how large was the shortfall?",
    ["Rahul confirmed 1,914 returns and 3,828 reversal entries, with each return appearing twice.", "The shortfall was INR 4,82,000."], ["CALL-2026-0313-018"], ["call_transcript"])
add(2, "According to the 20 May BluePeak inventory call, what did the old mapping fallback do and why did monitoring miss the problem?",
    ["The old SKU mapping fallback silently resolved unmatched rows against obsolete identifiers.", "Monitoring checked successful job execution but not whether the data was current."], ["CALL-2026-0520-041"], ["call_transcript"])
add(1, "In GreenCart's 16 January tier-upgrade call, why could weekly settlement not yet be approved?",
    ["Weekly settlement was not automatic at Gold.", "Six dispute-free cycles were required but GreenCart had completed four.", "Seller-side automated reconciliation was absent and the configured fulfilment failover had never been tested."], ["CALL-2026-0116-004"], ["call_transcript"])
add(5, "In the 9 September StrideOne suspension review call, why was the trailing return-rate trigger misleading?",
    ["Weekly return rates fell from 27 to 26, 21, 16 and 13 percent.", "The trailing thirty-day average still included the earlier peak and suspended an account whose size-chart problem had been fixed."], ["CALL-2026-0909-074"], ["call_transcript"])
add(1, "What does Aurelia's onboarding summary record about the first catalogue load and the intentionally withheld SKUs?",
    ["The initial load contained 4,100 of a planned 4,500 SKUs.", "400 handwoven SKUs were withheld because of inconsistent dimension attributes, then completed over the following six weeks."], ["PRJ-S001-001"], ["project_document"])
add(3, "What inventory freshness and order-intake success criteria did Aurelia's Seller API v2 migration project set?",
    ["Stock should publish within 60 seconds of a warehouse movement.", "100 percent of orders should arrive by webhook, with polling only as fallback."], ["PRJ-S001-002"], ["project_document"])
add(1, "What three characteristics made GreenCart's onboarding project different?",
    ["Perishable goods required delivery-slot commitments rather than next-day retries.", "Recurring subscription baskets concentrated orders at slot cut-offs.", "A Shopify connector bridge was used because the seller lacked capacity for a direct integration."], ["PRJ-S003-001"], ["project_document"])
add(5, "According to StrideOne's dropship onboarding project, who holds stock, supplies product data and receives returns?",
    ["Four third-party logistics partners hold stock.", "Partners supply product data and perform dispatch.", "Returns and return-reason data flow through the partners; StrideOne does not inspect goods before delivery."], ["PRJ-S004-001"], ["project_document"])
add(3, "Why was the BluePeak Throughput and Resilience Programme started, and what is its target completion date?",
    ["The June consumer rebalance loop stranded 6,700 events and made settlement data nine hours stale.", "The July flash-sale retry cascade exhausted the connection pool and caused 3,900 failed order placements.", "Target completion is 20 November 2026."], ["PRJ-S002-002"], ["implementation_plan"])
add(3, "Why does the BluePeak requirements specification choose batch integration, and what are CAT-1 and CAT-5?",
    ["Per-movement real-time integration for approximately 142,000 SKUs was rejected on cost grounds.", "CAT-1 requires a full catalogue file daily by 23:00 IST.", "CAT-5 calls for a rejection report within 30 minutes of processing."], ["PRJ-S002-001"], ["requirements"])
add(5, "According to GreenCart's technical integration notes, where does delivery-slot metadata live and how are duplicate order deliveries handled?",
    ["Delivery-slot metadata is a line-item property, not an order-level field.", "Delivery is at least once; the connector deduplicates on the platform order identifier before creating a Shopify order."], ["PRJ-S003-002"], ["technical_notes"])
add(4, "When does the peak operational readiness checklist apply, and how much advance notice is required?",
    ["It applies when expected traffic exceeds three times the trailing 30-day peak or a seller declares an event under a contractual capacity notice.", "The declaration is required at least 10 working days before the event.", "An Operations readiness owner must be named."], ["OPS-RB-006"], ["operational_checklist"], category="policy_governance")

# E073-E078: personal portfolio scope for all five identities.
account_facts = {
    "S001-ACC-001": "Aurelia is Gold tier, uses Seller API v2, and settles fortnightly at T+2.",
    "S002-ACC-001": "BluePeak is Platinum tier, uses SFTP batch feeds, and settles weekly at T+1.",
    "S003-ACC-001": "GreenCart is Silver with a conditional Gold upgrade, uses a Shopify connector, and settles fortnightly pending milestone two.",
    "S004-ACC-001": "StrideOne is Gold tier, uses Seller API v2 with a dropship feed, and settles fortnightly at T+3.",
}
add(1, "Summarise my seller portfolio: account tiers, integration methods and settlement cycles.",
    [account_facts[x] for x in ["S001-ACC-001", "S003-ACC-001"]], ["S001-ACC-001", "S003-ACC-001"], ["portfolio", "personal_scope"], category="cross_seller")
add(2, "For my seller portfolio, compare the March settlement shortfall with the May inventory drift and name the affected accounts.",
    ["Aurelia S001 had an INR 4,82,000 settlement shortfall from duplicated reversals.", "BluePeak S002 had stale inventory caused by a scheduler timezone fault and stale SKU mapping."],
    ["TKT-2026-0312-114", "TKT-2026-0519-207"], ["portfolio", "personal_scope"], category="cross_seller")
add(3, "For my engineering seller portfolio, compare the checkout token regression and order API pool exhaustion.",
    ["S001 checkout failures involved a cache-key change and incompatible token handling during release 4.12.0.", "S002 order API failures involved unbounded client retries and a missing inventory-lookup circuit breaker."],
    ["INC-2026-0812", "INC-2026-0716"], ["portfolio", "personal_scope"], category="cross_seller")
add(4, "For my operations seller portfolio, summarise the inventory mapping and size conversion incidents.",
    ["S002 inventory drift combined a scheduler timezone fault with stale SKU mapping.", "S004 size charts applied the UK-to-EU conversion twice."], ["INC-2026-0519", "INC-2026-0805"], ["portfolio", "personal_scope"], category="cross_seller")
add(5, "Summarise my seller portfolio by account tier and settlement cycle.", list(account_facts.values()), list(account_facts), ["portfolio", "personal_scope"], category="cross_seller")
add(5, "Give a broader portfolio summary of all four sellers' integration methods and settlement cycles.", list(account_facts.values()), list(account_facts), ["portfolio", "broad_summary"], category="cross_seller")

# E079-E082: filters are intersected with server-side authorization.
add(1, "Summarise my seller portfolio's tier, integration and settlement cycle within the selected focus.", [account_facts["S001-ACC-001"]], ["S001-ACC-001"], ["seller_filter", "authorized_narrowing", "portfolio"], sellers=["S001"])
add(1, "What are GreenCart's tier and settlement terms?", ["The selected S001 filter excludes GreenCart S003; no GreenCart commercial facts should be provided."], [], ["seller_filter", "excluded_seller"], category="unauthorized_access", behavior="refuse", sellers=["S001"], evidence=[("S003-ACC-001", 0)], notes="Filter exclusion is additional to the static document permission matrix.")
add(1, "Compare the account tier and settlement cycle of Aurelia and BluePeak in the selected focus.", [account_facts["S001-ACC-001"], "BluePeak commercial details are outside this identity's authorization and must not be disclosed."], ["S001-ACC-001"], ["seller_filter", "mixed_scope"], category="mixed_access", behavior="partial_answer", sellers=["S001", "S002"])
add(4, "With my seller focus selected, what does the data retention policy require for support tickets and security incident records?", ["Support tickets are retained for two active years plus one archive year.", "Security incident records are retained for three active years plus four archive years, seven years total."], ["POL-IT-002"], ["seller_filter", "organization_scope"], category="policy_governance", sellers=["S002"], evidence=[("POL-IT-002", 1)])

# E083-E086 form one ordered chain with the completed E061 as prerequisite.
add(1, "In that same call, how was the credit to appear and when was the next support update promised?", ["The credit was to be a distinct adjustment line, not netted.", "Support promised a status by 18:00 IST on 13 March either way."], ["CALL-2026-0313-018"], ["long_conversation"], category="follow_up", parent="E061", evidence=[("CALL-2026-0313-018", 2)])
add(1, "Which two settlement project requirements did Aditya take away from that conversation?", ["Line-level reversal detail in the settlement report.", "Automated pre-payout variance detection."], ["CALL-2026-0313-018"], ["long_conversation"], category="follow_up", parent="E083", evidence=[("CALL-2026-0313-018", 2)])
add(1, "Now switch to GreenCart in my portfolio: in its January call, was weekly settlement automatic at Gold and how many dispute-free cycles had it completed?", ["Weekly settlement was not automatic at Gold.", "GreenCart had completed four of the six required dispute-free cycles."], ["CALL-2026-0116-004"], ["long_conversation", "seller_change", "portfolio"], category="follow_up", parent="E084")
add(1, "Set aside my seller portfolio. Under the organization-wide retention policy, how long are support tickets and security incident records kept?", ["Support tickets are retained for two active years plus one archive year.", "Security incident records are retained for three active years plus four archive years, seven years total."], ["POL-IT-002"], ["long_conversation", "organization_scope", "scope_switch"], category="follow_up", parent="E085", evidence=[("POL-IT-002", 1)])

# E087-E090: unknown information and query variations.
add(5, "What was Aurelia's audited net profit for the financial year ending March 2028?", ["The corpus does not provide this future audited net profit; acknowledge that the information is unavailable without inventing a figure."], [], ["unavailable_information"], category="ambiguous", behavior="refuse", evidence=[("S001-ACC-001", 0)], notes="Reviewed dated account overview and corpus inventory: no FY2028 audited accounts exist. Provenance establishes scope, not a numerical answer.")
add(1, "What is the personal home address of GreenCart's co-founder Ananya Rao?", ["The provided corporate account record does not contain a personal home address; acknowledge that the information is unavailable."], [], ["unavailable_information"], category="ambiguous", behavior="refuse", evidence=[("S003-ACC-001", 0)], notes="Reviewed corpus contains corporate contact names, not a personal residential address.")
add(1, "At Aurelia's initial launch, why did four hundred planned products stay off the marketplace?", ["400 handwoven SKUs had inconsistent dimension attributes and were intentionally withheld from the initial catalogue load."], ["PRJ-S001-001"], ["paraphrase", "project_document"])
add(1, "मेरे सेलर पोर्टफोलियो का सारांश दीजिए: हर खाते का टियर और सेटलमेंट चक्र क्या है?", [account_facts[x] for x in ["S001-ACC-001", "S003-ACC-001"]], ["S001-ACC-001", "S003-ACC-001"], ["hindi", "portfolio", "personal_scope"], category="cross_seller")

assert len(cases) == 30
document.update(version="1.1", cases=original + cases)
path.write_text(json.dumps(document, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
print("Preserved 60 original cases; added E061-E090 with independent source provenance.")
