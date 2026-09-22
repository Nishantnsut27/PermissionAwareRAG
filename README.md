# Permission-Aware RAG for Enterprise Knowledge

A permission-aware enterprise knowledge assistant that combines retrieval-augmented generation with server-side authorization. The system is designed around a simple rule:

> Relevance determines what is useful; authorization determines what is allowed.

## Live Demo

[Streamlit application](https://permissionawarerag.streamlit.app/)

The current demo uses a prototype identity selector. Identity, role, clearance, and seller scope are resolved on the server and are not accepted from the client as authorization claims.

---

## The Problem with Conventional RAG

A conventional RAG pipeline usually looks like:

```text
User Query
    |
    v
Vector / Keyword Search
    |
    v
Top-K Context
    |
    v
LLM
    |
    v
Answer
```

This works for relevance, but relevance is not the same thing as authorization.

In an enterprise setting, two users can ask the same question while having different access rights. A document can also be highly relevant to a query while being outside the requesting user's permitted scope.

If authorization is added only after generation, the model may already have received information that the user was not allowed to retrieve.

The core problem is therefore:

**How do we retrieve useful information without allowing unauthorized information to reach the answering model?**

---

## Our Approach

PermissionAwareRAG makes authorization a first-class part of retrieval.

The permission engine derives an authorized scope from the server-side identity. That scope is applied to retrieval before documents are passed to the answering layer. Retrieved candidates are then authorization-checked again, reranked, checked at a final gate, and only then converted into LLM context.

### Permission-Aware Workflow

```mermaid
flowchart TD
    A["User Identity"] --> B["Permission Engine"]
    B --> C["Authorized Scope"]
    C --> D["Hybrid Retrieval"]
    D --> E["Authorization Re-check"]
    E --> F["Jina Reranking"]
    F --> G["Final Authorization Gate"]
    G --> H["Context Construction"]
    H --> I["LLM Generation"]
    I --> J["Deterministic Sources"]
    J --> K["Answer"]

    Q["User Query + Bounded Conversation"] --> D
```

The important security boundary is between retrieval and generation:

```text
Identity
   |
   v
Permission Engine
   |
   v
Authorized Retrieval Scope
   |
   v
Qdrant Hybrid Search
   |
   v
Post-Retrieval Authorization Check
   |
   v
Reranking
   |
   v
Final Authorization Gate
   |
   v
Authorized Context
   |
   v
LLM
```

### Workflow Reference

![Permission-aware RAG workflow](images/Permission_aware_rag_workflow.png)

---

## Authorization Model

The authorization layer combines role-based and attribute-based rules.

Access decisions consider:

- Role
- Department
- Clearance level
- Seller scope
- Document type
- Document classification
- Organization-wide document rules

Clearance controls the maximum classification a user can access. Seller scope controls seller-specific material. The role matrix controls which document types a role can read. Organization-wide restricted material can additionally require department alignment.

Authorization is implemented centrally in the `PermissionEngine`.

### Permission Matrix

| User | Department | Role | Clearance | Seller Scope |
|---|---|---|---|---|
| Aditya Verma | Business | Account Manager | CONFIDENTIAL | S001, S003 |
| Rahul Sharma | Support | Support Engineer | CONFIDENTIAL | S001, S002 |
| Vikram Singh | Engineering | Software Engineer | CONFIDENTIAL | S001, S002 |
| Neha Gupta | Operations | System Engineer | RESTRICTED | S002, S004 |
| Admin | IT | Platform Administrator | RESTRICTED | All sellers |

The UI displays the effective seller scope returned by the server. A caller can narrow its requested scope, but cannot use the request to expand its authorization.

![Permission matrix](images/permission_matrix.png)

---

## Retrieval Pipeline

### 1. Document Ingestion

The ingestion pipeline recursively discovers PDF and TXT documents, extracts content, derives metadata, validates metadata, cleans the text, and creates token-aware chunks.

Current dataset:

- 63 documents
- 52 PDF files
- 11 TXT files
- 209 generated chunks
- 0 failed documents
- 0 validation errors

Chunking uses:

- `RecursiveCharacterTextSplitter`
- `cl100k_base` tokenizer
- 800-token target size
- 120-token overlap
- Page-level provenance for PDF content

Metadata is preserved with every chunk so authorization decisions can be made after retrieval.

### 2. Indexing

Each chunk is indexed in Qdrant with both dense and sparse representations.

**Dense retrieval**

- Jina `jina-embeddings-v4`
- 2048-dimensional vectors
- Cosine similarity

**Sparse retrieval**

- Local BM25 encoder
- `k1 = 1.2`
- `b = 0.75`

**Hybrid retrieval**

- Dense top-k and sparse top-k candidates
- Reciprocal Rank Fusion
- Final retrieval pool before authorization-aware answering

The current indexed corpus contains 209 Qdrant points.

### 3. Permission-Aware Retrieval

The permission engine converts the identity into a Qdrant filter containing:

- Allowed classifications
- Allowed document types
- Authorized seller IDs

Rules that require comparing multiple metadata fields, such as organization-wide restricted documents and department matching, are enforced by the authorization engine after retrieval.

The system deliberately fails closed. An invalid identity, invalid scope, or unusable authorization filter does not silently turn into an unrestricted search.

### 4. Authorization Re-check

Candidate chunks are passed through the central permission engine again.

This provides defense in depth:

```text
Qdrant pre-filter
      |
      v
Candidate chunks
      |
      v
PermissionEngine.can_access()
      |
      v
Authorized chunks only
```

The LLM boundary is protected by an explicit invariant: an unauthorized chunk must never be included in the context sent for generation.

### 5. Reranking

Authorized candidates are reranked using Jina's multilingual reranker.

Reranking changes relevance ordering; it does not grant access.

A final authorization gate runs after reranking so that ranking can never reintroduce unauthorized content.

### 6. Context Construction and Citations

Only authorized chunks that fit the context budget are included in the final context.

Sources are generated deterministically from those same authorized chunks. The application adds the source list after generation, rather than allowing the model to invent page numbers or citations.

---

## Conversation Safety

Conversation history is treated as untrusted input.

It can help resolve references in follow-up questions, but it cannot:

- Change the user's identity
- Change clearance
- Expand seller scope
- Grant a role
- Override authorization decisions

The retrieval filter is always derived from the server-side identity.

This is important because a previous assistant response or a statement inside a conversation is not an authorization credential.

---

## Grounded Answer Generation

The answering layer receives only the authorized context.

The system prompt requires the model to:

- Use only supplied context
- Avoid inventing enterprise facts
- State when the context is insufficient
- Preserve identifiers and figures from the source
- Avoid generating its own source list
- Treat question, conversation, and retrieved documents as untrusted data

The current configured generation model is `openai/gpt-oss-120b` through the Groq-compatible API.

The answering service also supports streaming responses through the `/query/stream` endpoint.

---

## Application Architecture

```mermaid
flowchart LR
    U["Streamlit UI"] --> API["Answering HTTP API"]

    API --> ID["Identity Resolution"]
    ID --> PE["Permission Engine"]

    PE --> RET["Permission-Aware Retrieval"]
    RET --> Q["Qdrant"]

    RET --> AUTH["Authorization Re-check"]
    AUTH --> RR["Jina Reranker"]
    RR --> GATE["Final Authorization Gate"]

    GATE --> CTX["Context Builder"]
    CTX --> LLM["LLM"]

    LLM --> SRC["Verified Source Builder"]
    SRC --> API
    API --> U

    DATA["PDF / TXT Dataset"] --> ING["Ingestion"]
    ING --> CH["Chunks + Metadata"]
    CH --> IDX["Indexing"]
    IDX --> Q
```

---

## Real Application UI

The deployed application exposes the permission-aware workflow through a Streamlit interface:

1. Select a prototype identity.
2. View the identity's effective seller scope.
3. Ask a question about authorized enterprise material.
4. Retrieve and rerank authorized documents.
5. Generate a grounded answer.
6. Inspect the cited sources.
7. Review saved evaluation results.

![Permission-aware RAG UI](images/permission_aware_rag_ui.png)

---

## Evaluation

The repository contains a golden evaluation dataset with 60 cases covering:

- Authorized factual questions
- Multi-document retrieval
- Cross-seller queries
- Follow-up questions
- Unauthorized access attempts
- Mixed-access cases
- Policy and governance questions
- Ambiguous questions
- Prompt-injection attempts

The latest complete saved run is:

`evaluation/runs/full-20260921-v2.json`

### Latest Saved Run

| Metric | Result |
|---|---:|
| Cases | 60 |
| Passed | 48 |
| Failed | 12 |
| Errors | 0 |
| Recall@K | 86.67% |
| Precision@K | 50.98% |
| MRR | 74.63% |
| Groundedness | 97.95% |
| Answer relevancy | 98.59% |
| Answer correctness | 83.05% |
| Authorization accuracy | 98.33% |
| Unauthorized retrieval rate | 0.00% |
| Answer leakage rate | 4.55% |
| Prompt-injection blocked | 100% |

These numbers are from the saved evaluation run, not a claim of production-grade security. In particular, the non-zero answer leakage rate and the 12 failed cases are visible limitations of the current implementation and evaluation configuration.

The evaluation dashboard reads saved results and does not automatically start a quota-consuming evaluation run from the web UI.

---

## Environment

Copy `.env.example` to `.env` and provide the required service credentials.

Core variables include:

```env
JINA_API_KEY=
QDRANT_URL=
QDRANT_API_KEY=

GROQ_API_KEY_1=
GROQ_API_KEY_2=
GROQ_API_KEY_3=
GROQ_API_KEY_4=
GROQ_API_KEY_5=
```

The repository also supports configuration for embedding, reranking, retrieval, generation, API, conversation limits, and evaluation settings. See `.env.example` for the complete configuration surface.

Never commit real API keys.

---

## Local Setup

### 1. Create the environment

```bash
python -m venv .venv
```

Activate the environment and install dependencies:

```bash
pip install -r requirements.txt
```

### 2. Configure environment variables

```bash
copy .env.example .env
```

Fill in the required credentials.

### 3. Run ingestion

From the repository root:

```bash
python ingestion/run_ingestion.py
```

This generates the processed document catalog, chunks, and ingestion reports.

### 4. Build the Qdrant index

```bash
python indexing/pipeline.py
```

This embeds the chunks, builds the sparse representation, creates or validates the Qdrant collection, indexes the points, and runs retrieval sanity checks.

### 5. Start the answering API

```bash
python answering/serve.py
```

The local API listens on:

```text
http://127.0.0.1:8000
```

### 6. Start the Streamlit UI

In another terminal:

```bash
streamlit run app.py
```

For Windows, the repository also provides `run.bat`, which starts the API, waits for its health endpoint, and then launches Streamlit.

---

## API Surface

The answering service exposes:

| Method | Endpoint | Purpose |
|---|---|---|
| GET | `/health` | Service health check |
| GET | `/users` | Prototype identity directory |
| POST | `/query` | Generate a complete answer |
| POST | `/query/stream` | Stream an answer through SSE |
| GET | `/evaluation` | Read saved evaluation summary |
| GET | `/evaluation/matrix` | Read saved evaluation details |

Evaluation execution is intentionally kept outside the HTTP request path and is run explicitly from the evaluation CLI.

---

## Technology Stack

| Layer | Technology |
|---|---|
| UI | Streamlit |
| API | Python HTTP server |
| Ingestion | Python, pypdf |
| Chunking | LangChain text splitters, tiktoken |
| Dense embeddings | Jina Embeddings v4 |
| Sparse retrieval | Local BM25 |
| Vector database | Qdrant |
| Hybrid fusion | Reciprocal Rank Fusion |
| Reranking | Jina Reranker |
| LLM | `openai/gpt-oss-120b` via Groq-compatible API |
| Evaluation | Python-based golden dataset and judge pipeline |

---

## Security Design Principles

### Authorization is not a prompt

The model is never asked whether a user should have access to a document. The permission engine makes that decision before context construction.

### Relevance is not authorization

A semantically relevant document can still be inaccessible. Retrieval therefore operates inside an authorization-derived scope.

### Reranking does not grant access

Reranking can change ordering, but every selected chunk is checked again.

### Conversation is not identity

Previous messages cannot change the server-side identity or permissions.

### Fail closed

Invalid identities and invalid authorization scopes are rejected instead of being converted into unrestricted retrieval.

### Citations come from authorized evidence

The application constructs the final source list from the chunks that were actually authorized and included in context.

---

## Current Limitations

The current repository is a working prototype rather than a complete enterprise identity platform.

- The UI identity selector is a demo mechanism, not real user authentication.
- Authentication and identity federation are not implemented as a production identity provider integration.
- The latest evaluation still contains failed cases.
- The latest saved evaluation reports a non-zero answer leakage rate.
- Evaluation runs are intentionally executed separately from the web request path.
- Authorization policy is currently represented by the repository's policy and user definitions.

These limitations are intentionally documented rather than hidden behind the RAG interface.

---

## Repository Images

The `images/` directory contains the approved documentation visuals used by this README:

- `Permission_aware_rag_workflow.png`
- `permission_matrix.png`
- `permission_aware_rag_ui.png`

---

## Project Goal

PermissionAwareRAG explores how enterprise RAG can be designed so that access control is part of the retrieval architecture rather than an afterthought.

The central design goal is:

```text
User
  |
  v
Identity + Policy
  |
  v
Authorized Retrieval
  |
  v
Grounded Generation
  |
  v
Cited Answer
```

The result is a RAG system where the answer is constrained by both **what is relevant** and **what the requesting identity is authorized to access**.
