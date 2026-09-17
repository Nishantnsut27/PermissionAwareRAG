# Phase 1 - Synthetic Enterprise Dataset

Permission-Aware Enterprise Knowledge System with Strict Access Controls

## Purpose

This dataset is the Phase 1 deliverable for a prototype that will demonstrate
permission-aware enterprise knowledge retrieval. The prototype will need to
answer employee questions using only the information that the requesting
employee is authorised to access.

Phase 1 produces the enterprise knowledge itself: a small, realistic,
interconnected corpus of documents across several departments, sensitivity
levels and seller accounts. No application, retrieval system, authentication,
authorisation logic or evaluation set is part of this deliverable.

## Synthetic Data Disclaimer

**All content in this dataset is synthetic.** The company, the sellers, the
people, the incidents, the financial figures and the conversations are
fictional and were generated for this prototype. No reference to any real
organisation, person or event is intended. No production data was used.

The dataset is deliberately small - approximately 63 documents - so that
a developer can inspect it manually and explain it during a technical review.
Quality and interconnection were prioritised over volume.

## The Fictional Company

| Field | Value |
|---|---|
| Company | Nexora Commerce Pvt. Ltd. |
| Platform | Nexora Marketplace |
| Sector | Multi-category e-commerce marketplace |
| Headquarters | Bengaluru, Karnataka, India |
| Founded | 2019 |
| Sellers in dataset | 4 |
| Employees in dataset | 5 |

Nexora operates a marketplace on which third-party sellers list catalogues,
receive orders through platform integrations, fulfil those orders, and are
paid through a scheduled settlement cycle. Knowledge about each seller is
spread across five departments and six document categories.

## Departments

| Department | Responsibility |
|---|---|
| Business | Owns seller relationships, commercial terms, account strategy and tier management. |
| Support | Front-line seller support, ticket triage, escalation and customer communication. |
| Engineering | Owns platform services, integrations, reliability and incident remediation. |
| Operations | Runs settlement, fulfilment, onboarding operations and service recovery procedures. |
| IT | Owns identity, access, security operations and platform administration. |

## Users

The dataset uses exactly five identities. Each has a department, a role, a
seller scope and a clearance level. Scope and clearance are independent: both
must be satisfied before a document is accessible.

| Name | Department | Role | Seller Scope | Clearance |
|---|---|---|---|---|
| Aditya Verma | Business | Account Manager | S001, S003 | CONFIDENTIAL |
| Rahul Sharma | Support | Support Engineer | S001, S002 | CONFIDENTIAL |
| Vikram Singh | Engineering | Software Engineer | S001, S002 | CONFIDENTIAL |
| Neha Gupta | Operations | System Engineer | S002, S004 | RESTRICTED |
| Admin | IT | Platform Administrator | Organization-wide | RESTRICTED |

Three of the five identities are seller-scoped at CONFIDENTIAL clearance and
one is seller-scoped at RESTRICTED clearance. Only the Platform Administrator
holds organization-wide scope. This arrangement means that no single
non-administrative identity can see the whole dataset, which is the property
the future system is intended to demonstrate.

## Sellers

| Seller ID | Company | Category | Tier | Onboarded |
|---|---|---|---|---|
| S001 | Aurelia Home Decor | Home & Living | Gold | 2023-02-14 |
| S002 | BluePeak Electronics | Consumer Electronics | Platinum | 2022-06-01 |
| S003 | GreenCart Organics | Grocery & Fresh | Silver (conditional Gold upgrade in progress) | 2025-08-11 |
| S004 | StrideOne Footwear | Fashion & Footwear | Gold | 2025-09-22 |

The four sellers were designed to be structurally different rather than
variations of one another:

- **S001 Aurelia Home Decor** is an own-brand decor manufacturer with a
  real-time API integration, a single fulfilment centre, and a finance team
  that reconciles every settlement cycle.
- **S002 BluePeak Electronics** is the largest account on the platform, a
  multi-brand reseller with a very large catalogue, nightly batch
  integration, and two fulfilment centres.
- **S003 GreenCart Organics** is a subscription grocery business and the only
  perishable seller, with slot-based delivery, a connector-based integration,
  and a refund-on-report category exception.
- **S004 StrideOne Footwear** is a dropship operator with no warehouse of its
  own, four logistics partners, and a return rate driven by catalogue
  attribute accuracy.

## Document Categories

| Category | Location | Approx. count | Format |
|---|---|---|---|
| Organization and identity | identity/ | 1 | PDF |
| Seller account documents | sellers/ | 4 | PDF |
| Project and account documents | documents/projects/ | 13 | PDF |
| Support tickets | documents/support/ | 12 | PDF |
| Call transcripts | documents/calls/ | 10 | TXT |
| Engineering incident reports | documents/engineering/ | 8 | PDF |
| Operations runbooks | documents/operations/ | 6 | PDF |
| Company policies | documents/policies/ | 8 | PDF |
| Scenario context | scenario-context/ | 1 | TXT |

## Classification Levels

Every document carries one of four classifications. Classification is applied
consistently and is never random.

| Level | Meaning | Example |
|---|---|---|
| PUBLIC | Shareable externally without review | Published marketplace terms |
| INTERNAL | Any employee, regardless of seller scope | Runbooks, general policies |
| CONFIDENTIAL | Seller-specific or commercially sensitive | Account documents, tickets, incidents |
| RESTRICTED | Security, access or risk material | Credential exposure, suspension review |

Operations runbooks and most company policies are INTERNAL, because any
employee may need them and they carry no seller-identifying commercial data.
Anything naming a seller together with commercial, financial or defect detail
is CONFIDENTIAL. Material that would assist an attacker, or that records a
credential event or an account risk determination, is RESTRICTED.

## The Scenario Concept

The most important structural feature of this dataset is that documents are
**not independent**. They are grouped into twelve business scenarios, each of
which is documented from several departmental perspectives.

For example, scenario SC-001 describes a settlement shortfall on seller S001.
It is visible in six places:

- a project status report owned by Operations,
- a support ticket owned by Support recording what the seller reported,
- a call transcript recording what the seller said,
- an engineering incident report recording the technical root cause,
- a company policy stating the organisational rule that applies,
- a recovery runbook describing how the problem was recovered.

Each document is written for its own audience and uses different language and
different levels of technical detail. The underlying facts - dates,
identifiers, root cause, people, resolution - are identical across all of
them.

| Scenario | Seller | Title | Date |
|---|---|---|---|
| SC-001 | S001 | Seller S001 Settlement Shortfall | 2026-03-12 |
| SC-002 | S002 | Seller S002 Inventory Synchronisation Drift | 2026-05-19 |
| SC-003 | S002 | Seller S002 Order API Timeout Under Peak Load | 2026-07-16 |
| SC-004 | S001 | Seller S001 Tax Invoice Mismatch | 2026-02-24 |
| SC-005 | S003 | Seller S003 Shipping Label Provider Failover | 2026-06-18 |
| SC-006 | S004 | Seller S004 Size Chart Defect and Return Spike | 2026-08-05 |
| SC-007 | S003 | Seller S003 Duplicate Refund Processing | 2026-09-03 |
| SC-008 | S002 | Seller S002 Credential Exposure (Security) | 2026-04-21 |
| SC-009 | S003 | Seller S003 Conditional Tier Upgrade and Payment Terms | 2026-01-15 |
| SC-010 | S001 | Seller S001 Checkout Deployment Regression | 2026-08-12 |
| SC-011 | S002 | Seller S002 Order Event Loss (Consumer Lag) | 2026-06-02 |
| SC-012 | S004 | Seller S004 Account Suspension Review | 2026-09-08 |

Some scenarios are causally linked. SC-006, an Engineering catalogue defect,
produced the elevated return rate that triggered SC-012, an Operations
account suspension. That link is preserved across every document that
mentions either event.

The canonical facts for all twelve scenarios are recorded in
`scenario-context/scenario_overview.txt`.

## Folder Structure

```
phase-1-data/
|-- README.md
|-- identity/
|   `-- organization_and_users.pdf
|-- sellers/
|   |-- S001/  S002/  S003/  S004/        (account overviews)
|-- documents/
|   |-- projects/     S001/ S002/ S003/ S004/
|   |-- support/      S001/ S002/ S003/ S004/
|   |-- calls/        S001/ S002/ S003/ S004/
|   |-- engineering/  S001/ S002/ S003/ S004/
|   |-- operations/   organization-wide/
|   `-- policies/     organization-wide/
`-- scenario-context/
    `-- scenario_overview.txt
```

## Document Metadata

Every PDF begins with a metadata block recording Document ID, Seller ID,
Department, Classification, Owner, Created Date, Related Scenario ID and
Document Type. PDFs also carry page numbers and a classification label on
every page, and embed document metadata (title, author, subject, keywords) in
the PDF properties.

## Approximate Dataset Size

- Total documents: approximately 63
- PDF documents: approximately 52
- TXT documents: approximately 11
- Interconnected scenarios: 12
- Departments: 5
- Sellers: 4
- Users: 5

## How To Read This Dataset

1. Start with `identity/organization_and_users.pdf` for the organisational
   model, the five users, and the access rules.
2. Read `scenario-context/scenario_overview.txt` for the canonical facts of
   all twelve scenarios.
3. Pick one scenario and read its documents across the folders to see how a
   single event is fragmented across departments.
4. Compare a seller's account overview against the tickets, calls and
   incidents for that seller to see the permission boundaries in practice.
