# Case Study

Enterprise-facing narrative for the three domain implementations in agent-orchestrator (commit `094b419`). Written for solution architects and technical hiring managers evaluating consulting-style delivery of governed multi-agent systems.

Related: [ARCHITECTURE.md](ARCHITECTURE.md) · [GOVERNANCE.md](GOVERNANCE.md) · [SECURITY.md](SECURITY.md) · [AWS_SERVICE_MAPPING.md](AWS_SERVICE_MAPPING.md)

---

## Executive Summary

Agent Orchestrator is an enterprise-grade **multi-agent AI system for document analysis**, delivered as a Python monorepo with shared governance infrastructure. Three complete domain implementations — RFP response analysis, vendor evaluation, and contract risk review — prove that specialist agents, LLM-as-a-judge evaluation, guardrails, human-in-the-loop review, and immutable audit logging can be reused without cross-domain coupling. The pattern is designed for consulting delivery: stand up core once, extend domains independently, and map each control to AWS managed-service equivalents for production readiness discussions.

---

## Domain 1: RFP Response Analyzer

### Problem

Organizations spend days manually reviewing RFP documents: extracting requirements, mapping them to response capabilities, and identifying gaps. The work is slow, inconsistently documented, and hard to standardize across bid teams. Errors in extraction or missed gaps create bid risk that is difficult to audit after the fact.

### Approach

Three specialist agents decompose the analysis inside `packages/domain-rfp/`:

| Agent | Responsibility |
|---|---|
| **Requirements Extractor** | Processes documents via `DocumentProcessor` / `TextAnalyzer` (Textract and Comprehend adapters) and extracts structured requirements |
| **Capability Mapper** | Maps requirements to known enterprise capabilities (full / partial / none) using LLM analysis |
| **Gap Analyzer** | Performs chain-of-thought gap analysis and bid-risk assessment for unmatched or weak mappings |

An **`OutputEvaluator`** (LLM-as-a-judge) scores completeness, consistency, contradiction, and reasoning quality.

### Architecture

Sequential pipeline (`RfpAnalysisPipeline`): extract → map → gap → evaluate. Optional input/output **guardrails**, confidence-gated **HITL**, and **audit** logging. Unlike vendor/contract domains, RFP uses document OCR/NLP adapters and does **not** wire RAG. It may optionally register agent capabilities with an `OrchestratorEngine`, but execution remains hardcoded sequential calls. Details: [ARCHITECTURE.md](ARCHITECTURE.md#agent-pipeline-architecture).

### Governance

- Default confidence threshold `0.85` before auto-complete  
- Reasoning traces in specialist and evaluator prompts  
- Immutable audit entries for invocations and evaluation  
- **Note:** RFP logs HITL-triggering audit events as `ActionType.EVALUATION` (vendor/contract use `HITL_DECISION`) — documented inconsistency in [GOVERNANCE.md](GOVERNANCE.md#hitl-audit-inconsistency-documented)

### Outcome for stakeholders

Bid teams receive structured requirements, capability coverage, and gap risk with a quantified confidence score and a human review path when automation is uncertain.

---

## Domain 2: Vendor Evaluation Pipeline

### Problem

Vendor assessments often rely on subjective judgment, inconsistent criteria, and scattered source documents. Procurement and architecture teams need **standardized, evidence-based** evaluations that can be compared across vendors and defended in review meetings.

### Approach

Three specialist agents in `packages/domain-vendor/`:

| Agent | Responsibility |
|---|---|
| **Capability Researcher** | Analyzes vendor capability claims; optionally retrieves grounding chunks via `RAGRetriever` |
| **Pricing Analyst** | Evaluates pricing models, costs, and commercial risks |
| **Market Positioner** | Assesses segment fit, advantages, and market risks using capability and pricing summaries |

A **`VendorOutputEvaluator`** judges completeness, consistency, contradiction, and **evidence quality**.

### Architecture

Sequential pipeline (`VendorEvaluationPipeline`): research → price → position → evaluate. Optional **RAG** for the researcher, optional **PromptRegistry** version tracking, guardrails, evaluation, and HITL. No Textract/Comprehend path on this pipeline — input is vendor name + document text. No `OrchestratorEngine` usage.

### Governance

- Evidence-quality criterion pushes the judge to scrutinize thin or unsupported claims  
- RAG-backed research can attach retrieved chunks for grounding checks on output guardrails  
- HITL-trigger audits use `ActionType.HITL_DECISION`  
- Shared guardrails → evaluator → HITL → audit stack ([GOVERNANCE.md](GOVERNANCE.md))

### Outcome for stakeholders

Procurement receives a structured capability, pricing, and market assessment with confidence scoring and a review gate when evidence is weak or safety checks flag content.

---

## Domain 3: Contract Risk Reviewer

### Problem

Contract review under time pressure misses material risks: liability caps, indemnification gaps, IP ownership ambiguity, unfavorable termination, and compliance omissions. Findings need **clause-level provenance**, not a single opaque summary score.

### Approach

Three specialist agents in `packages/domain-contract/`:

| Agent | Responsibility |
|---|---|
| **Clause Risk Analyst** | Examines clauses for risk factors with severity-oriented structured output and reasoning |
| **Compliance Checker** | Evaluates regulatory/policy themes (e.g., data protection concepts such as GDPR/CCPA framing, SOC 2, ISO 27001-oriented checks as prompted) |
| **Terms Comparator** | Compares contract terms to preferred/standard language; optionally RAG-retrieves reference terms at query time |

A **`ContractOutputEvaluator`** uses five criteria: completeness, consistency, **risk_identification_quality**, **reasoning_depth**, and **source_citation_accuracy**. `_post_parse` clamps criterion scores into `[0.0, 1.0]`.

### Architecture

Sequential pipeline (`ContractReviewPipeline`): risk → compliance → terms → evaluate. Distinctive traits:

- Outer **try/except** on `review()` maps unexpected failures to `FAILED` status instead of raising through the caller  
- Optional RAG on the terms comparator  
- Optional prompt version registration  
- Same guardrails / HITL / audit composition as sibling domains  

### Governance

- Source-citation criterion pressures the judge to verify findings against clause text  
- HITL-trigger audits use `ActionType.HITL_DECISION`  
- Defensive failure handling reduces silent crashes in review workflows  

### Outcome for stakeholders

Legal and risk teams get layered findings (risk, compliance, terms) with confidence, criterion scores, and a human gate for borderline or flagged reviews.

---

## Shared Infrastructure

All three domains share `packages/core/` without importing each other:

| Capability | Module |
|---|---|
| Model / embedding providers | `cloud/` |
| Guardrails | `guardrails/` |
| Evaluator base class | `evaluation/` |
| HITL jobs | `hitl/` |
| Audit | `audit/` |
| RBAC / secrets ABC | `security/` |
| Logging / metrics | `logging/`, `metrics/` |
| RAG primitives | `rag/` |
| Prompt versioning | `prompts/` |

**Adding a fourth domain** requires domain agents, prompts, models, an evaluator subclass of `BaseOutputEvaluator`, and a pipeline file that composes guardrails/HITL/audit the same way — **without modifying** existing domains. Core changes are only needed for new shared primitives.

---

## Results and Metrics

What this repository demonstrates at Batch 8 / commit `094b419`:

| Signal | Evidence |
|---|---|
| Test coverage | **172** pytest tests passing |
| Pattern reuse | Three complete domain pipelines on one core |
| Governance stack | Input/output guardrails → LLM-as-a-judge → HITL → immutable audit |
| Provider flexibility | Bedrock and local inference behind `ModelProvider` / `EmbeddingProvider` |
| AWS integration surfaces | Bedrock, Textract, Comprehend adapters; Secrets Manager path interface-only |
| Portfolio documentation | Architecture, governance, security, case study, AWS service mapping |

Honest limitations (also tracked in architecture/security docs): sequential pipelines (not Step Functions), unused Redis, OrchestratorEngine/DAGExecutor not used as runtime, local embedding dimension mismatch vs `vector(1024)`, and production Secrets Manager implementation not yet built.

---

## Consulting Delivery Framing

A typical engagement narrative using this pattern:

1. **Discover** — identify a high-stakes document workflow (bid, vendor, contract) where incomplete automation is unacceptable without review.
2. **Stand up core** — providers, Postgres/pgvector, guardrails, evaluator base, HITL, audit, logging.
3. **Implement one domain** — three specialists + evaluator + pipeline composition.
4. **Govern** — set confidence thresholds with the business owner; enable HITL and RBAC for approvers.
5. **Extend** — add a second domain without rewriting core; prove reuse.
6. **Map to AWS** — use [AWS_SERVICE_MAPPING.md](AWS_SERVICE_MAPPING.md) to discuss Bedrock, A2I, Step Functions, ECS/EKS, Secrets Manager, and CloudTrail as production targets.

The repository is the reference implementation for that story: the code shows the control stack; the docs show the architectural and security rationale.

### What “done” looks like for a domain

- Specialist agents return structured outputs consumed by the next stage  
- Domain evaluator subclass with criteria matching stakeholder language  
- Pipeline wires optional guardrails, metrics, audit, and HITL  
- Tests cover happy path, low confidence → pending review, and guardrail block/flag behavior where applicable  
- Operators can explain every auto-complete vs human-review decision from audit + job records  

Across all three domains, the same story holds: automation accelerates analysis, governance decides when humans must intervene, and the audit trail makes that decision path reviewable.

---

## How to Read This with the Rest of the Suite

| If you need… | Read |
|---|---|
| Component tree and data flow | [ARCHITECTURE.md](ARCHITECTURE.md) |
| Confidence, HITL, audit enums | [GOVERNANCE.md](GOVERNANCE.md) |
| RBAC, triggers, secrets, logging | [SECURITY.md](SECURITY.md) |
| Exam-oriented AWS mapping | [AWS_SERVICE_MAPPING.md](AWS_SERVICE_MAPPING.md) |
| Clone / run | [README.md](../README.md) |
