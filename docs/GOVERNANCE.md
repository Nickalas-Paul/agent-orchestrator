# Governance

Human-in-the-loop design, audit trail, confidence thresholds, guardrails, and defense-in-depth for agent-orchestrator at commit `094b419`.

Related: [ARCHITECTURE.md](ARCHITECTURE.md) · [SECURITY.md](SECURITY.md) · [CASE_STUDY.md](CASE_STUDY.md) · [AWS_SERVICE_MAPPING.md](AWS_SERVICE_MAPPING.md)

---

## Defense-in-Depth Stack

High-stakes document analysis (RFP response, vendor selection, contract risk) requires overlapping controls. The GenAI Security Scoping Matrix principle applies: higher-risk AI systems need more layers. Each layer below catches what the previous one missed.

| Layer | Component | Role |
|---|---|---|
| 1. Input guardrails | `GuardrailsEngine.check_input()` | Blocks or flags unsafe content **before** specialist LLM calls |
| 2. Output guardrails | `GuardrailsEngine.check_output()` | Filters or flags model/pipeline outputs **after** generation |
| 3. LLM-as-a-judge | `BaseOutputEvaluator` (+ domain subclasses) | Scores specialist quality; emits confidence and findings |
| 4. HITL gate | `HITLManager` | Persists jobs and pauses for human review when confidence is low or outputs were flagged |
| 5. Audit trail | `AuditLogger` | Insert-only records of invocations, evaluations, guardrail events, and HITL state |

```
Input ──► Guardrails (input)
              │ BLOCK → pipeline FAILED
              │ PASS / FLAG
              ▼
         Specialist agents (sequential)
              │
              ▼
         Evaluator (LLM-as-a-judge)
              │
              ▼
         Guardrails (output) ── FLAG/BLOCK forces HITL
              │
              ▼
         Confidence < threshold? ── yes ──► HITLManager (PENDING_REVIEW)
              │ no                              │
              ▼                                 ▼
         COMPLETED                         Human review
              │                                 │
              └────────── AuditLogger ◄─────────┘
```

Guardrails are **composed in pipeline files**, not embedded in agent implementations. See [ARCHITECTURE.md](ARCHITECTURE.md) for pipeline sequences.

---

## Guardrails Design

### Components

| Type | Location |
|---|---|
| `GuardrailsEngine` | `packages/core/guardrails/engine.py` |
| `GuardrailConfig` | `packages/core/guardrails/models.py` |
| `GuardrailResult` / `GuardrailCheckResult` | same |
| `GuardrailAction` | enum: **`PASS`**, **`FLAG`**, **`BLOCK`** |

Input checks (when enabled) include prompt-injection heuristics (`INJECTION_PATTERNS`), optional PII detection via an injected `TextAnalyzer` (Comprehend adapter or local mock), and optional topic-boundary enforcement. Output checks include content-safety pattern matching, optional PII detection, and optional grounding checks against retrieved chunks.

### Actions

| Action | Behavior in pipelines |
|---|---|
| **PASS** | Content proceeds; no forced HITL from guardrails |
| **FLAG** | Content is allowed through, but pipelines set `hitl_triggered = True` so review is required even if evaluator confidence is above threshold |
| **BLOCK** | Input path: pipeline returns **`FAILED`** (or equivalent failed result) without running specialists. Output path: FLAG/BLOCK both force HITL review on the completed specialist/evaluator path |

Composition pattern (all three domains):

```python
# Inside pipeline.py — not inside agent classes
if self._guardrails is not None:
    input_result = await self._guardrails.check_input(...)
    ...
    output_result = await self._guardrails.check_output(...)
```

**Implementation gap vs managed Bedrock Guardrails:** this engine is rule/pattern-based (regex heuristics, keyword/content-safety patterns, optional analyzer hooks). Amazon Bedrock Guardrails uses managed policy classifiers and contextual grounding checks. Gap analysis: [AWS_SERVICE_MAPPING.md](AWS_SERVICE_MAPPING.md).

---

## Evaluator and Confidence Model

Specialist agents produce structured JSON-like outputs. A **second LLM call** (the domain evaluator) judges those outputs against domain criteria.

### Behavior (shared `BaseOutputEvaluator`)

- Builds system/user messages via `_build_messages`, then joins contents into a single prompt.
- Invokes the model at **temperature `0.2`** and **`max_tokens=4096`** for consistent scoring.
- Parses JSON with `extract_json_object`; clamps confidence with `clamp_confidence` into **`[0.0, 1.0]`**.
- On parse failure, returns structured error payload with **fallback confidence `0.3`** and recommendation `flag_for_review`.
- Emits findings, contradictions, summary, and recommendation (`approve` or `flag_for_review`).

### Domain criteria

| Domain | Evaluator | Criteria |
|---|---|---|
| RFP | `OutputEvaluator` | completeness, consistency, contradiction, reasoning_quality |
| Vendor | `VendorOutputEvaluator` | completeness, consistency, contradiction, evidence_quality |
| Contract | `ContractOutputEvaluator` | completeness, consistency, risk_identification_quality, reasoning_depth, source_citation_accuracy |

Contract additionally clamps `criterion_scores` in `_post_parse`. Pipelines compare `eval_response.confidence_score` to a configurable **`confidence_threshold`** (default **`0.85`**). Below threshold → HITL.

Pipelines call **`evaluate_outputs()`**, not `evaluate()` directly — wrappers preserve domain-specific signatures while delegating to the shared skeleton. Architecture details: [ARCHITECTURE.md](ARCHITECTURE.md#evaluator-architecture).

---

## HITL Gate Design

### Components

| Type | Role |
|---|---|
| `HITLManager` | Persist jobs, list pending, submit reviews (`packages/core/hitl/manager.py`) |
| `PipelineJob` | Full pipeline inputs/outputs, evaluator reasoning, confidence, status |
| `HITLReview` | Human decision payload (`job_id`, `reviewer`, `decision`, `notes`) |
| `JobStatus` | `RUNNING`, `COMPLETED`, `PENDING_REVIEW`, `APPROVED`, `REJECTED`, `FAILED` |
| `ReviewDecision` | `APPROVE`, `REJECT` |

RBAC (`HITLAccessControl`) lives in `packages/core/security/` and is **composed beside** the manager — `HITLManager` was not modified to embed roles. Consumers are expected to call `access.check_permission(...)` before privileged operations such as `submit_review`. See [SECURITY.md](SECURITY.md#access-control-rbac).

### Workflow

1. Pipeline runs specialists + evaluator (+ optional output guardrails).
2. If confidence &lt; threshold **or** output guardrails FLAG/BLOCK, `hitl_triggered` is set.
3. When `HITLManager` is injected, `save_job(PipelineJob(...))` writes Postgres state with full pipeline payloads.
4. Status becomes `PENDING_REVIEW` when review is required, otherwise `COMPLETED`.
5. A reviewer later calls `submit_review(HITLReview)` with `APPROVE` or `REJECT` and optional notes; the manager updates job status and can write audit entries for the decision.

There is **no** `REVISION_REQUESTED` decision in the current model — only approve/reject.

### HITL audit inconsistency (documented)

When HITL is triggered, domains disagree on `ActionType` for the audit entry that records pending review:

| Domain | Action type used for HITL-trigger audit |
|---|---|
| **RFP** | `ActionType.EVALUATION` |
| **Vendor** | `ActionType.HITL_DECISION` |
| **Contract** | `ActionType.HITL_DECISION` |

All three still log the evaluator step itself as `ActionType.EVALUATION` where applicable. The RFP HITL-pending path reuses `EVALUATION` instead of `HITL_DECISION`. This is a **known legacy inconsistency** retained for honesty in ops/docs; normalize in a future batch if audit analytics require a single enum.

---

## Audit Trail

### Models

| Type | Notes |
|---|---|
| `AuditLogger` | Insert-only writer (`packages/core/audit/`) |
| `AuditEntry` | Session, agent, action, payloads, confidence, HITL fields, tokens, cost |
| `ActionType` | **Seven** values: `INVOKE`, `TOOL_CALL`, `EVALUATION`, `HITL_DECISION`, `RETRIEVAL`, `EMBEDDING`, `GUARDRAIL_CHECK` |
| `HITLStatus` | `PENDING_REVIEW`, `APPROVED`, `REJECTED` |

Pipelines log specialist invocations (`INVOKE` by default), evaluation steps (`EVALUATION`), and HITL-related entries as above. Guardrails may also emit `GUARDRAIL_CHECK` when FLAG/BLOCK results are audited.

### Immutability

Application code only INSERTs. Database triggers `no_audit_update` and `no_audit_delete` on `agent_audit_logs` reject UPDATE/DELETE regardless of client. Trigger details and threat framing: [SECURITY.md](SECURITY.md#audit-immutability).

---

## Explainability Design

Opaque model text is turned into inspectable decisions through four mechanisms:

1. **Chain-of-thought style prompts** — specialist and evaluator templates instruct structured reasoning (steps, findings, citations where domain-appropriate).
2. **Quantified confidence** — evaluator returns clamped `overall_confidence` plus findings/contradictions/recommendation; Contract also surfaces per-criterion scores.
3. **Full decision path in audit** — inputs/outputs, action types, HITL status, token/cost fields support after-the-fact review.
4. **HITL snapshots** — `PipelineJob` stores pipeline inputs/outputs and evaluator reasoning for the human reviewer.

### Interpretability vs performance

Reasoning-heavy prompts and a dedicated judge call increase token spend (often several times a single specialist call for the evaluation stage alone). This architecture **prioritizes interpretability** because RFP analysis, vendor evaluation, and contract risk are high-stakes: incomplete or contradictory automation without an audit trail is worse than higher latency/cost. Confidence thresholds and HITL absorb residual uncertainty instead of silently auto-approving borderline outputs.

---

## Operational Defaults (verified)

| Setting | Default in code |
|---|---|
| Pipeline confidence threshold | `0.85` |
| Evaluator temperature | `0.2` |
| Evaluator max tokens | `4096` |
| Parse-failure confidence | `0.3` |
| Guardrail PII default action | `FLAG` |

---

## Pipeline Integration Summary

| Domain | Input guardrails | Output guardrails | Evaluator method | HITL persistence | HITL-trigger audit type |
|---|---|---|---|---|---|
| RFP | Yes (text/path inputs) | Yes | `evaluate_outputs` | Optional `HITLManager` | `EVALUATION` |
| Vendor | Yes | Yes (may pass retrieved chunks) | `evaluate_outputs` | Optional `HITLManager` | `HITL_DECISION` |
| Contract | Yes | Yes (may pass retrieved chunks) | `evaluate_outputs` | Optional `HITLManager` | `HITL_DECISION` |

When `hitl_manager` or `guardrails` is omitted at construction, those layers are skipped — pipelines remain runnable in tests and local demos without Postgres HITL tables. Production-style governance injects all optional dependencies.

### Why optional composition matters

Consulting deliveries often need a **progressive enablement** path: specialists first, then evaluation, then HITL and audit as stakeholders accept the workflow. Optional constructor dependencies keep the same pipeline class usable from unit tests (no DB) through demos (metrics only) to governed environments (full stack), without forking domain code.

---

## Related Reading

- Component inventory and pipeline diagrams: [ARCHITECTURE.md](ARCHITECTURE.md)
- RBAC, secrets, logging classification, triggers: [SECURITY.md](SECURITY.md)
- Enterprise narrative per domain: [CASE_STUDY.md](CASE_STUDY.md)
- Managed-service equivalents (A2I, Bedrock Guardrails, CloudTrail): [AWS_SERVICE_MAPPING.md](AWS_SERVICE_MAPPING.md)
