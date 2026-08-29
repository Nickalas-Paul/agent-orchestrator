# Security

Threat model, access controls, data protection, secrets, audit immutability, and logging classification for agent-orchestrator at commit `094b419`.

Related: [ARCHITECTURE.md](ARCHITECTURE.md) · [GOVERNANCE.md](GOVERNANCE.md) · [CASE_STUDY.md](CASE_STUDY.md) · [AWS_SERVICE_MAPPING.md](AWS_SERVICE_MAPPING.md)

---

## Shared Responsibility Context

Amazon Web Services operates the cloud infrastructure and managed AI services (Bedrock, Textract, Comprehend, Secrets Manager, KMS, and others). **This application sits entirely in the customer responsibility zone**: how models are invoked, how prompts and documents are validated, who may approve HITL jobs, how secrets are loaded in development, how audit rows are written, and how sensitive logs are routed.

Every control in this document is an application- or schema-level customer control. Mapping those controls to managed AWS equivalents (and the remaining gaps) is documented in [AWS_SERVICE_MAPPING.md](AWS_SERVICE_MAPPING.md).

```
┌──────────────────────────────────────────────────────────┐
│ AWS: physical facilities, hypervisor, managed services   │
└──────────────────────────────────────────────────────────┘
┌──────────────────────────────────────────────────────────┐
│ Customer (this repo): providers, guardrails, RBAC,       │
│ secrets abstraction, audit INSERT + DB triggers,         │
│ sensitivity logging, pipeline composition                │
└──────────────────────────────────────────────────────────┘
```

---

## Input Validation

First line of defense before specialist agents run: `GuardrailsEngine.check_input()` composed inside each domain `pipeline.py`.

| Check family | Mechanism in this codebase |
|---|---|
| Prompt injection | Regex / heuristic `INJECTION_PATTERNS` |
| PII | Optional `TextAnalyzer` (Comprehend adapter or local mock) when enabled |
| Topic boundary | Optional blocked-topic / domain boundary checks |
| Content policy (input path) | Config-gated checks aggregated into `GuardrailResult` |

`GuardrailAction.BLOCK` on input causes the pipeline to return a **failed** result without invoking the specialist chain. `FLAG` allows continuation but is recorded for later HITL pressure on the output side (see [GOVERNANCE.md](GOVERNANCE.md)).

### Gap vs Bedrock Guardrails

Implementation is **rule-based pattern matching** plus optional analyzer hooks. Amazon Bedrock Guardrails applies managed content classifiers, denied topics, and contextual grounding at the model API boundary. Novel phrasings that evade regex are a residual risk until a managed policy plane (or stronger classifiers) is wired.

---

## Output Filtering

`GuardrailsEngine.check_output()` applies post-generation checks on evaluator (and related) text:

| Check family | Purpose |
|---|---|
| Content safety | Pattern-based unsafe content detection |
| PII | Same analyzer path as input when enabled |
| Grounding | Optional comparison against retrieved RAG chunks |

Output **FLAG** or **BLOCK** forces HITL review in domain pipelines even when evaluator confidence is above the threshold. That prevents unsafe or poorly grounded text from being treated as auto-approved completion. Governance flow: [GOVERNANCE.md](GOVERNANCE.md#guardrails-design).

---

## Access Control (RBAC)

Application-layer HITL authorization lives in `packages/core/security/roles.py` — **not** inside `packages/core/hitl/`.

### Roles (`HITLRole`)

| Role | Intent |
|---|---|
| `REVIEWER` | Least privilege: inspect workload |
| `APPROVER` | Can complete reviews |
| `ADMIN` | Full HITL permissions including threshold overrides |

### Permissions (`HITLPermission`)

| Permission | REVIEWER | APPROVER | ADMIN |
|---|---|---|---|
| `VIEW_JOBS` | ✓ | ✓ | ✓ |
| `LIST_PENDING` | ✓ | ✓ | ✓ |
| `SUBMIT_REVIEW` | | ✓ | ✓ |
| `OVERRIDE_CONFIDENCE` | | | ✓ |

### Composition pattern

`HITLAccessControl` exposes:

- `check_permission(role, permission, context=None)` — raises `PermissionError` on deny; logs grant/deny
- `has_permission(role, permission)` — boolean probe

**RBAC is composed alongside `HITLManager`, not embedded in it.** Batch 8 added security modules without modifying `HITLManager`. Callers must invoke access control before privileged HITL APIs, for example:

```python
access.check_permission(HITLRole.APPROVER, HITLPermission.SUBMIT_REVIEW)
hitl_manager.submit_review(review)
```

There is no HTTP session or IAM federation in-repo; production would source roles from IAM / SSO / LDAP and keep this policy evaluation at the application boundary.

---

## Secrets Management

`packages/core/security/secrets.py`:

| Type | Behavior |
|---|---|
| `SecretsProvider` (ABC) | Abstract `get_secret(key, default=None)` and `has_secret(key)` |
| `EnvironmentSecretsProvider` | Reads `os.environ`, optional key `prefix` for namespacing |

**Local development:** environment variables (and `.env` loaded by tooling) are appropriate.

**Production path:** swap to an AWS Secrets Manager + KMS-backed implementation behind the same ABC. That class is **documented in comments only** — not implemented. The secrets module contains **no live boto3 calls**.

Agent and model code should depend on the interface so credential storage can change without rewriting specialists.

---

## Audit Immutability

Defense in depth for evidence integrity:

### Application level

`AuditLogger` performs **INSERT-only** writes of `AuditEntry` rows. Application paths are not designed to UPDATE or DELETE audit history.

### Database level

Postgres triggers on `agent_audit_logs` (from `infra/init-db/05-audit-immutability-trigger.sql`):

| Trigger | Event | Effect |
|---|---|---|
| `no_audit_update` | `BEFORE UPDATE` | Raises via `prevent_audit_modification()` |
| `no_audit_delete` | `BEFORE DELETE` | Raises via `prevent_audit_modification()` |

Any client — application pool, `psql`, admin tools — that attempts UPDATE/DELETE receives an exception stating the table is append-only. This closes the gap where application discipline alone could be bypassed.

Action types and HITL audit field semantics: [GOVERNANCE.md](GOVERNANCE.md#audit-trail).

---

## Logging Separation (Data Classification)

`packages/core/logging/logger.py` implements sensitivity-aware structlog routing.

| Construct | Behavior |
|---|---|
| `SensitivityLevel.OPERATIONAL` | Agent invocations, latency, errors, counts |
| `SensitivityLevel.SENSITIVE` | Document content, PII detections, audit fallback details |
| `get_logger(name)` | Default logger; **does not** bind sensitivity (unchanged operational default) |
| `get_sensitive_logger(name)` | Binds `sensitivity="sensitive"` |
| `filter_by_sensitivity` | Structlog processor: prefixes `[SENSITIVE]`, mirrors to dedicated logger when handlers exist |

Sensitive sink defaults to **stderr** (separate from primary stdout JSON stream) and optionally appends to `SENSITIVE_LOG_FILE` when set.

| Stream | Audience / retention intent |
|---|---|
| Operational | Broadly accessible ops metrics and diagnostics |
| Sensitive | Restricted access, different retention, avoid copying into general log aggregators without redaction controls |

---

## Known Security Debt

### P2 — `load_dotenv()` in library / tooling modules

`load_dotenv` is called in three places under `packages/core/`:

1. `packages/core/cloud/config.py`
2. `packages/core/services/config.py`
3. `packages/core/database/migrate.py`

Library modules should ideally not auto-load `.env`; that belongs at application entrypoints once `SecretsProvider` is wired through config. Tracked as P2.

### Embedding dimension mismatch

Schema and `VectorStore` use **`vector(1024)`**; local `nomic-embed-text` embeddings are **384**-dimensional. Misaligned vectors break or confuse retrieval when local embeddings are written into the production schema. Align dimensions before production RAG cutover. See [ARCHITECTURE.md](ARCHITECTURE.md#rag-pipeline).

### Other residual risks (honest scope)

| Risk | Mitigation today | Residual |
|---|---|---|
| Pattern-only guardrails | FLAG/BLOCK + HITL | Novel attacks may evade regex |
| Secrets in env for local | `EnvironmentSecretsProvider` | Not rotation-capable; no KMS |
| Compose default Postgres host port `5432` | `.env` sets `POSTGRES_PORT=5433` | Misconfiguration if `.env` skipped |
| Redis published but unused | None required | Attack surface of unused service if exposed |

---

## Threat-Oriented View (concise)

| Threat | Control in this repo | Limit |
|---|---|---|
| Prompt injection / jailbreak text in documents | Input injection heuristics → BLOCK/PASS | Pattern evasion |
| Accidental PII in prompts or outputs | PII checks + sensitive logging | Analyzer coverage / false negatives |
| Unauthorized HITL approval | `HITLAccessControl` permissions | Must be called by consumers |
| Tampered audit history | INSERT-only logger + `no_audit_update` / `no_audit_delete` | Protect DB credentials and backups separately |
| Secret sprawl in code | `SecretsProvider` ABC; env provider for local | Production Secrets Manager not built |
| Sensitive data in general logs | `SensitivityLevel` + `get_sensitive_logger` | Call sites must choose the sensitive logger |

This is not a formal STRIDE assessment; it is a map from implemented controls to the risks they target for portfolio and design-review conversations.

---

## Summary Checklist for Reviewers

- [ ] Input/output guardrails composed at pipeline boundary  
- [ ] HITL RBAC checked before submit/override operations  
- [ ] Secrets accessed via `SecretsProvider` abstraction  
- [ ] Audit append-only at app **and** DB trigger layers (`no_audit_update`, `no_audit_delete`)  
- [ ] Sensitive content logged via `get_sensitive_logger` / sensitivity processor  
- [ ] Customer responsibility acknowledged relative to AWS shared responsibility model  

For operational governance (confidence thresholds, HITL workflow, explainability), continue to [GOVERNANCE.md](GOVERNANCE.md).
