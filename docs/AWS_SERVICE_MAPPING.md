# AWS Service Mapping — agent-orchestrator

Maps every component built in this project to its AWS managed service equivalent.
Primary study reference for AIF-C01 exam service-to-use-case questions.

## How to Use This Document

For each mapping: understand what YOUR component does, what the AWS service does,
how they are similar, how they differ, and which exam objectives the mapping covers.
The "gap" between your implementation and the managed service is where the exam
value lives — the managed service handles things your implementation does not.

Exam objective numbers below align with the AWS Certified AI Practitioner (AIF-C01)
domains commonly tested for service selection, responsible AI, and security.

---

## Service Mappings

### Model Inference

| Your Component | AWS Equivalent | Exam Objectives |
|---|---|---|
| `BedrockProvider` (`packages/core/cloud/`) | Amazon Bedrock | 2.1.2, 3.1.1, 3.2.1 |

**What yours does:** Abstract `ModelProvider` interface with `BedrockProvider` (calls Bedrock Messages API) and `LocalProvider` (calls Ollama-compatible endpoint). Constructor injection lets pipelines swap providers without code changes.

**What Bedrock does:** Managed service providing access to multiple foundation models (Claude, Titan, Llama, Mistral, and others) through a unified API. Handles hosting, scaling, versioning, and billing. Includes model customization (fine-tuning / continued pre-training), provisioned throughput, and cross-region inference profiles.

**Gap:** Your provider handles invocation and token/latency normalization only. Bedrock also manages model lifecycle, auto-scaling, custom model training jobs, provisioned throughput reservation, and marketplace model subscriptions.

---

### Embeddings and RAG

| Your Component | AWS Equivalent | Exam Objectives |
|---|---|---|
| `EmbeddingProvider` + pgvector RAG (`packages/core/rag/`, `packages/core/cloud/embeddings.py`) | Amazon Bedrock Knowledge Bases + OpenSearch Serverless / Amazon RDS | 2.2.1, 3.1.2, 3.2.2 |

**What yours does:** Chunks documents, embeds text via Bedrock Titan or local embedding models, stores vectors in Postgres with pgvector, and retrieves top-k chunks for specialist agents (vendor capability research, contract terms comparison).

**What AWS does:** Bedrock Knowledge Bases orchestrate ingestion, chunking, embedding, and retrieval against vector stores such as OpenSearch Serverless or Aurora/RDS with pgvector. Managed sync from S3 sources, metadata filters, and citation-ready retrieval APIs reduce custom plumbing.

**Gap:** You own chunking strategy, embedding provider selection, table schemas, and retriever wiring. Knowledge Bases add managed ingestion pipelines, multi-data-source connectors, and automatic index maintenance.

---

### Guardrails

| Your Component | AWS Equivalent | Exam Objectives |
|---|---|---|
| `GuardrailsEngine` (`packages/core/guardrails/`) | Amazon Bedrock Guardrails | 4.1.1, 4.2.1, 5.2.1 |

**What yours does:** Application-layer input/output checks (injection heuristics, PII patterns via Comprehend adapter hooks, domain-scoped FLAG/BLOCK actions) wired optionally into RFP, vendor, and contract pipelines. Results can force HITL review.

**What Bedrock Guardrails does:** Declares denied topics, content filters, word filters, sensitive-information filters, and contextual grounding checks applied at the Bedrock API boundary. Policies are versioned and reusable across applications.

**Gap:** Yours is code-defined and pipeline-composed. Bedrock Guardrails is a managed policy plane with console/API configuration, grounding checks against reference sources, and centralized enforcement without changing agent code.

---

### Prompt Versioning

| Your Component | AWS Equivalent | Exam Objectives |
|---|---|---|
| `PromptRegistry` (`packages/core/prompts/`) | Amazon Bedrock Prompt Management | 2.1.3, 3.2.1 |

**What yours does:** Stores named prompt versions with history in Postgres, registers domain prompts at pipeline construction, and supports retrieval of versioned templates for reproducibility.

**What Bedrock Prompt Management does:** Managed storage of prompt templates (variables, variants), versioning, and invocation by ARN so applications do not embed prompt text. Integrates with Bedrock Agents and Flows.

**Gap:** You implement registry CRUD and domain registration. Bedrock adds console UX, ARN-based references, A/B variants, and tight integration with Agents/Flows without a custom schema.

---

### LLM-as-a-Judge Evaluation

| Your Component | AWS Equivalent | Exam Objectives |
|---|---|---|
| `BaseOutputEvaluator` (`packages/core/evaluation/`) | Amazon Bedrock Model Evaluation | 2.3.1, 3.1.3, 4.2.2 |

**What yours does:** Shared `BaseOutputEvaluator` skeleton (invoke at temperature 0.2, parse JSON judgment, clamp confidence) with domain subclasses for RFP, vendor, and contract specialist outputs. Pipelines use confidence thresholds to trigger HITL.

**What Bedrock Model Evaluation does:** Managed evaluation jobs comparing models or prompt variants using automatic metrics and/or human workforces. Supports LLM-as-a-judge style scoring and leaderboard-style reporting.

**Gap:** You evaluate live pipeline specialist outputs in-process. Bedrock Model Evaluation is batch/job oriented for model selection and regression, with managed datasets, metrics dashboards, and optional human evaluation loops.

---

### Human-in-the-Loop

| Your Component | AWS Equivalent | Exam Objectives |
|---|---|---|
| `HITLManager` + `HITLAccessControl` (`packages/core/hitl/`, `packages/core/security/`) | Amazon A2I (Augmented AI) + AWS IAM | 4.2.2, 5.1.1, 5.1.2 |

**What yours does:** Persists `pipeline_jobs`, pauses for review when confidence is low or guardrails flag output, records approve/reject via `submit_review`, and applies application-layer RBAC (`REVIEWER` / `APPROVER` / `ADMIN`) through `HITLAccessControl` composed beside the manager.

**What A2I + IAM does:** A2I routes low-confidence predictions to human review workforces (private or Amazon Mechanical Turk / vendor). IAM roles and policies authorize who can claim and complete tasks. Review results flow back into the ML pipeline.

**Gap:** Your HITL is a Postgres job table plus Python RBAC. A2I adds workforce management, task UI templates, SNS/EventBridge integration, and IAM-native authorization at scale.

---

### Audit Logging and Immutability

| Your Component | AWS Equivalent | Exam Objectives |
|---|---|---|
| `AuditLogger` + Postgres immutability trigger (`packages/core/audit/`, `infra/init-db/05-audit-immutability-trigger.sql`) | AWS CloudTrail + Amazon CloudWatch + S3 Object Lock | 5.1.3, 5.2.2, 5.3.1 |

**What yours does:** Insert-only `AuditLogger` writes to `agent_audit_logs`; application never UPDATEs/DELETEs. A Postgres trigger raises on UPDATE/DELETE so immutability is enforced in the database (closes audit finding P3). Structlog fallback preserves entries if DB write fails.

**What AWS does:** CloudTrail records API activity across accounts/regions. CloudWatch Logs stores application telemetry. S3 Object Lock (WORM) and CloudTrail log file integrity validation prevent tampering with evidence stores.

**Gap:** You protect one application table. CloudTrail is organization-wide control-plane audit; Object Lock is storage-layer WORM. Production designs often combine app audit rows with CloudTrail and locked S3 archives.

---

### Document Processing (OCR / Forms)

| Your Component | AWS Equivalent | Exam Objectives |
|---|---|---|
| `DocumentProcessor` Textract adapter (`packages/core/services/`) | Amazon Textract | 2.2.2, 3.1.1 |

**What yours does:** `DocumentProcessor` ABC with local and AWS Textract-backed implementations selected via `TEXTRACT_PROVIDER`. RFP pipeline uses it to extract text/blocks from uploaded documents before specialist agents run.

**What Textract does:** Fully managed OCR, forms, tables, queries, and expense/ID analysis with synchronous and asynchronous APIs, multipage PDF support, and confidence scores per block.

**Gap:** Your adapter normalizes results for agents. Textract adds specialized analyzers, async job orchestration with SNS, and SLA-backed accuracy/scale without self-hosting OCR.

---

### Text Analytics and PII

| Your Component | AWS Equivalent | Exam Objectives |
|---|---|---|
| `TextAnalyzer` Comprehend adapter (`packages/core/services/`) | Amazon Comprehend | 2.2.2, 4.1.2, 5.2.1 |

**What yours does:** `TextAnalyzer` ABC exposing entity/PII-oriented analysis via local mocks or AWS Comprehend, configured by `COMPREHEND_PROVIDER`. Used for enrichment and guardrail-related PII signals.

**What Comprehend does:** Managed NLP for entities, key phrases, sentiment, language detection, PII detection/redaction, custom classification, and custom entity recognition.

**Gap:** You consume a thin analyzer interface. Comprehend adds custom model training, real-time and batch endpoints, and PII redaction APIs suitable for compliance pipelines.

---

### Cost and Token Metrics

| Your Component | AWS Equivalent | Exam Objectives |
|---|---|---|
| `MetricsTracker` (`packages/core/metrics/`) | AWS Budgets + AWS Cost Explorer | 3.3.1, 5.3.2 |

**What yours does:** In-memory per-call token and estimated USD cost tracking keyed by agent and session, with model-family pricing heuristics and pipeline business metrics (`BusinessMetrics`).

**What AWS does:** Cost Explorer analyzes historical spend; Budgets alerts when forecasted or actual costs exceed thresholds. Cost allocation tags attribute Bedrock and other AI spend to teams or products.

**Gap:** Your tracker is application-local and estimate-based. Budgets/Cost Explorer operate on billed AWS usage with org-level reporting, anomaly detection, and reserved/provisioned throughput economics.

---

### Secrets Management

| Your Component | AWS Equivalent | Exam Objectives |
|---|---|---|
| `SecretsProvider` / `EnvironmentSecretsProvider` (`packages/core/security/secrets.py`) | AWS Secrets Manager + AWS KMS | 5.1.1, 5.1.2, 5.2.2 |

**What yours does:** ABC for secret retrieval with `EnvironmentSecretsProvider` reading `os.environ` (local/dev). Callers depend on the interface so production can swap implementations without changing agents.

**What Secrets Manager + KMS does:** Stores and rotates secrets (DB credentials, API keys); KMS encrypts secret payloads and controls decrypt via IAM. Applications fetch secrets at runtime instead of baking them into images or `.env` files.

**Gap:** Environment provider is plaintext env vars (and committed defaults for local Postgres). Secrets Manager adds rotation, audit of secret access, fine-grained IAM, and envelope encryption with KMS CMKs.

---

### Logging Separation by Sensitivity

| Your Component | AWS Equivalent | Exam Objectives |
|---|---|---|
| `SensitivityLevel` + `get_sensitive_logger` (`packages/core/logging/`) | Amazon CloudWatch Logs + Amazon Macie | 4.1.2, 5.2.1, 5.2.2 |

**What yours does:** Structlog processor `filter_by_sensitivity` tags sensitive events with `[SENSITIVE]`, optionally mirrors them to a dedicated stderr/file sink (`SENSITIVE_LOG_FILE`). `get_sensitive_logger` binds `sensitivity="sensitive"`; default `get_logger` behavior stays operational.

**What CloudWatch + Macie does:** CloudWatch Logs stores and filters application logs (metric filters, subscription filters to Firehose/S3). Macie discovers and classifies sensitive data (PII) in S3, supporting data-protection governance beyond app logging.

**Gap:** You classify at emit-time in process. CloudWatch provides durable multi-account log architecture; Macie discovers sensitive data at rest that never appeared in your logs.

---

### Relational Storage and Vectors

| Your Component | AWS Equivalent | Exam Objectives |
|---|---|---|
| `DatabasePool` + pgvector (`packages/core/database/`, `infra/`) | Amazon RDS for PostgreSQL + pgvector | 3.2.2, 5.1.3 |

**What yours does:** Process-level `psycopg2` pool to Postgres for audit logs, pipeline jobs, prompt versions, and RAG vectors. Docker Compose runs `pgvector/pgvector:pg16` with init SQL under `infra/init-db/`.

**What RDS does:** Managed PostgreSQL with automated backups, Multi-AZ, patching, encryption at rest, and support for the pgvector extension for production vector workloads alongside OLTP tables.

**Gap:** You run a local container with default credentials. RDS adds HA, PITR, Parameter Groups, IAM DB auth, and operational tooling that exams frequently contrast with self-managed databases.

---

### Pipeline Orchestration

| Your Component | AWS Equivalent | Exam Objectives |
|---|---|---|
| Sequential Python pipelines (`packages/domain-*/pipeline.py`) | AWS Step Functions | 2.1.1, 3.2.1, 5.1.3 |

**What yours does:** Each domain runs specialist agents in a **hardcoded sequential order** (direct async method calls). Outputs from one agent feed the next; evaluators, guardrails, and HITL are composed in the pipeline file. `OrchestratorEngine` / `DAGExecutor` exist in core but are not the domain runtime.

**What Step Functions does:** Managed workflow orchestration with visual state machines, built-in error handling and retries, parallel branches, timeouts, and native integration with 200+ AWS services (including Bedrock, Lambda, SNS/SQS, and DynamoDB). Executions are observable and auditable at the state level.

**Gap:** Your pipelines are sequential Python with application-level try/except (Contract) and no visual monitor, automatic retry policy, or dynamic branching. Step Functions adds durable orchestration, service integrations, and operational visibility that this codebase does not implement.

---

### Local Containers and Deployment

| Your Component | AWS Equivalent | Exam Objectives |
|---|---|---|
| Docker Compose (`infra/docker-compose.yml` — Postgres + Redis) | Amazon ECS or Amazon EKS | 3.2.1, 5.1.1, 5.3.1 |

**What yours does:** Local development stack via Docker Compose: `pgvector/pgvector:pg16` and `redis:7-alpine`, with init SQL mounted for schema and audit triggers. Application code does not currently consume Redis.

**What ECS / EKS does:** Managed container orchestration for production. **ECS** suits simpler service topologies and pairs with Fargate for serverless compute. **EKS** runs Kubernetes-native workloads with cluster control-plane management by AWS. Both integrate with IAM, VPC networking, load balancing, and auto scaling.

**Gap:** Docker Compose is local-oriented: no multi-AZ auto scaling, service mesh, or production service discovery as first-class features. ECS/EKS provide production orchestration, deployment strategies, and cloud-native operations that Compose does not replace.

---

## Cross-Cutting Study Notes

1. **Abstraction pattern:** Almost every mapping uses an ABC or optional dependency (`ModelProvider`, `SecretsProvider`, `GuardrailsEngine`, `HITLAccessControl`). On the exam, prefer the managed service that matches the *capability*, then note how IAM/KMS/CloudTrail secure it.
2. **Responsible AI:** Guardrails + HITL + evaluator confidence thresholds mirror AWS guidance: filter content, escalate uncertain outputs to humans, and measure quality.
3. **Security defense in depth:** App insert-only audit + DB trigger ≈ application controls + preventive controls; Secrets Manager + KMS ≈ protect credentials; sensitivity logging ≈ data classification.
4. **Cost awareness:** Token metrics locally; Budgets/Cost Explorer and Bedrock provisioned throughput for production spend control.

---

## Services Not Mapped

AWS AI/ML services **not** represented in this codebase (useful for elimination-style exam questions):

| Service | One-line description |
|---|---|
| Amazon SageMaker | End-to-end platform to build, train, tune, deploy, and monitor custom ML models. |
| Amazon Personalize | Managed real-time personalization and recommendation based on user activity. |
| Amazon Forecast | Time-series forecasting service (legacy/maintenance posture; know the use case). |
| Amazon Lex | Conversational bots (ASR + NLU) for chat and voice interfaces. |
| Amazon Polly | Text-to-speech neural and standard voices. |
| Amazon Transcribe | Speech-to-text transcription and speaker diarization. |
| Amazon Translate | Neural machine translation between languages. |
| Amazon Kendra | Enterprise intelligent search over documents with relevance tuning. |
| Amazon Rekognition | Image and video analysis (faces, labels, moderation, text in image). |
| Amazon Fraud Detector | Managed fraud detection models for online activities. |
| AWS Panorama | Edge computer-vision appliance/SDK for on-premises cameras. |
| Amazon Lookout for Vision / Equipment / Metrics | Anomaly detection for industrial vision, equipment, and metrics (know family). |
| Amazon Q Business / Amazon Q Developer | Generative assistants grounded in enterprise data or developer workflows. |
| Amazon Bedrock Agents / Flows | Managed agent orchestration and visual generative workflows (beyond raw InvokeModel). |
| Amazon OpenSearch Service (standalone) | Search and analytics engine; often paired with RAG outside Knowledge Bases. |
| AWS Glue | Serverless ETL/catalog for preparing training or RAG corpora at scale. |

When a question asks for **custom model training at scale**, prefer SageMaker. When it asks for **recommendations**, prefer Personalize. When it asks for **enterprise document search** without you building RAG, prefer Kendra. When it asks for **FM access without managing infrastructure**, prefer Bedrock.
