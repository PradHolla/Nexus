  ## Nexus Productionization Plan (AWS Hybrid, FERPA-Ready, Pilot-Gated)

  ### Summary

  - Build a single-region, production architecture on AWS that is campus-scale capable but pilot-gated at launch.
  - Keep current frontend contract backward compatible while replacing in-process orchestration/state with managed services.
  - Target: 99.5% availability, 200 concurrent quiz generations, department-level isolation, hard cost guardrails, 8-12 week pilot.

  ### Implementation Changes

  - Runtime topology
      - Frontend: S3 + CloudFront.
      - API/SSE service: FastAPI on ECS Fargate behind ALB (keep SSE for quiz generation).
      - Ingestion orchestration: SQS + Step Functions (Standard) with ECS Fargate workers for CPU/ML-heavy parsing, embedding, and reranking.
      - Use Lambda only for lightweight control tasks (event adapters, scheduled cleanup, notifications), not heavy document/ML execution.
  - State and data plane
      - Replace jobs.db (SQLite) with RDS PostgreSQL for job state, audit records, tenant metadata, and quotas.
      - Move file storage to S3 (raw uploads + normalized chunk artifacts).
      - Move vector storage to Qdrant Cloud (managed), per-tenant namespace strategy with department-level isolation and course partition keys.
      - Add ElastiCache Redis for request throttling tokens, short-lived quiz generation state, and hot metadata cache.
  - Auth, tenancy, compliance
      - University SSO (OIDC/SAML) + JWT validation at API layer; RBAC roles (faculty, dept_admin, platform_admin).
      - Enforce tenant scoping in every query path (metadata filters + DB row-level policy).
      - FERPA-ready controls: encryption in transit/at rest (KMS), immutable audit logs, retention/deletion policies, least-privilege IAM, secrets in AWS Secrets Manager.
  - ML/RAG hardening
      - Introduce model router with primary+fallback providers and per-model timeout/retry/circuit-breaker policies.
      - Add deterministic prompt/version registry and evaluation gates (context precision, faithfulness, answer relevance thresholds).
      - Add async dead-letter handling for failed ingestions and quarantine workflow persistence in PostgreSQL.
  - Observability and ops
      - OpenTelemetry traces, structured logs, and metrics to CloudWatch/Grafana (latency, queue lag, token/cost, failure classes).
      - SLO dashboards and alerting with error budgets.
      - CI/CD with IaC (Terraform), blue/green deploys for ECS, migration pipeline for DB schema changes.
      - Cost controls: per-department quotas, request rate caps, monthly budget alarms, graceful degradation under budget pressure.

  ### Public API / Interface Changes

  - Preserve existing endpoints and event semantics for:
      - POST /api/ingest
      - GET /api/jobs/{job_id}
      - GET /api/courses/{course_id}/files
      - POST /api/generate-quiz (SSE)
  - Add non-breaking preferred v1 endpoints:
      - POST /api/v1/uploads/presign (direct S3 upload)
      - POST /api/v1/ingestions (submit object keys + metadata)
      - GET /api/v1/jobs/{job_id} (normalized status model)
  - Event contract additions (non-breaking):
      - Add correlation_id, tenant_id, cost_bucket, and trace_id fields to logs/events for observability and audit.

  ### Test Plan and Acceptance Criteria

  - Load tests:
      - Sustain 200 concurrent quiz generations with bounded p95 latency target and no cross-tenant leakage.
      - Ingestion burst tests with queue backpressure and worker autoscaling.
  - Resilience tests:
      - Provider timeout/failure drills (primary model down -> fallback).
      - Worker crash/retry/idempotency validation and DLQ replay.
  - Security/compliance tests:
      - RBAC negative tests, tenant isolation tests, audit log completeness, secrets/key rotation checks.
  - Quality tests:
      - Automated RAG eval regression suite with minimum thresholds for faithfulness/relevance before release.
  - Release gates:
      - Pilot go-live only after SLO dashboards, alerting, runbooks, and on-call readiness are validated in staging.

  ### Assumptions and Defaults

  - Cloud/platform: AWS-first, single region (us-east-1 unless changed).
  - Compute strategy: hybrid serverless + containers (not Kubernetes for v1).
  - Vector store: managed Qdrant Cloud (not self-hosted Qdrant).
  - Compatibility: current frontend behavior remains supported during migration.
  - Launch strategy: scale-ready architecture with pilot traffic gating, then expand.