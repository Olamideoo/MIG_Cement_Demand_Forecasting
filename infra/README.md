# Infrastructure

Terraform (or CDK) for the AWS deployment described in WORKFLOW.md Phase 8.

Path A - ECS Fargate (recommended):

- ECR repositories for the api and dashboard images
- ECS Fargate services behind an ALB, path routing `/api/*` -> api, `/*` -> dashboard
- S3 for MLflow artifacts and processed data
- RDS Postgres (or SQLite on EFS) for the MLflow backend store
- Secrets Manager for credentials
- CloudWatch log groups and alarms on 5xx rate and task health

Nothing here yet - Phase 8.

Cost guardrail: set a budget alarm before the first `terraform apply`, and keep
the teardown command documented at the top of this file once it exists.
