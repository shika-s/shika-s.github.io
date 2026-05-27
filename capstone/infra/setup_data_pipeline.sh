#!/usr/bin/env bash
# =============================================================
# AWS Setup: Data Pipeline (ECS Fargate + EventBridge Scheduler)
# =============================================================
# Run each step manually — this script is a guide, not meant
# to be executed all at once. Review each command before running.
#
# Prerequisites:
#   - AWS CLI configured with profile "wfp"
#   - Docker installed
#   - Existing ECS cluster: wfp-ml-cluster
#   - AWS_ACCOUNT_ID env var set (e.g. export AWS_ACCOUNT_ID=123456789012)
#   - Existing S3 bucket: wfp-ml-pipeline-data-${AWS_ACCOUNT_ID}
#   - Existing IAM roles: wfp-ecs-execution-role, wfp-ecs-task-role
#
# After setup, the pipeline runs automatically:
#   EventBridge (1st of month) → ECS (data pipeline)
#     → uploads joined/ to S3
#     → S3 event triggers Lambda
#     → Lambda starts ML pipeline
# =============================================================

set -euo pipefail

PROFILE="wfp"
REGION="us-east-1"
ACCOUNT_ID="${AWS_ACCOUNT_ID:?ERROR: Set AWS_ACCOUNT_ID env var before running}"
ECR_REGISTRY="${ACCOUNT_ID}.dkr.ecr.${REGION}.amazonaws.com"

# ── Replace these with your actual values ──────────────────────
SUBNET="REPLACE_WITH_SUBNET_ID"
SECURITY_GROUP="REPLACE_WITH_SECURITY_GROUP_ID"

# ── Substitute __AWS_ACCOUNT_ID__ in JSON templates ───────────
echo "=== Replacing __AWS_ACCOUNT_ID__ with ${ACCOUNT_ID} in JSON files ==="
for f in "${SCRIPT_DIR}/task-definition-data.json" \
         "${SCRIPT_DIR}/eventbridge-schedule.json" \
         "${SCRIPT_DIR}/eventbridge-ecs-policy.json"; do
  sed -i.bak "s/__AWS_ACCOUNT_ID__/${ACCOUNT_ID}/g" "$f"
  rm -f "${f}.bak"
done

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

# ============================================================
# STEP 1: Store API credentials in Secrets Manager
# ============================================================
echo "=== Step 1: Create Secrets Manager secrets ==="
echo "Run:"
echo "  # FEWS NET credentials"
echo "  aws secretsmanager create-secret \\"
echo "    --name wfp/fewsnet-credentials \\"
echo "    --secret-string '{\"FEWSNET_USER\":\"YOUR_USER\",\"FEWSNET_PWD\":\"YOUR_PWD\"}' \\"
echo "    --region ${REGION} --profile ${PROFILE}"
echo ""
echo "  # ACLED credentials"
echo "  aws secretsmanager create-secret \\"
echo "    --name wfp/acled-credentials \\"
echo "    --secret-string '{\"ACLED_EMAIL\":\"YOUR_EMAIL\",\"ACLED_PWD\":\"YOUR_PWD\"}' \\"
echo "    --region ${REGION} --profile ${PROFILE}"
echo ""

# ============================================================
# STEP 2: Create ECR repository
# ============================================================
echo "=== Step 2: Create ECR repository ==="
echo "Run:"
echo "  aws ecr create-repository \\"
echo "    --repository-name wfp-data-pipeline \\"
echo "    --region ${REGION} --profile ${PROFILE}"
echo ""

# ============================================================
# STEP 3: Build and push Docker image
# ============================================================
echo "=== Step 3: Build and push Docker image ==="
echo "Run:"
echo "  cd ${PROJECT_ROOT}"
echo ""
echo "  # Login to ECR"
echo "  aws ecr get-login-password --region ${REGION} --profile ${PROFILE} | \\"
echo "    docker login --username AWS --password-stdin ${ECR_REGISTRY}"
echo ""
echo "  # Build"
echo "  docker build -f Dockerfile.data -t wfp-data-pipeline ."
echo ""
echo "  # Tag and push"
echo "  docker tag wfp-data-pipeline:latest ${ECR_REGISTRY}/wfp-data-pipeline:latest"
echo "  docker push ${ECR_REGISTRY}/wfp-data-pipeline:latest"
echo ""

# ============================================================
# STEP 4: Create CloudWatch log group
# ============================================================
echo "=== Step 4: Create CloudWatch log group ==="
echo "Run:"
echo "  aws logs create-log-group \\"
echo "    --log-group-name /ecs/wfp-data-pipeline \\"
echo "    --region ${REGION} --profile ${PROFILE}"
echo ""

# ============================================================
# STEP 5: Register ECS task definition
# ============================================================
echo "=== Step 5: Register ECS task definition ==="
echo "Run:"
echo "  aws ecs register-task-definition \\"
echo "    --cli-input-json file://infra/task-definition-data.json \\"
echo "    --region ${REGION} --profile ${PROFILE}"
echo ""

# ============================================================
# STEP 6: Test ECS task manually
# ============================================================
echo "=== Step 6: Test ECS task manually ==="
echo "Run:"
echo "  aws ecs run-task \\"
echo "    --cluster wfp-ml-cluster \\"
echo "    --task-definition wfp-data-pipeline \\"
echo "    --launch-type FARGATE \\"
echo "    --network-configuration '{\"awsvpcConfiguration\":{\"subnets\":[\"${SUBNET}\"],\"securityGroups\":[\"${SECURITY_GROUP}\"],\"assignPublicIp\":\"ENABLED\"}}' \\"
echo "    --overrides '{\"containerOverrides\":[{\"name\":\"wfp-data-pipeline\",\"command\":[\"poetry\",\"run\",\"python\",\"scripts/run_data_pipeline.py\",\"--countries\",\"cmr\"]}]}' \\"
echo "    --region ${REGION} --profile ${PROFILE}"
echo ""
echo "  # Monitor logs:"
echo "  aws logs tail /ecs/wfp-data-pipeline --follow --profile ${PROFILE}"
echo ""

# ============================================================
# STEP 7: Create IAM role for EventBridge Scheduler
# ============================================================
echo "=== Step 7: Create EventBridge Scheduler IAM role ==="
echo "Run:"
echo "  aws iam create-role \\"
echo "    --role-name wfp-eventbridge-scheduler-role \\"
echo "    --assume-role-policy-document file://infra/eventbridge-trust-policy.json \\"
echo "    --profile ${PROFILE}"
echo ""
echo "  aws iam put-role-policy \\"
echo "    --role-name wfp-eventbridge-scheduler-role \\"
echo "    --policy-name ecs-run-task \\"
echo "    --policy-document file://infra/eventbridge-ecs-policy.json \\"
echo "    --profile ${PROFILE}"
echo ""

# ============================================================
# STEP 8: Create EventBridge schedule (monthly)
# ============================================================
echo "=== Step 8: Create EventBridge schedule ==="
echo ""
echo "First, update infra/eventbridge-schedule.json:"
echo "  - Replace REPLACE_WITH_SUBNET_ID with: ${SUBNET}"
echo "  - Replace REPLACE_WITH_SECURITY_GROUP_ID with: ${SECURITY_GROUP}"
echo ""
echo "Then run:"
echo "  aws scheduler create-schedule \\"
echo "    --name wfp-data-pipeline-monthly \\"
echo "    --schedule-expression 'cron(0 6 1 * ? *)' \\"
echo "    --target file://infra/eventbridge-schedule.json \\"
echo "    --flexible-time-window '{\"Mode\":\"OFF\"}' \\"
echo "    --state ENABLED \\"
echo "    --region ${REGION} --profile ${PROFILE}"
echo ""
echo "  # Schedule: 1st of every month at 6:00 AM UTC"
echo ""

# ============================================================
# STEP 9: Verify setup
# ============================================================
echo "=== Step 9: Verify setup ==="
echo "Run:"
echo "  # Check task definition"
echo "  aws ecs describe-task-definition --task-definition wfp-data-pipeline --profile ${PROFILE}"
echo ""
echo "  # Check schedule"
echo "  aws scheduler get-schedule --name wfp-data-pipeline-monthly --profile ${PROFILE}"
echo ""
echo "  # Check secrets"
echo "  aws secretsmanager describe-secret --secret-id wfp/fewsnet-credentials --profile ${PROFILE}"
echo "  aws secretsmanager describe-secret --secret-id wfp/acled-credentials --profile ${PROFILE}"
echo ""

echo "=== Setup guide complete ==="
echo ""
echo "Pipeline flow after setup:"
echo "  EventBridge (1st of month, 6 AM UTC)"
echo "    → ECS Fargate (data pipeline: fetch → preprocess → merge → feature engineering)"
echo "    → Uploads joined/ + feature_engineering/ data to S3"
echo "    → S3 event triggers existing Lambda"
echo "    → Lambda starts ML pipeline on ECS"
