# ══════════════════════════════════════════════════════════════════════════════
# Z-Image Makefile
# ══════════════════════════════════════════════════════════════════════════════
SHELL := /bin/bash
-include .env
export CDK_DEFAULT_ACCOUNT
export CDK_DEFAULT_REGION
export AWS_REGION := $(CDK_DEFAULT_REGION)
export AWS_DEFAULT_REGION := $(CDK_DEFAULT_REGION)
export AWS_PAGER :=
ENV     := $(or $(ZIMAGE_ENV),dev)
ACCOUNT := $(CDK_DEFAULT_ACCOUNT)
REGION  := $(CDK_DEFAULT_REGION)

PLATFORM  := $(ENV)-zimg-platform
CICD      := $(ENV)-zimg-cicd
INFRA     := $(ENV)-zimg-inference-infra
SFN       := $(ENV)-zimg-compile-sfn
INFERENCE_ECS := $(ENV)-zimg-inference-ecs
WEB       := $(ENV)-zimg-web

.DEFAULT_GOAL := help

# ── Quality ──────────────────────────────────────────────────────────────────
.PHONY: check check-inference check-web lint lint-inference lint-web lint-frontend format \
        test test-inference test-web security security-web synth

check: check-inference check-web ## Both quality gates (inference + web)

check-inference: lint-inference test-inference security synth-inference ## Inference quality gate
	@echo "[check] inference quality gate passed"

check-web: lint-web test-web security-web synth-web ## Web quality gate
	@echo "[check] web quality gate passed"

lint: lint-inference lint-web lint-frontend ## Lint all code

lint-inference:                  ## Lint infra + runtime + tests (Python)
	python -m ruff check infra/ runtime/ tests/ --no-fix
	python -m ruff format --check infra/ runtime/ tests/
	@echo "[lint] scanning for dead code (vulture)..."
	@python -m vulture || { echo "[lint] FAILED: dead code found"; exit 1; }

lint-web:                        ## Lint web backend (Python)
	python -m ruff check web/backend/ tests/unit/web/ --no-fix
	python -m ruff format --check web/backend/ tests/unit/web/

lint-frontend:                   ## Lint web frontend (TypeScript)
	@cd web/frontend && npm run lint 2>/dev/null || { echo "[lint] frontend eslint failed"; exit 1; }

format:                          ## Auto-fix formatting + imports
	python -m ruff format infra/ runtime/ tests/ web/backend/
	python -m ruff check --fix --select I infra/ runtime/ tests/ web/backend/

test:                            ## Run all tests with coverage
	pytest -q --cov

test-inference:                  ## Run infra + runtime tests only
	pytest tests/unit/infra/ tests/unit/runtime/ tests/integration/ tests/cdk/ -q

test-web:                        ## Run web backend tests only
	pytest tests/unit/web/ -q

security:                        ## Security scan: full repo
	@echo "[security] running ASH scan (inference context)..."
	ash --mode local \
	  --source-dir . \
	  --output-dir .ash/out \
	  -c .ash/ash.yaml

security-web:                    ## Security scan (same config, web context)
	@echo "[security] running ASH scan (web context)..."
	ash --mode local \
	  --source-dir . \
	  --output-dir .ash/out-web \
	  -c .ash/ash.yaml

synth: synth-inference synth-web ## CDK synth all stacks

synth-inference:                 ## CDK synth inference stacks (+ cdk-nag)
	CDK_DEFAULT_ACCOUNT=$(ACCOUNT) CDK_DEFAULT_REGION=$(REGION) \
	npx cdk synth $(INFRA) $(SFN) $(INFERENCE_ECS) --quiet

synth-web:                       ## CDK synth web stack (+ cdk-nag)
	CDK_DEFAULT_ACCOUNT=$(ACCOUNT) CDK_DEFAULT_REGION=$(REGION) \
	npx cdk synth $(WEB) --quiet

# ── Deploy ───────────────────────────────────────────────────────────────────
.PHONY: deploy deploy-inference deploy-web bootstrap upload-contexts \
        run-inference-pipeline run-web-pipeline

deploy: deploy-inference deploy-web ## Deploy all: upload + trigger both pipelines

deploy-inference: upload-contexts ## Upload inference context + trigger pipeline
	@echo "[deploy] triggering inference pipeline..."
	$(MAKE) run-inference-pipeline

deploy-web:                      ## Upload web context + trigger pipeline
	@echo "[deploy] packaging and triggering web pipeline..."
	@mkdir -p "$$TMPDIR/zimage-contexts"
	@echo "[make] zipping web context..."
	@rm -f "$$TMPDIR/zimage-contexts/web.zip"
	@zip -rq "$$TMPDIR/zimage-contexts/web.zip" web/ infra/ runtime/lib/ tests/ .ash/ Makefile pyproject.toml cdk.json uv.lock \
	  -x "*/__pycache__/*" -x "*/node_modules/*" -x ".venv/*" -x "cdk.out/*" -x "cdk.out.*/*"
	@printf 'CDK_DEFAULT_ACCOUNT=$(ACCOUNT)\nCDK_DEFAULT_REGION=$(REGION)\nZIMAGE_ENV=$(ENV)\n' > "$$TMPDIR/zimage-contexts/.env"
	@cd "$$TMPDIR/zimage-contexts" && zip -q web.zip .env && rm .env
	@SOURCE_BUCKET=$$(aws cloudformation describe-stacks --stack-name $(CICD) \
	  --query 'Stacks[0].Outputs[?OutputKey==`SourceBucketName`].OutputValue' --output text); \
	echo "[make] uploading web to s3://$$SOURCE_BUCKET/build-contexts/web/context.zip"; \
	aws s3 cp "$$TMPDIR/zimage-contexts/web.zip" \
	  s3://$$SOURCE_BUCKET/build-contexts/web/context.zip --only-show-errors
	$(MAKE) run-web-pipeline

bootstrap:                       ## One-time: CDK deploy platform + cicd stacks
	CDK_DEFAULT_ACCOUNT=$(ACCOUNT) CDK_DEFAULT_REGION=$(REGION) \
	npx cdk deploy $(PLATFORM) $(CICD) --require-approval never --exclusively

upload-contexts:                 ## Zip and upload inference contexts (runtime + cdk)
	@mkdir -p "$$TMPDIR/zimage-contexts"
	@echo "[make] zipping runtime context..."
	@(cd runtime && zip -rq "$$TMPDIR/zimage-contexts/runtime.zip" . -x "*/__pycache__/*")
	@echo "[make] zipping cdk source..."
	@zip -rq "$$TMPDIR/zimage-contexts/cdk.zip" . \
	  -x "*.pyc" -x "*__pycache__*" -x ".venv/*" -x "cdk.out/*" -x "cdk.out.*/*" -x "node_modules/*" \
	  -x ".git/*" -x ".coverage" -x ".mypy_cache/*" -x ".idea/*" -x "*.md" -x ".env"
	@printf 'CDK_DEFAULT_ACCOUNT=$(ACCOUNT)\nCDK_DEFAULT_REGION=$(REGION)\nZIMAGE_ENV=$(ENV)\n' > "$$TMPDIR/zimage-contexts/.env"
	@cd "$$TMPDIR/zimage-contexts" && zip -q cdk.zip .env && rm .env
	@SOURCE_BUCKET=$$(aws cloudformation describe-stacks --stack-name $(CICD) \
	  --query 'Stacks[0].Outputs[?OutputKey==`SourceBucketName`].OutputValue' --output text); \
	[ -z "$$SOURCE_BUCKET" ] && { echo "ERROR: $(CICD) not deployed"; exit 1; } || \
	for target in runtime cdk; do \
	  echo "[make] uploading $$target to s3://$$SOURCE_BUCKET/build-contexts/$$target/context.zip"; \
	  aws s3 cp "$$TMPDIR/zimage-contexts/$$target.zip" \
	    s3://$$SOURCE_BUCKET/build-contexts/$$target/context.zip --only-show-errors; \
	done

run-inference-pipeline:          ## Trigger inference pipeline
	@PIPELINE_NAME=$(ENV)-zimg-inference-pipeline; \
	echo "[make] pipeline: $$PIPELINE_NAME"; \
	EXEC_ID=$$(aws codepipeline start-pipeline-execution --name $$PIPELINE_NAME \
	  --query 'pipelineExecutionId' --output text); \
	echo "[make] execution: $$EXEC_ID"; \
	echo "[make] https://$(REGION).console.aws.amazon.com/codesuite/codepipeline/pipelines/$$PIPELINE_NAME/executions/$$EXEC_ID/visualization"

run-web-pipeline:                ## Trigger web pipeline
	@PIPELINE_NAME=$(ENV)-zimg-web-pipeline; \
	EXEC_ID=$$(aws codepipeline start-pipeline-execution --name $$PIPELINE_NAME \
	  --query 'pipelineExecutionId' --output text); \
	echo "[make] pipeline: $$PIPELINE_NAME"; \
	echo "[make] execution: $$EXEC_ID"; \
	echo "[make] https://$(REGION).console.aws.amazon.com/codesuite/codepipeline/pipelines/$$PIPELINE_NAME/executions/$$EXEC_ID/visualization"

# ── Compile (Step Functions) ─────────────────────────────────────────────────
.PHONY: compile-models wait-compile warmup-768 warmup-1024 warmup-tail

warmup-768:                      ## Submit warmup-768 Batch job directly (skip SFN)
	@JOB_ID=$$(aws batch submit-job \
	  --job-name "manual-warmup-768-$$(date +%H%M%S)" \
	  --job-queue $(ENV)-zimg-warmup-jq \
	  --job-definition $(ENV)-zimg-warmup-jd \
	  --parameters resolution=768,artifacts_bucket=$(ENV)-zimg-artifacts-$(ACCOUNT)-$(REGION) \
	  --query 'jobId' --output text); \
	echo "[warmup-768] jobId=$$JOB_ID"; \
	echo "$$JOB_ID" > /tmp/zimage-warmup-768

warmup-1024:                     ## Submit warmup-1024 Batch job directly (skip SFN)
	@JOB_ID=$$(aws batch submit-job \
	  --job-name "manual-warmup-1024-$$(date +%H%M%S)" \
	  --job-queue $(ENV)-zimg-warmup-jq \
	  --job-definition $(ENV)-zimg-warmup-jd \
	  --parameters resolution=1024,artifacts_bucket=$(ENV)-zimg-artifacts-$(ACCOUNT)-$(REGION) \
	  --query 'jobId' --output text); \
	echo "[warmup-1024] jobId=$$JOB_ID"; \
	echo "$$JOB_ID" > /tmp/zimage-warmup-1024

warmup-tail:                     ## Tail last submitted warmup job logs
	@JOB_ID=$$(cat /tmp/zimage-warmup-768 2>/dev/null || cat /tmp/zimage-warmup-1024 2>/dev/null); \
	[ -n "$$JOB_ID" ] || { echo "no warmup job recorded"; exit 1; }; \
	echo "[warmup-tail] jobId=$$JOB_ID"; \
	for i in $$(seq 1 60); do \
	  STATUS=$$(aws batch describe-jobs --jobs $$JOB_ID --region $(REGION) --query 'jobs[0].status' --output text); \
	  echo "[$$i/60] $$(date +%H:%M:%S) status=$$STATUS"; \
	  case $$STATUS in SUCCEEDED|FAILED) break ;; esac; \
	  sleep 30; \
	done; \
	STREAM=$$(aws batch describe-jobs --jobs $$JOB_ID --region $(REGION) --query 'jobs[0].container.logStreamName' --output text); \
	[ "$$STREAM" = "None" ] || aws logs get-log-events --log-group-name /aws/batch/job --log-stream-name "$$STREAM" --region $(REGION) --query 'events[-30:].message' --output text

compile-models:                  ## Start compile state machine execution
	@EXEC_NAME=$$(python3 -c "import uuid;print(uuid.uuid4())"); \
	aws stepfunctions start-execution \
	  --state-machine-arn arn:aws:states:$(REGION):$(ACCOUNT):stateMachine:$(SFN) \
	  --name $$EXEC_NAME --input '{}' \
	  --query 'executionArn' --output text | tee /tmp/zimage-last-sfn-exec

wait-compile:                    ## Wait for current SFN execution to finish
	@EXEC_ARN=$$(cat /tmp/zimage-last-sfn-exec 2>/dev/null || \
	  aws stepfunctions list-executions \
	    --state-machine-arn arn:aws:states:$(REGION):$(ACCOUNT):stateMachine:$(SFN) \
	    --max-items 1 --query 'executions[0].executionArn' --output text); \
	for i in $$(seq 1 90); do \
	  STATUS=$$(aws stepfunctions describe-execution \
	    --execution-arn $$EXEC_ARN --query 'status' --output text); \
	  echo "[$$i/90] $$(date +%H:%M:%S) $$STATUS"; \
	  case $$STATUS in SUCCEEDED|FAILED|TIMED_OUT|ABORTED) break ;; esac; \
	  sleep 60; \
	done; \
	[ "$$STATUS" = "SUCCEEDED" ] || { echo "[make] SFN failed"; exit 1; }

# ── Local Dev ────────────────────────────────────────────────────────────────
.PHONY: dev dev-stop

dev-stop:                        ## Kill dev server processes
	@-lsof -ti :8000 | xargs kill 2>/dev/null || true
	@-lsof -ti :3000 | xargs kill 2>/dev/null || true
	@echo "[dev-stop] done"

dev:                             ## Start local dev server (auth bypassed, mock inference)
	@uv sync --quiet
	@cd web/frontend && npm install --silent
	@-lsof -ti :8000 | xargs kill 2>/dev/null || true
	@-lsof -ti :3000 | xargs kill 2>/dev/null || true
	@sleep 1
	@echo "[dev] starting backend (port 8000) + frontend (port 3000)..."
	@echo "[dev] auth bypassed, mock inference (ZIMAGE_LOCAL=true)"
	@cd web/backend && ZIMAGE_LOCAL=true PYTHONUNBUFFERED=1 \
	  python -m uvicorn server:app --reload --port 8000 & \
	  cd web/frontend && npx vite --port 3000 --strictPort & \
	  wait

# ── Users ─────────────────────────────────────────────────────────────────────
.PHONY: url create-admin create-user

url:                             ## Show deployed web app URL
	@AWS_PAGER="" aws cloudformation describe-stacks --stack-name $(WEB) --region $(REGION) \
	  --query 'Stacks[0].Outputs[?OutputKey==`WebUrl`].OutputValue' --output text 2>/dev/null \
	  || echo "Web stack not deployed"

create-admin:                    ## Create admin user: make create-admin EMAIL=you@example.com
	@[ -n "$(EMAIL)" ] || { echo "Usage: make create-admin EMAIL=you@example.com"; exit 1; }
	@PASSWORD=$$(openssl rand -base64 12)A1! && \
	POOL_ID=$$(AWS_PAGER="" aws cloudformation describe-stacks --stack-name $(WEB) --region $(REGION) \
	  --query 'Stacks[0].Outputs[?OutputKey==`UserPoolId`].OutputValue' --output text) && \
	aws cognito-idp admin-create-user --user-pool-id $$POOL_ID --username "$(EMAIL)" \
	  --user-attributes Name=email,Value="$(EMAIL)" Name=email_verified,Value=true \
	  --message-action SUPPRESS --region $(REGION) 2>/dev/null || true && \
	aws cognito-idp admin-set-user-password --user-pool-id $$POOL_ID \
	  --username "$(EMAIL)" --password "$$PASSWORD" --permanent --region $(REGION) && \
	aws cognito-idp admin-add-user-to-group --user-pool-id $$POOL_ID \
	  --username "$(EMAIL)" --group-name admins --region $(REGION) && \
	aws cognito-idp admin-add-user-to-group --user-pool-id $$POOL_ID \
	  --username "$(EMAIL)" --group-name users --region $(REGION) && \
	echo "[create-admin] done: $(EMAIL) added to admins + users groups" && \
	echo "[create-admin] password: $$PASSWORD"

create-user:                     ## Create non-admin user: make create-user EMAIL=user@example.com
	@[ -n "$(EMAIL)" ] || { echo "Usage: make create-user EMAIL=user@example.com"; exit 1; }
	@PASSWORD=$$(openssl rand -base64 12)A1! && \
	POOL_ID=$$(AWS_PAGER="" aws cloudformation describe-stacks --stack-name $(WEB) --region $(REGION) \
	  --query 'Stacks[0].Outputs[?OutputKey==`UserPoolId`].OutputValue' --output text) && \
	aws cognito-idp admin-create-user --user-pool-id $$POOL_ID --username "$(EMAIL)" \
	  --user-attributes Name=email,Value="$(EMAIL)" Name=email_verified,Value=true \
	  --message-action SUPPRESS --region $(REGION) 2>/dev/null || true && \
	aws cognito-idp admin-set-user-password --user-pool-id $$POOL_ID \
	  --username "$(EMAIL)" --password "$$PASSWORD" --permanent --region $(REGION) && \
	aws cognito-idp admin-add-user-to-group --user-pool-id $$POOL_ID \
	  --username "$(EMAIL)" --group-name users --region $(REGION) && \
	echo "[create-user] done: $(EMAIL) added to users group (non-admin)" && \
	echo "[create-user] password: $$PASSWORD"


# ── Teardown ─────────────────────────────────────────────────────────────────
.PHONY: destroy force-destroy clean-retained

destroy: _pre-destroy            ## Destroy all stacks (reversed order)
	CDK_DEFAULT_ACCOUNT=$(ACCOUNT) CDK_DEFAULT_REGION=$(REGION) \
	npx cdk destroy --all --force --output cdk.out.$(ENV) || { \
	  rm -rf cdk.out.$(ENV); \
	  echo "[destroy] cdk destroy failed; skipping clean-retained so live stacks keep their resources."; \
	  echo "[destroy] fix the error above (or use 'make force-destroy') and re-run."; \
	  exit 1; }
	@rm -rf cdk.out.$(ENV)
	$(MAKE) clean-retained

force-destroy: _pre-destroy      ## Force delete all stacks via CloudFormation
	@for stack in $(WEB) $(INFERENCE_ECS) $(SFN) $(INFRA) $(CICD) $(PLATFORM); do \
	  echo "[make] force-deleting $$stack..."; \
	  aws cloudformation delete-stack --stack-name $$stack --region $(REGION) 2>/dev/null || true; \
	  aws cloudformation wait stack-delete-complete --stack-name $$stack --region $(REGION) 2>/dev/null || true; \
	done
	$(MAKE) clean-retained

_pre-destroy:                    ## Disassociate ECS capacity providers + delete services before stack destroy
	@CLUSTER=$(ENV)-zimg-inference-cluster; \
	echo "[pre-destroy] disassociating capacity providers from $$CLUSTER..."; \
	aws ecs put-cluster-capacity-providers --cluster $$CLUSTER \
	  --capacity-providers '[]' --default-capacity-provider-strategy '[]' \
	  --region $(REGION) >/dev/null 2>&1 || true; \
	SVCS=$$(aws ecs list-services --cluster $$CLUSTER --region $(REGION) \
	  --query 'serviceArns[]' --output text 2>/dev/null); \
	if [ -n "$$SVCS" ] && [ "$$SVCS" != "None" ]; then \
	  for svc in $$SVCS; do \
	    echo "[pre-destroy] deleting service $$(echo $$svc | sed 's|.*/||')..."; \
	    aws ecs update-service --cluster $$CLUSTER --service $$svc --desired-count 0 \
	      --region $(REGION) >/dev/null 2>&1 || true; \
	    aws ecs delete-service --cluster $$CLUSTER --service $$svc --force \
	      --region $(REGION) >/dev/null 2>&1 || true; \
	  done; \
	  echo "[pre-destroy] waiting for services to become inactive..."; \
	  aws ecs wait services-inactive --cluster $$CLUSTER --services $$SVCS \
	    --region $(REGION) 2>/dev/null || true; \
	fi; \
	echo "[pre-destroy] deleting cluster $$CLUSTER..."; \
	aws ecs delete-cluster --cluster $$CLUSTER --region $(REGION) >/dev/null 2>&1 || true; \
	echo "[pre-destroy] cleaning up VPC ENIs..."; \
	VPC_ID=$$(aws cloudformation describe-stacks --stack-name $(PLATFORM) --region $(REGION) \
	  --query 'Stacks[0].Outputs[?OutputKey==`VpcId`].OutputValue' --output text 2>/dev/null); \
	if [ -n "$$VPC_ID" ] && [ "$$VPC_ID" != "None" ]; then \
	  for eni in $$(aws ec2 describe-network-interfaces \
	    --filters Name=vpc-id,Values=$$VPC_ID Name=status,Values=available \
	    --query 'NetworkInterfaces[].NetworkInterfaceId' --output text --region $(REGION) 2>/dev/null); do \
	    echo "[pre-destroy] deleting ENI $$eni..."; \
	    aws ec2 delete-network-interface --network-interface-id $$eni --region $(REGION) 2>/dev/null || true; \
	  done; \
	fi; \
	echo "[pre-destroy] scaling ASGs to 0..."; \
	for asg in $$(aws autoscaling describe-auto-scaling-groups --region $(REGION) \
	  --query "AutoScalingGroups[?contains(AutoScalingGroupName,'$(ENV)-zimg')].AutoScalingGroupName" \
	  --output text 2>/dev/null); do \
	  aws autoscaling update-auto-scaling-group --auto-scaling-group-name $$asg \
	    --min-size 0 --desired-capacity 0 --region $(REGION) 2>/dev/null || true; \
	done; \
	echo "[pre-destroy] terminating EC2 instances..."; \
	INSTANCES=$$(aws ec2 describe-instances \
	  --filters Name=vpc-id,Values=$$VPC_ID Name=instance-state-name,Values=running,shutting-down,stopping \
	  --query 'Reservations[].Instances[].InstanceId' --output text --region $(REGION) 2>/dev/null); \
	if [ -n "$$INSTANCES" ] && [ "$$INSTANCES" != "None" ]; then \
	  aws ec2 terminate-instances --instance-ids $$INSTANCES --region $(REGION) >/dev/null 2>&1 || true; \
	  echo "[pre-destroy] waiting for instances to terminate..."; \
	  aws ec2 wait instance-terminated --instance-ids $$INSTANCES --region $(REGION) 2>/dev/null || true; \
	fi; \
	echo "[pre-destroy] done."

clean-retained:                  ## Clean any orphaned resources (tagged App=z-image-inferentia)
	@echo "[clean] ECR repos..."
	@for arn in $$(aws resourcegroupstaggingapi get-resources \
	  --tag-filters Key=App,Values=z-image-inferentia-$(ENV) \
	  --resource-type-filters ecr:repository \
	  --query 'ResourceTagMappingList[].ResourceARN' --output text --region $(REGION) 2>/dev/null); do \
	  repo=$$(echo $$arn | sed 's|.*/||'); \
	  aws ecr delete-repository --repository-name $$repo --force --region $(REGION) 2>/dev/null && \
	  echo "  deleted $$repo" || true; \
	done
	@echo "[clean] S3 buckets..."
	@for arn in $$(aws resourcegroupstaggingapi get-resources \
	  --tag-filters Key=App,Values=z-image-inferentia-$(ENV) \
	  --resource-type-filters s3 \
	  --query 'ResourceTagMappingList[].ResourceARN' --output text --region $(REGION) 2>/dev/null); do \
	  bucket=$$(echo $$arn | sed 's|.*:::||'); \
	  aws s3 rm s3://$$bucket --recursive --region $(REGION) 2>/dev/null; \
	  aws s3api delete-bucket --bucket $$bucket --region $(REGION) 2>/dev/null && \
	  echo "  deleted $$bucket" || true; \
	done
	@echo "[clean] Log groups..."
	@for arn in $$(aws resourcegroupstaggingapi get-resources \
	  --tag-filters Key=App,Values=z-image-inferentia-$(ENV) \
	  --resource-type-filters logs:log-group \
	  --query 'ResourceTagMappingList[].ResourceARN' --output text --region $(REGION) 2>/dev/null); do \
	  lg=$$(echo $$arn | sed 's|.*:log-group:||'); \
	  aws logs delete-log-group --log-group-name $$lg --region $(REGION) 2>/dev/null && \
	  echo "  deleted $$lg" || true; \
	done
	@echo "[clean] NAT gateways..."
	@for arn in $$(aws resourcegroupstaggingapi get-resources \
	  --tag-filters Key=App,Values=z-image-inferentia-$(ENV) \
	  --resource-type-filters ec2:natgateway \
	  --query 'ResourceTagMappingList[].ResourceARN' --output text --region $(REGION) 2>/dev/null); do \
	  natid=$$(echo $$arn | sed 's|.*/||'); \
	  aws ec2 delete-nat-gateway --nat-gateway-id $$natid --region $(REGION) 2>/dev/null && \
	  echo "  deleted $$natid" || true; \
	done
	@echo "[clean] VPC endpoints..."
	@for arn in $$(aws resourcegroupstaggingapi get-resources \
	  --tag-filters Key=App,Values=z-image-inferentia-$(ENV) \
	  --resource-type-filters ec2:vpc-endpoint \
	  --query 'ResourceTagMappingList[].ResourceARN' --output text --region $(REGION) 2>/dev/null); do \
	  vpce=$$(echo $$arn | sed 's|.*/||'); \
	  aws ec2 delete-vpc-endpoints --vpc-endpoint-ids $$vpce --region $(REGION) 2>/dev/null && \
	  echo "  deleted $$vpce" || true; \
	done
	@echo "[clean] VPC flow logs..."
	@for arn in $$(aws resourcegroupstaggingapi get-resources \
	  --tag-filters Key=App,Values=z-image-inferentia-$(ENV) \
	  --resource-type-filters ec2:vpc-flow-log \
	  --query 'ResourceTagMappingList[].ResourceARN' --output text --region $(REGION) 2>/dev/null); do \
	  flid=$$(echo $$arn | sed 's|.*/||'); \
	  aws ec2 delete-flow-logs --flow-log-ids $$flid --region $(REGION) 2>/dev/null && \
	  echo "  deleted $$flid" || true; \
	done
	@echo "[clean] KMS keys (7-day scheduled deletion)..."
	@for arn in $$(aws resourcegroupstaggingapi get-resources \
	  --tag-filters Key=App,Values=z-image-inferentia-$(ENV) \
	  --resource-type-filters kms:key \
	  --query 'ResourceTagMappingList[].ResourceARN' --output text --region $(REGION) 2>/dev/null); do \
	  aws kms schedule-key-deletion --key-id $$arn --pending-window-in-days 7 --region $(REGION) 2>/dev/null && \
	  echo "  scheduled $$arn" || true; \
	done
	@echo "[clean] Deactivating Batch job definitions..."
	@for arn in $$(aws resourcegroupstaggingapi get-resources \
	  --tag-filters Key=App,Values=z-image-inferentia-$(ENV) \
	  --resource-type-filters batch:job-definition \
	  --query 'ResourceTagMappingList[].ResourceARN' --output text --region $(REGION) 2>/dev/null); do \
	  aws batch deregister-job-definition --job-definition $$arn --region $(REGION) 2>/dev/null && \
	  echo "  deregistered $$arn" || true; \
	done
	@echo "[clean] done."

# ── Housekeeping ─────────────────────────────────────────────────────────────
.PHONY: clean help

clean:                           ## Remove build artifacts
	rm -rf cdk.out "$$TMPDIR/zimage-contexts"
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true

help:                            ## Show this help
	@printf "\n\033[1m── Quality ──\033[0m\n"
	@printf "  \033[36m%-20s\033[0m %s\n" "check" "Both quality gates (inference + web)"
	@printf "  \033[36m%-20s\033[0m %s\n" "check-inference" "Inference: lint + test + security + synth"
	@printf "  \033[36m%-20s\033[0m %s\n" "check-web" "Web: lint + test + security"
	@printf "  \033[36m%-20s\033[0m %s\n" "lint" "Lint all (inference + web)"
	@printf "  \033[36m%-20s\033[0m %s\n" "lint-inference" "Lint infra + runtime + tests"
	@printf "  \033[36m%-20s\033[0m %s\n" "lint-web" "Lint web backend"
	@printf "  \033[36m%-20s\033[0m %s\n" "test" "Test all (inference + web)"
	@printf "  \033[36m%-20s\033[0m %s\n" "test-inference" "Test infra + runtime"
	@printf "  \033[36m%-20s\033[0m %s\n" "test-web" "Test web backend"
	@printf "  \033[36m%-20s\033[0m %s\n" "security" "Security scan (full repo)"
	@printf "  \033[36m%-20s\033[0m %s\n" "format" "Auto-fix formatting"
	@printf "  \033[36m%-20s\033[0m %s\n" "synth" "CDK synth (templates + cdk-nag)"
	@printf "\n\033[1m── Deploy ──\033[0m\n"
	@printf "  \033[36m%-20s\033[0m %s\n" "deploy" "Deploy all (CDK + inference + web)"
	@printf "  \033[36m%-20s\033[0m %s\n" "deploy-inference" "CDK + inference pipeline"
	@printf "  \033[36m%-20s\033[0m %s\n" "deploy-web" "Web pipeline only (fast, no CDK)"
	@printf "\n\033[1m── Compile ──\033[0m\n"
	@printf "  \033[36m%-20s\033[0m %s\n" "compile-models" "Start compile state machine"
	@printf "  \033[36m%-20s\033[0m %s\n" "wait-compile" "Wait for SFN execution"
	@printf "\n\033[1m── Local Dev ──\033[0m\n"
	@printf "  \033[36m%-20s\033[0m %s\n" "dev" "Start local dev (mock inference, auth bypassed)"
	@printf "\n\033[1m── Users ──\033[0m\n"
	@printf "  \033[36m%-20s\033[0m %s\n" "create-admin" "Create admin user (auto-generates password)"
	@printf "  \033[36m%-20s\033[0m %s\n" "create-user" "Create non-admin user (auto-generates password)"
	@printf "  \033[36m%-20s\033[0m %s\n" "url" "Show deployed web app URL"
	@printf "\n\033[1m── Teardown ──\033[0m\n"
	@printf "  \033[36m%-20s\033[0m %s\n" "destroy" "cdk destroy all stacks"
	@printf "  \033[36m%-20s\033[0m %s\n" "force-destroy" "cfn delete-stack all stacks"
	@printf "  \033[36m%-20s\033[0m %s\n" "clean-retained" "Delete tagged resources"
	@printf "  \033[36m%-20s\033[0m %s\n" "clean" "Remove cdk.out/ + __pycache__/"
	@printf "\n"
