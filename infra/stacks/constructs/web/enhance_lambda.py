from __future__ import annotations

from pathlib import Path

from aws_cdk import Duration, RemovalPolicy
from aws_cdk import aws_iam as iam
from aws_cdk import aws_lambda as lambda_
from aws_cdk import aws_logs as logs
from cdk_nag import NagSuppressions
from constructs import Construct

LAMBDA_DIR = Path(__file__).resolve().parents[3] / "lambdas" / "enhance_prompt"


class EnhancePromptLambda(Construct):
    def __init__(
        self,
        scope: Construct,
        construct_id: str,
        *,
        prefix: str,
        region: str,
        model_id: str = "qwen.qwen3-32b-v1:0",
    ) -> None:
        super().__init__(scope, construct_id)

        log_group = logs.LogGroup(
            self,
            "Logs",
            log_group_name=f"/aws/lambda/{prefix}-enhance-prompt",
            retention=logs.RetentionDays.ONE_MONTH,
            removal_policy=RemovalPolicy.DESTROY,
        )

        from infra.stacks.constructs.web.lambda_policies import lambda_execution_policies

        self.role = iam.Role(
            self,
            "Role",
            role_name=f"{prefix}-enhance-prompt-role",
            assumed_by=iam.ServicePrincipal("lambda.amazonaws.com"),
            managed_policies=lambda_execution_policies(self, log_group),
        )

        self.role.add_to_policy(
            iam.PolicyStatement(
                actions=["bedrock:InvokeModel", "bedrock:InvokeModelWithResponseStream"],
                resources=[
                    f"arn:aws:bedrock:{region}::foundation-model/{model_id}",
                ],
            )
        )

        self.function = lambda_.Function(
            self,
            "Function",
            function_name=f"{prefix}-enhance-prompt",
            runtime=lambda_.Runtime.PYTHON_3_13,
            handler="index.handler",
            code=lambda_.Code.from_asset(str(LAMBDA_DIR)),
            role=self.role,
            memory_size=512,
            timeout=Duration.seconds(60),
            log_group=log_group,
            tracing=lambda_.Tracing.ACTIVE,
            reserved_concurrent_executions=10,
            environment={
                "ZIMAGE_REGION": region,
                "ENHANCE_MODEL_ID": model_id,
            },
        )

        NagSuppressions.add_resource_suppressions(
            self,
            [
                {
                    "id": "AwsSolutions-IAM5",
                    "reason": "X-Ray tracing requires Resource '*'; log group ARN uses :* suffix.",
                },
            ],
            apply_to_children=True,
        )
