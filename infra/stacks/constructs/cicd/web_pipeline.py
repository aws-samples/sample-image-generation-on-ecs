from __future__ import annotations

from aws_cdk import aws_codepipeline as codepipeline
from aws_cdk import aws_codepipeline_actions as codepipeline_actions
from aws_cdk import aws_iam as iam
from aws_cdk import aws_kms as kms
from aws_cdk import aws_s3 as s3
from cdk_nag import NagSuppressions
from constructs import Construct

from infra.stacks.constructs.cicd.quality_gate import (
    QualityGate,
    QualityGateTargets,
    _build_project,
)


class WebCicdPipeline(Construct):
    def __init__(
        self,
        scope: Construct,
        construct_id: str,
        *,
        pipeline_name: str,
        source_bucket: s3.IBucket,
        source_key: str,
        kms_key: kms.IKey,
        prefix: str,
        region: str,
        environment: str,
    ) -> None:
        super().__init__(scope, construct_id)

        self.quality_gate = QualityGate(
            self,
            "QualityGate",
            kms_key=kms_key,
            name_prefix=f"{prefix}-web-quality-gate",
            targets=QualityGateTargets(
                lint="lint-web",
                test="test-web",
                security="security-web",
                synth="synth-web",
            ),
        )

        deploy_project = _build_project(
            self,
            "DeployWeb",
            project_name=f"{prefix}-web-deploy",
            script_file="deploy_web.sh",
            kms_key=kms_key,
            timeout_minutes=30,
            env_vars={
                "REGION": region,
                "ZIMG_PREFIX": prefix,
                "ZIMAGE_ENV": environment,
            },
        )
        deploy_project.role.add_managed_policy(
            iam.ManagedPolicy(
                self,
                "DeployPolicy",
                managed_policy_name=f"{prefix}-web-deploy-policy",
                statements=[
                    iam.PolicyStatement(
                        sid="CdkBootstrapAssumeRole",
                        actions=["sts:AssumeRole"],
                        resources=[f"arn:aws:iam::*:role/cdk-*-{region}"],
                    ),
                    iam.PolicyStatement(
                        sid="StsGetCaller",
                        actions=["sts:GetCallerIdentity"],
                        resources=["*"],
                    ),
                    iam.PolicyStatement(
                        sid="S3SyncStaticBucket",
                        actions=[
                            "s3:PutObject",
                            "s3:DeleteObject",
                            "s3:ListBucket",
                            "s3:GetObject",
                        ],
                        resources=[
                            f"arn:aws:s3:::{prefix}-web-static-*",
                            f"arn:aws:s3:::{prefix}-web-static-*/*",
                        ],
                    ),
                    iam.PolicyStatement(
                        sid="CloudFrontInvalidation",
                        actions=[
                            "cloudfront:CreateInvalidation",
                            "cloudfront:ListDistributions",
                        ],
                        resources=["*"],
                    ),
                ],
            )
        )

        source_output = codepipeline.Artifact("SourceOutput")

        self.pipeline = codepipeline.Pipeline(
            self,
            "Pipeline",
            pipeline_name=pipeline_name,
            pipeline_type=codepipeline.PipelineType.V2,
            cross_account_keys=False,
            stages=[
                codepipeline.StageProps(
                    stage_name="Source",
                    actions=[
                        codepipeline_actions.S3SourceAction(
                            action_name="S3Source",
                            bucket=source_bucket,
                            bucket_key=source_key,
                            output=source_output,
                            trigger=codepipeline_actions.S3Trigger.EVENTS,
                        ),
                    ],
                ),
                codepipeline.StageProps(
                    stage_name="QualityGate",
                    actions=[
                        codepipeline_actions.CodeBuildAction(
                            action_name="Lint",
                            project=self.quality_gate.lint_project,
                            input=source_output,
                        ),
                        codepipeline_actions.CodeBuildAction(
                            action_name="Tests",
                            project=self.quality_gate.test_project,
                            input=source_output,
                        ),
                        codepipeline_actions.CodeBuildAction(
                            action_name="SecurityScan",
                            project=self.quality_gate.security_project,
                            input=source_output,
                        ),
                    ]
                    + (
                        [
                            codepipeline_actions.CodeBuildAction(
                                action_name="Synth",
                                project=self.quality_gate.synth_project,
                                input=source_output,
                            )
                        ]
                        if self.quality_gate.synth_project
                        else []
                    ),
                ),
                codepipeline.StageProps(
                    stage_name="Deploy",
                    actions=[
                        codepipeline_actions.CodeBuildAction(
                            action_name="DeployWebApp",
                            project=deploy_project,
                            input=source_output,
                        ),
                    ],
                ),
            ],
        )

        NagSuppressions.add_resource_suppressions(
            self,
            [
                {"id": "AwsSolutions-IAM5", "reason": "CodePipeline + CDK deploy permissions."},
                {"id": "AwsSolutions-S1", "reason": "Pipeline artifacts bucket is ephemeral."},
            ],
            apply_to_children=True,
        )
