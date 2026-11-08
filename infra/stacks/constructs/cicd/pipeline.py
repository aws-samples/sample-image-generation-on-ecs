from __future__ import annotations

from aws_cdk import aws_codepipeline as codepipeline
from aws_cdk import aws_codepipeline_actions as codepipeline_actions
from aws_cdk import aws_ecr as ecr
from aws_cdk import aws_kms as kms
from aws_cdk import aws_s3 as s3
from cdk_nag import NagSuppressions
from constructs import Construct

from infra.stacks.constructs.cicd.build_projects import ImageBuildProject
from infra.stacks.constructs.cicd.compile_models import CompileModelsProject
from infra.stacks.constructs.cicd.deploy_endpoints import DeployEndpointsProject
from infra.stacks.constructs.cicd.deploy_stage import DeployProject
from infra.stacks.constructs.cicd.quality_gate import QualityGate, QualityGateTargets
from infra.utils.enums import BuildTarget


class CombinedCicdPipeline(Construct):
    def __init__(
        self,
        scope: Construct,
        construct_id: str,
        *,
        pipeline_name: str,
        targets: list[dict],
        source_bucket: s3.IBucket,
        image_repos: dict[str, ecr.IRepository],
        kms_key: kms.IKey,
        account: str,
        region: str,
        prefix: str,
        environment: str = "dev",
    ) -> None:
        super().__init__(scope, construct_id)

        self.quality_gate = QualityGate(
            self,
            "QualityGate",
            kms_key=kms_key,
            name_prefix=f"{prefix}-quality-gate",
            targets=QualityGateTargets(
                lint="lint-inference",
                test="test-inference",
                security="security",
                synth="synth-inference",
            ),
        )

        self.build_projects: dict[str, ImageBuildProject] = {}
        for target_cfg in targets:
            name = target_cfg["name"]
            repo = image_repos[name]
            self.build_projects[name] = ImageBuildProject(
                self,
                f"{name}Build",
                target_name=name,
                repo=repo,
                kms_key=kms_key,
                base_image=target_cfg.get("base_image"),
                resolution=target_cfg.get("resolution"),
                is_inference=BuildTarget(name).is_neuron_serving
                if name in BuildTarget._value2member_map_
                else False,
                prefix=prefix,
            )

        self.deploy_project = DeployProject(
            self, "Deploy", kms_key=kms_key, account=account, prefix=prefix
        )

        self.compile_models = CompileModelsProject(
            self, "CompileModels", kms_key=kms_key, prefix=prefix
        )

        self.endpoint_stack_short_names = ["inference-ecs"]
        self.deploy_endpoint_projects: dict[str, DeployEndpointsProject] = {
            short: DeployEndpointsProject(
                self,
                f"Deploy-{short}",
                kms_key=kms_key,
                account=account,
                endpoint_stack_short=short,
                prefix=prefix,
                environment=environment,
            )
            for short in self.endpoint_stack_short_names
        }

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
                            bucket_key="build-contexts/cdk/context.zip",
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
                    stage_name="Build",
                    actions=[
                        codepipeline_actions.CodeBuildAction(
                            action_name=f"{name}Build",
                            project=self.build_projects[name].project,
                            input=source_output,
                        )
                        for name in self.build_projects
                    ],
                ),
                codepipeline.StageProps(
                    stage_name="Deploy",
                    actions=[
                        codepipeline_actions.CodeBuildAction(
                            action_name="CdkDeploy",
                            project=self.deploy_project.project,
                            input=source_output,
                        ),
                    ],
                ),
                codepipeline.StageProps(
                    stage_name="CompileModels",
                    actions=[
                        codepipeline_actions.CodeBuildAction(
                            action_name="CompileModels",
                            project=self.compile_models.project,
                            input=source_output,
                        ),
                    ],
                ),
            ]
            + (
                [
                    codepipeline.StageProps(
                        stage_name="DeployEndpoints",
                        actions=[
                            codepipeline_actions.CodeBuildAction(
                                action_name=f"Deploy-{short}",
                                project=self.deploy_endpoint_projects[short].project,
                                input=source_output,
                            )
                            for short in self.endpoint_stack_short_names
                        ],
                    )
                ]
                if self.endpoint_stack_short_names
                else []
            ),
        )

        NagSuppressions.add_resource_suppressions(
            self,
            [
                {
                    "id": "AwsSolutions-IAM5",
                    "reason": "CodePipeline auto-generated role requires S3 wildcards.",
                },
                {
                    "id": "AwsSolutions-S1",
                    "reason": "Pipeline artifacts bucket is ephemeral.",
                },
            ],
            apply_to_children=True,
        )
