from __future__ import annotations

from aws_cdk import RemovalPolicy
from aws_cdk import aws_ecr as ecr
from aws_cdk import aws_events as events
from aws_cdk import aws_events_targets as targets
from aws_cdk import aws_sns as sns
from constructs import Construct


class SecureEcrRepo(Construct):
    def __init__(
        self,
        scope: Construct,
        construct_id: str,
        *,
        repo_name: str,
        findings_topic: sns.ITopic,
        keep_last_n: int = 10,
    ) -> None:
        super().__init__(scope, construct_id)

        self.repository = ecr.Repository(
            self,
            "Repo",
            repository_name=repo_name,
            image_scan_on_push=True,
            image_tag_mutability=ecr.TagMutability.IMMUTABLE,
            encryption=ecr.RepositoryEncryption.KMS,
            lifecycle_rules=[
                ecr.LifecycleRule(
                    description=f"Keep last {keep_last_n} images",
                    max_image_count=keep_last_n,
                    rule_priority=1,
                )
            ],
            empty_on_delete=True,
            removal_policy=RemovalPolicy.DESTROY,
        )

        events.Rule(
            self,
            "ScanFindingsRule",
            description=f"Forward HIGH/CRITICAL scan findings for {repo_name} to SNS",
            event_pattern=events.EventPattern(
                source=["aws.ecr"],
                detail_type=["ECR Image Scan"],
                detail={
                    "repository-name": [repo_name],
                    "finding-severity-counts": {
                        "$or": [
                            {"CRITICAL": [{"numeric": [">", 0]}]},
                            {"HIGH": [{"numeric": [">", 0]}]},
                        ]
                    },
                },
            ),
            targets=[targets.SnsTopic(findings_topic)],
        )

        events.Rule(
            self,
            "InspectorFindingsRule",
            description=f"Forward Inspector v2 HIGH/CRITICAL findings for {repo_name} to SNS",
            event_pattern=events.EventPattern(
                source=["aws.inspector2"],
                detail_type=["Inspector2 Finding"],
                detail={
                    "severity": ["HIGH", "CRITICAL"],
                    "resources": {"type": ["AWS_ECR_CONTAINER_IMAGE"]},
                },
            ),
            targets=[targets.SnsTopic(findings_topic)],
        )

    @property
    def repository_uri(self) -> str:
        return self.repository.repository_uri

    def grant_pull(self, grantee) -> None:
        self.repository.grant_pull(grantee)
