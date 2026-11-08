from __future__ import annotations

from aws_cdk import CfnOutput, Stack
from aws_cdk import aws_sns as sns
from aws_cdk import aws_sns_subscriptions as sns_subs
from constructs import Construct

from infra.config.settings import DeploySettings
from infra.stacks.constructs.cicd.pipeline import CombinedCicdPipeline
from infra.stacks.constructs.cicd.web_pipeline import WebCicdPipeline
from infra.stacks.constructs.storage.ecr import SecureEcrRepo
from infra.stacks.constructs.storage.s3 import SecureBucket
from infra.stacks.platform import PlatformStack
from infra.utils.constants import CUDA_BASE_IMAGE_URI, NEURON_BASE_IMAGE_URI
from infra.utils.enums import BuildTarget


class CicdStack(Stack):
    def __init__(
        self,
        scope: Construct,
        construct_id: str,
        *,
        settings: DeploySettings,
        platform: PlatformStack,
        **kwargs,
    ) -> None:
        super().__init__(scope, construct_id, **kwargs)

        self.findings_topic = sns.Topic(
            self,
            "FindingsTopic",
            display_name="Z-Image findings (image scans + CW alarms)",
            master_key=platform.key_logs,
            enforce_ssl=True,
        )
        self.findings_topic.add_subscription(
            sns_subs.EmailSubscription(settings.security.findings_email)
        )

        self.source_bucket = SecureBucket(
            self,
            "BuildSource",
            encryption_key=platform.key_artifacts,
            bucket_name=f"{settings.prefix}-build-source-{self.account}-{self.region}",
            log_bucket=platform.access_logs_bucket,
            versioned=True,
            event_bridge_enabled=True,
        ).bucket

        self.inference_repo = SecureEcrRepo(
            self,
            "InferenceRepo",
            repo_name=f"{settings.prefix}-{BuildTarget.SERVING_512}",
            findings_topic=self.findings_topic,
        )
        self.inference_repo_768 = SecureEcrRepo(
            self,
            "InferenceRepo768",
            repo_name=f"{settings.prefix}-{BuildTarget.SERVING_768}",
            findings_topic=self.findings_topic,
        )
        self.inference_repo_1024 = SecureEcrRepo(
            self,
            "InferenceRepo1024",
            repo_name=f"{settings.prefix}-{BuildTarget.SERVING_1024}",
            findings_topic=self.findings_topic,
        )
        self.compile_cpu_repo = SecureEcrRepo(
            self,
            "CompileCpuRepo",
            repo_name=f"{settings.prefix}-{BuildTarget.COMPILER}",
            findings_topic=self.findings_topic,
        )
        self.compile_verify_repo = SecureEcrRepo(
            self,
            "CompileVerifyRepo",
            repo_name=f"{settings.prefix}-{BuildTarget.VERIFIER}",
            findings_topic=self.findings_topic,
        )
        self.inference_gpu_repo = SecureEcrRepo(
            self,
            "InferenceGpuRepo",
            repo_name=f"{settings.prefix}-{BuildTarget.SERVING_GPU}",
            findings_topic=self.findings_topic,
        )

        self.combined_pipeline = CombinedCicdPipeline(
            self,
            "CombinedPipeline",
            pipeline_name=f"{settings.prefix}-inference-pipeline",
            targets=[
                {
                    "name": BuildTarget.SERVING_512.value,
                    "base_image": NEURON_BASE_IMAGE_URI.format(region=settings.region),
                    "resolution": 512,
                },
                {
                    "name": BuildTarget.SERVING_768.value,
                    "base_image": NEURON_BASE_IMAGE_URI.format(region=settings.region),
                    "resolution": 768,
                },
                {
                    "name": BuildTarget.SERVING_1024.value,
                    "base_image": NEURON_BASE_IMAGE_URI.format(region=settings.region),
                    "resolution": 1024,
                },
                {
                    "name": BuildTarget.SERVING_GPU.value,
                    "base_image": CUDA_BASE_IMAGE_URI.format(region=settings.region),
                },
                {
                    "name": BuildTarget.COMPILER.value,
                    "base_image": NEURON_BASE_IMAGE_URI.format(region=settings.region),
                },
                {
                    "name": BuildTarget.VERIFIER.value,
                    "base_image": NEURON_BASE_IMAGE_URI.format(region=settings.region),
                },
            ],
            source_bucket=self.source_bucket,
            image_repos={
                BuildTarget.SERVING_512.value: self.inference_repo.repository,
                BuildTarget.SERVING_768.value: self.inference_repo_768.repository,
                BuildTarget.SERVING_1024.value: self.inference_repo_1024.repository,
                BuildTarget.SERVING_GPU.value: self.inference_gpu_repo.repository,
                BuildTarget.COMPILER.value: self.compile_cpu_repo.repository,
                BuildTarget.VERIFIER.value: self.compile_verify_repo.repository,
            },
            kms_key=platform.key_logs,
            account=settings.account,
            region=settings.region,
            prefix=settings.prefix,
            environment=str(settings.environment),
        )

        self.web_pipeline = WebCicdPipeline(
            self,
            "WebPipeline",
            pipeline_name=f"{settings.prefix}-web-pipeline",
            source_bucket=self.source_bucket,
            source_key="build-contexts/web/context.zip",
            kms_key=platform.key_logs,
            prefix=settings.prefix,
            region=settings.region,
            environment=str(settings.environment),
        )

        CfnOutput(self, "SourceBucketName", value=self.source_bucket.bucket_name)
        CfnOutput(self, "PipelineName", value=self.combined_pipeline.pipeline.pipeline_name)
        CfnOutput(self, "WebPipelineName", value=self.web_pipeline.pipeline.pipeline_name)
        CfnOutput(self, "InferenceRepoUri", value=self.inference_repo.repository.repository_uri)
        CfnOutput(
            self, "InferenceRepoUri768", value=self.inference_repo_768.repository.repository_uri
        )
        CfnOutput(
            self, "InferenceRepoUri1024", value=self.inference_repo_1024.repository.repository_uri
        )
        CfnOutput(
            self, "InferenceGpuRepoUri", value=self.inference_gpu_repo.repository.repository_uri
        )
        CfnOutput(self, "CompileCpuRepoUri", value=self.compile_cpu_repo.repository.repository_uri)
        CfnOutput(
            self, "CompileVerifyRepoUri", value=self.compile_verify_repo.repository.repository_uri
        )
        CfnOutput(self, "FindingsTopicArn", value=self.findings_topic.topic_arn)

    @property
    def inference_image_uri_512(self) -> str:
        return f"{self.inference_repo.repository.repository_uri}:latest"

    @property
    def inference_image_uri_768(self) -> str:
        return f"{self.inference_repo_768.repository.repository_uri}:latest"

    @property
    def inference_image_uri_1024(self) -> str:
        return f"{self.inference_repo_1024.repository.repository_uri}:latest"

    @property
    def inference_gpu_image_uri(self) -> str:
        return f"{self.inference_gpu_repo.repository.repository_uri}:latest"

    @property
    def compile_cpu_image_uri(self) -> str:
        return f"{self.compile_cpu_repo.repository.repository_uri}:latest"

    @property
    def compile_verify_image_uri(self) -> str:
        return f"{self.compile_verify_repo.repository.repository_uri}:latest"
