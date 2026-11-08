from __future__ import annotations

from enum import StrEnum


class Environment(str):
    def __new__(cls, value: str):
        if not value.isalnum() or not value.islower():
            raise ValueError(f"Environment must be lowercase alphanumeric, got: {value!r}")
        return super().__new__(cls, value)

    @property
    def value(self) -> str:
        return str(self)


class PipelineStage(StrEnum):
    SOURCE = "Source"
    QUALITY_GATE = "QualityGate"
    BUILD = "Build"
    DEPLOY = "Deploy"
    COMPILE_MODELS = "CompileModels"
    DEPLOY_ENDPOINTS = "DeployEndpoints"


class BuildTarget(StrEnum):
    SERVING_512 = "serving-512"
    SERVING_768 = "serving-768"
    SERVING_1024 = "serving-1024"
    SERVING_GPU = "serving-gpu"
    COMPILER = "compiler"
    VERIFIER = "verifier"
    WEB = "web"

    @property
    def dockerfile(self) -> str:
        if self.is_neuron_serving:
            return "runtime/Dockerfile.inference"
        if self == BuildTarget.SERVING_GPU:
            return "runtime/Dockerfile.inference_gpu"
        if self == BuildTarget.COMPILER:
            return "runtime/Dockerfile.compile_cpu"
        if self == BuildTarget.VERIFIER:
            return "runtime/Dockerfile.compile_verify"
        if self == BuildTarget.WEB:
            return "web/Dockerfile"
        return "runtime/Dockerfile.inference"

    @property
    def context_dir(self) -> str:
        if self == BuildTarget.WEB:
            return "web/"
        return "runtime/"

    @property
    def is_serving(self) -> bool:
        return self.value.startswith("serving")

    @property
    def is_neuron_serving(self) -> bool:
        return self.is_serving and "gpu" not in self.value
