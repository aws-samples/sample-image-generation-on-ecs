from __future__ import annotations

from runtime.lib.paths import (
    CACHE_768_PREFIX,
    CACHE_1024_PREFIX,
)

VPC_CIDR = "10.20.0.0/16"

HF_REPO_ID = "Tongyi-MAI/Z-Image-Turbo"
HF_MODEL_REVISION = "main"

ZIMAGE_UPSTREAM_REPO = "https://github.com/Tongyi-MAI/Z-Image.git"
ZIMAGE_UPSTREAM_COMMIT = "26f23eda" + "626ffadd" + "a020b04f" + "f79488e1" + "d72004cd"

AWS_DLC_ACCOUNT_ID = "763104351884"
NEURON_BASE_IMAGE_URI = (
    f"{AWS_DLC_ACCOUNT_ID}.dkr.ecr.{{region}}.amazonaws.com/pytorch-inference-neuronx"
    "@sha256:dab7cc0b74bec3e9a28e60ef2029b9b945525b2a3fd091de4e45633ce1df262e"
)
CUDA_BASE_IMAGE_URI = (
    f"{AWS_DLC_ACCOUNT_ID}.dkr.ecr.{{region}}.amazonaws.com/pytorch-inference"
    "@sha256:aaf6233a339e556f7ddbe4f103054986d21e7317d13da636934112828dbc1576"
)

CACHE_PREFIX_768 = CACHE_768_PREFIX
CACHE_PREFIX_1024 = CACHE_1024_PREFIX

CDK_BOOTSTRAP_QUALIFIER = "hnb659fds"
