from __future__ import annotations

from runtime.lib.cache_version import compute_cache_version

NEURON_GRAPH_ENV: dict[str, str] = {
    "PJRT_DEVICE": "NEURON",
    "NEURON_FUSE_SOFTMAX": "1",
    "NEURON_RT_EXEC_TIMEOUT": "120",
    "NEURON_CC_FLAGS": "--model-type transformer --enable-fast-loading-neuron-binaries --auto-cast matmult --retry_failed_compilation",
    "ZIMAGE_ENCODER_MODE": "subprocess",
    "USE_TRACED_VAE": "1",
    "VAE_ON_CPU": "0",
    "TOKENIZERS_PARALLELISM": "false",
    "OMP_NUM_THREADS": "16",
    "MKL_NUM_THREADS": "16",
}


def neuron_env_for_resolution(resolution: int) -> dict[str, str]:
    env = {
        **NEURON_GRAPH_ENV,
        "NEURON_CACHE_VERSION": compute_cache_version(),
        "ZIMAGE_DEFAULT_RESOLUTION": str(resolution),
    }
    if resolution == 512:
        env["FUSION_DT_COUNTER"] = "1"
    if resolution == 1024:
        env["VAE_ON_CPU"] = "1"
    return env
