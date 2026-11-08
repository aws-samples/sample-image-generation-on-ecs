from __future__ import annotations

WEIGHTS_DIR = "ckpts/Z-Image-Turbo"
COMPILED_MODELS_DIR = "compiled_models"
COMPILE_PREFIX = "compile"
CACHE_768_PREFIX = "cache_768"
CACHE_1024_PREFIX = "cache_1024"

NEURON_NEFFS = {
    512: ["transformer_512.pt", "encoder_256.pt", "vae_512.pt"],
    768: ["encoder_256.pt", "vae_768.pt"],
    1024: ["encoder_256.pt", "vae_1024.pt"],
}
ALL_TRACED_NEFFS = [
    "transformer_512.pt",
    "encoder_256.pt",
    "vae_512.pt",
    "vae_768.pt",
    "vae_1024.pt",
]

NEFF_TRANSFORMER_512 = "transformer_512.pt"
NEFF_ENCODER_256 = "encoder_256.pt"
NEFF_VAE_512 = "vae_512.pt"
NEFF_VAE_768 = "vae_768.pt"
NEFF_VAE_1024 = "vae_1024.pt"

DEFAULT_ARTIFACTS_DIR = "/opt/ml/zimage_artifacts"
EFS_MOUNT_PATH = "/mnt/efs"
EFS_ARTIFACTS_PATH = "/mnt/efs/artifacts"
