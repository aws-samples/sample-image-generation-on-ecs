from __future__ import annotations

from runtime.lib.paths import (
    NEFF_ENCODER_256,
    NEFF_TRANSFORMER_512,
    NEFF_VAE_512,
    NEFF_VAE_768,
    NEFF_VAE_1024,
)

TRACE_SPECS = [
    {
        "step": "TraceVAE512",
        "script": "trace_vae_512.py",
        "output": NEFF_VAE_512,
        "instance": "ml.c6i.8xlarge",
    },
    {
        "step": "TraceVAE768",
        "script": "trace_vae_768.py",
        "output": NEFF_VAE_768,
        "instance": "ml.c6i.8xlarge",
    },
    {
        "step": "TraceVAE1024",
        "script": "trace_vae_1024.py",
        "output": NEFF_VAE_1024,
        "instance": "ml.c6i.8xlarge",
    },
]

NEURON_TRACE_SPECS = [
    {
        "step": "TraceTransformer512",
        "script": "trace_transformer.py",
        "output": NEFF_TRANSFORMER_512,
    },
    {
        "step": "TraceEncoder",
        "script": "trace_encoder.py",
        "output": NEFF_ENCODER_256,
    },
]
