import hashlib
import json
import os
import signal
import sys
from dataclasses import dataclass
from pathlib import Path

from loguru import logger

from lib.paths import COMPILED_MODELS_DIR, WEIGHTS_DIR

ROOT = Path(os.environ.get("ZIMAGE_ROOT", "/opt/ml/code"))
SRC = ROOT / "zimage"
COMPILE_DIR = ROOT / "compile"
COMPILE_SCRIPTS = COMPILE_DIR
LOGS_DIR = ROOT / "logs"
OUTPUTS_DIR = ROOT / "outputs"
RESULTS_DIR = ROOT / "results"
GOLDEN_512 = OUTPUTS_DIR / "golden_512.png"
GOLDEN_768 = OUTPUTS_DIR / "golden_768.png"

for d in (LOGS_DIR, OUTPUTS_DIR, RESULTS_DIR, ROOT / COMPILED_MODELS_DIR):
    d.mkdir(parents=True, exist_ok=True)

PROMPT = (
    "Young Chinese woman in red Hanfu, intricate embroidery. "
    "Impeccable makeup, red floral forehead pattern. "
    "Elaborate high bun, golden phoenix headdress, red flowers, beads. "
    "Holds round folding fan with lady, trees, bird. "
    "Neon lightning-bolt lamp, bright yellow glow, above extended left palm. "
    "Soft-lit outdoor night background, silhouetted tiered pagoda, "
    "blurred colorful distant lights."
)

STEPS = 8
SEED = 42
GUIDANCE_SCALE = 0.0
DTYPE_STR = "bfloat16"
MODEL_PATH = os.environ.get("ZIMAGE_MODEL_PATH", WEIGHTS_DIR)

EXP40_FRESH_1024_S = 20.425
EXP40_CACHED_1024_S = 16.625


def assert_neuron_available():
    if os.environ.get("ZIMAGE_ALLOW_CPU_COMPILE") == "1":
        return
    if not Path("/dev/neuron0").exists():
        logger.error("/dev/neuron0 missing — Neuron runtime not accessible.")
        sys.exit(2)


@dataclass
class CompileConfig:
    resolution: int
    max_sequence_length: int = 512
    steps: int = STEPS
    seed: int = SEED
    guidance_scale: float = GUIDANCE_SCALE
    vae_mode: str = "cpu_fp32"

    extra_cc_flags: str = ""

    attention_backend: str = "native"

    @property
    def height(self) -> int:
        return self.resolution

    @property
    def width(self) -> int:
        return self.resolution

    @property
    def cache_tag(self) -> str:
        return f"{self.resolution}_{self.max_sequence_length}"

    @property
    def cache_path(self) -> Path:
        return ROOT / COMPILED_MODELS_DIR / f"cache_{self.cache_tag}"

    @property
    def image_tokens(self) -> int:
        return (self.resolution // 8 // 2) ** 2

    @property
    def total_tokens(self) -> int:
        return self.image_tokens + self.max_sequence_length


def setup(cfg: CompileConfig, timeout_min: int = 60, hlo_dump_dir: str | None = None) -> None:
    sys.path.insert(0, str(SRC))
    sys.path.insert(0, str(ROOT))
    sys.path.insert(0, str(COMPILE_SCRIPTS))
    os.chdir(ROOT)

    venv_bin = os.environ.get("VIRTUAL_ENV", "/usr/local") + "/bin"
    if venv_bin not in os.environ.get("PATH", ""):
        os.environ["PATH"] = venv_bin + ":" + os.environ.get("PATH", "")

    def _handler(sig, frame):
        raise TimeoutError(f"Compile exceeded {timeout_min}-min watchdog")

    signal.signal(signal.SIGALRM, _handler)
    signal.alarm(timeout_min * 60)

    cfg.cache_path.mkdir(parents=True, exist_ok=True)

    os.environ.setdefault("PJRT_DEVICE", "NEURON")
    os.environ.setdefault("NEURON_RT_VISIBLE_CORES", "0,1")
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    os.environ.setdefault("NEURON_FUSE_SOFTMAX", "1")
    os.environ.setdefault("NEURON_RT_EXEC_TIMEOUT", "120")
    os.environ["NEURON_COMPILE_CACHE_URL"] = str(cfg.cache_path)

    if cfg.vae_mode == "cpu_fp32":
        os.environ["USE_TRACED_VAE"] = "0"
        os.environ["VAE_ON_CPU"] = "1"
    elif cfg.vae_mode == "neuron_lazy":
        os.environ["USE_TRACED_VAE"] = "0"
        os.environ["VAE_ON_CPU"] = "0"
    elif cfg.vae_mode == "neuron_traced_512":
        os.environ["USE_TRACED_VAE"] = "1"
        os.environ["VAE_ON_CPU"] = "0"
    else:
        raise ValueError(f"unknown vae_mode: {cfg.vae_mode}")

    base_flags = (
        "--model-type transformer "
        "--enable-fast-loading-neuron-binaries "
        "--auto-cast matmult "
        "--retry_failed_compilation"
    )
    os.environ["NEURON_CC_FLAGS"] = base_flags + (
        " " + cfg.extra_cc_flags.strip() if cfg.extra_cc_flags.strip() else ""
    )

    if hlo_dump_dir:
        Path(hlo_dump_dir).mkdir(parents=True, exist_ok=True)
        os.environ["XLA_FLAGS"] = (
            f"--xla_dump_to={hlo_dump_dir} --xla_dump_hlo_as_text --xla_dump_hlo_as_proto"
        )
        os.environ["XLA_IR_DEBUG"] = "1"
        os.environ["NEURON_FRAMEWORK_DEBUG"] = "1"
    logger.info(f"setup] HLO dump: {hlo_dump_dir}")

    for lock in cfg.cache_path.rglob("*.lock"):
        try:
            lock.unlink()
        except OSError:
            pass

    logger.info(f"setup] resolution    : {cfg.resolution}×{cfg.resolution}")
    logger.info(f"setup] max_seq_len   : {cfg.max_sequence_length}")
    logger.info(
        f"setup] total_tokens  : {cfg.total_tokens} "
        f"(image={cfg.image_tokens} + caption={cfg.max_sequence_length})"
    )
    logger.info(f"setup] vae_mode      : {cfg.vae_mode}")
    logger.info(f"setup] cache         : {cfg.cache_path}")
    logger.info(f"setup] compile flags : {os.environ['NEURON_CC_FLAGS']}")
    logger.info(f"setup] watchdog      : {timeout_min} min")


def extract_instruction_count(log_path: Path) -> int | None:
    if not log_path.exists():
        return None
    import re

    patterns = [
        re.compile(r"Instructions generated by compiler\s+([\d,]+)"),
        re.compile(r"HLO verifier:\s+([\d,]+)\s+instructions"),
        re.compile(r"(\d[\d,]*)\s+instructions\s+generated"),
    ]
    best = 0
    text = log_path.read_text(errors="ignore")
    for pat in patterns:
        for m in pat.finditer(text):
            n = int(m.group(1).replace(",", ""))
            if n > best:
                best = n
    return best or None


def save_results(results: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(results, indent=2, default=str))
    logger.info(f" [results] → {path}")


def pixel_sha256(path: Path) -> str:
    import numpy as np
    from PIL import Image

    arr = np.array(Image.open(str(path)).convert("RGB"))
    return hashlib.sha256(arr.astype("uint8").tobytes()).hexdigest()


def image_stats(path: Path) -> dict:
    import numpy as np
    from PIL import Image

    arr = np.array(Image.open(str(path)).convert("RGB")).astype("float32")
    return {
        "mean": round(float(arr.mean()), 3),
        "std": round(float(arr.std()), 3),
        "min": float(arr.min()),
        "max": float(arr.max()),
        "sha256_prefix": pixel_sha256(path)[:16],
    }


def compare_to_golden(output_path: Path, golden_path: Path) -> dict:
    import numpy as np
    from PIL import Image

    if not golden_path.exists():
        return {"verdict": "NO_GOLDEN", "message": f"{golden_path} missing"}
    g = np.array(Image.open(str(golden_path)).convert("RGB")).astype("float32")
    o = np.array(Image.open(str(output_path)).convert("RGB")).astype("float32")
    if g.shape != o.shape:
        return {
            "verdict": "SHAPE_MISMATCH",
            "golden_shape": list(g.shape),
            "output_shape": list(o.shape),
        }
    diff = np.abs(g - o)
    mae = float(diff.mean())
    mse = float(np.mean(diff**2))
    psnr = float(10 * np.log10(255**2 / (mse + 1e-10)))
    bit_exact = diff.max() == 0
    if bit_exact:
        verdict = "BIT_EXACT"
    elif mae < 1.0:
        verdict = "NEAR_IDENTICAL"
    elif mae < 5.0:
        verdict = "VISUALLY_IDENTICAL"
    elif mae < 20.0:
        verdict = "SIMILAR"
    else:
        verdict = "DIFFERENT"
    return {
        "verdict": verdict,
        "bit_exact": bit_exact,
        "mae": round(mae, 4),
        "mse": round(mse, 4),
        "psnr_db": round(psnr, 3),
    }
