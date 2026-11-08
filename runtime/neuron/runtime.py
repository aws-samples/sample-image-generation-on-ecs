from __future__ import annotations

import gc
import json
import math
import os
import threading
import time
from pathlib import Path

import torch
from loguru import logger

_TRACED_VAE = None

_PROMPT_EMBED_CACHE = {}


def _mark_step_every_n_layers() -> int:
    try:
        return max(0, int(os.environ.get("ZIMAGE_MARK_STEP_EVERY_N_LAYERS", "0")))
    except (TypeError, ValueError):
        return 0


_PATCHES_APPLIED = False


def _precompute_freqs_cis_real(
    dim: list[int], end: list[int], theta: float = 256.0
) -> list[torch.Tensor]:
    with torch.device("cpu"):
        tables = []
        for d, e in zip(dim, end, strict=False):
            freqs = 1.0 / (theta ** (torch.arange(0, d, 2, dtype=torch.float64) / d))
            t = torch.arange(e, dtype=torch.float64)
            angles = torch.outer(t, freqs).float()
            tables.append(torch.stack([torch.cos(angles), torch.sin(angles)], dim=-1))
        return tables


def _rope_call_real(self, ids: torch.Tensor) -> torch.Tensor:
    assert ids.ndim == 2
    assert ids.shape[-1] == len(self.axes_dims)
    device = ids.device

    if self.freqs_cis is None:
        self.freqs_cis = _precompute_freqs_cis_real(self.axes_dims, self.axes_lens, self.theta)
        self.freqs_cis = [f.to(device) for f in self.freqs_cis]
    elif self.freqs_cis[0].device != device:
        self.freqs_cis = [f.to(device) for f in self.freqs_cis]

    result = [self.freqs_cis[i][ids[:, i]] for i in range(len(self.axes_dims))]
    return torch.cat(result, dim=1)


def _apply_rotary_emb_real(x_in: torch.Tensor, freqs_cis: torch.Tensor) -> torch.Tensor:
    x_f = x_in.float()
    x_r = x_f.reshape(*x_f.shape[:-1], -1, 2)
    x_re, x_im = x_r[..., 0], x_r[..., 1]

    f = freqs_cis.unsqueeze(2).float()
    cos_f, sin_f = f[..., 0], f[..., 1]

    out = torch.stack([x_re * cos_f - x_im * sin_f, x_re * sin_f + x_im * cos_f], dim=-1).flatten(
        -2
    )
    return out.type_as(x_in)


def _build_nki_backend():
    try:
        from neuronxcc.nki._private_kernels.attention import attention_isa_kernel
        from torch_neuronx.xla_impl.ops import nki_jit

        _call = nki_jit()(attention_isa_kernel)
        return _call
    except Exception as exc:
        logger.info(f"[neuron_utils] NKI kernel not available: {exc}")
        return None


_nki_flash_lock = threading.Lock()
_nki_flash_call = None


def _nki_attention(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attn_mask: torch.Tensor | None = None,
    dropout_p: float = 0.0,
    is_causal: bool = False,
    scale: float | None = None,
) -> torch.Tensor:
    global _nki_flash_call
    if _nki_flash_call is None:
        with _nki_flash_lock:
            if _nki_flash_call is None:
                _nki_flash_call = _build_nki_backend()

    head_dim = query.shape[-1]
    use_nki = (_nki_flash_call is not None) and (head_dim <= 128)

    q = query.transpose(1, 2)
    k = key.transpose(1, 2)
    v = value.transpose(1, 2)

    if not use_nki:
        if attn_mask is not None:
            if attn_mask.ndim == 2:
                attn_mask = attn_mask[:, None, None, :]
            if attn_mask.dtype == torch.bool:
                float_mask = torch.zeros_like(attn_mask, dtype=q.dtype)
                float_mask.masked_fill_(~attn_mask, float("-inf"))
                attn_mask = float_mask
        out = torch.nn.functional.scaled_dot_product_attention(
            q,
            k,
            v,
            attn_mask=attn_mask,
            dropout_p=dropout_p,
            is_causal=is_causal,
            scale=scale,
        )
        return out.transpose(1, 2).contiguous()

    bs, n_head, q_len, d_head = q.shape
    k_len = k.shape[2]

    q_bf = q.contiguous().to(torch.bfloat16)
    k_bf = k.contiguous().to(torch.bfloat16)
    v_bf = v.contiguous().to(torch.bfloat16)

    q_nki = q_bf.permute(0, 1, 3, 2).reshape(bs * n_head, d_head, q_len)
    k_nki = k_bf.permute(0, 1, 3, 2).reshape(bs * n_head, d_head, k_len)
    v_nki = v_bf.reshape(bs * n_head, k_len, d_head)

    out_nki = torch.zeros((bs * n_head, q_len, d_head), dtype=torch.bfloat16, device=q.device)
    attn_scale = scale if scale is not None else (1.0 / math.sqrt(d_head))
    _nki_flash_call(
        q_nki,
        k_nki,
        v_nki,
        attn_scale,
        out_nki,
        kernel_name="AttentionMMSoftmaxMMWithoutSwap",
    )

    out = out_nki.reshape(bs, n_head, q_len, d_head)
    return out.transpose(1, 2).contiguous().type_as(query)


def _patch_transformer_forward() -> None:
    from torch.nn.utils.rnn import pad_sequence as _pad_seq
    from zimage.transformer import SEQ_MULTI_OF, ZImageTransformer2DModel

    _cpu_backend_trace_env = (
        os.environ.get("PJRT_DEVICE") == "CPU" or os.environ.get("ZIMAGE_ALLOW_CPU_COMPILE") == "1"
    )

    def _xla_safe_patchify(self, all_image, all_cap_feats, patch_size, f_patch_size):
        pH = pW = patch_size
        pF = f_patch_size
        device = all_image[0].device

        _cpu_backend_trace = _cpu_backend_trace_env
        mask_device = torch.device("cpu") if _cpu_backend_trace else device

        all_image_out, all_image_size, all_image_pos_ids, all_image_pad_mask = [], [], [], []
        all_cap_pos_ids, all_cap_pad_mask, all_cap_feats_out = [], [], []

        for image, cap_feat in zip(all_image, all_cap_feats, strict=False):
            cap_ori_len = len(cap_feat)
            cap_padding_len = (-cap_ori_len) % SEQ_MULTI_OF

            cap_padded_pos_ids = self.create_coordinate_grid(
                size=(cap_ori_len + cap_padding_len, 1, 1),
                start=(1, 0, 0),
                device=device,
            ).flatten(0, 2)
            all_cap_pos_ids.append(cap_padded_pos_ids)

            if cap_padding_len > 0:
                all_cap_pad_mask.append(
                    torch.cat(
                        [
                            torch.zeros((cap_ori_len,), dtype=torch.bool, device=mask_device),
                            torch.ones((cap_padding_len,), dtype=torch.bool, device=mask_device),
                        ],
                        dim=0,
                    )
                )
                all_cap_feats_out.append(
                    torch.cat([cap_feat, cap_feat[-1:].repeat(cap_padding_len, 1)], dim=0)
                )
            else:
                all_cap_pad_mask.append(
                    torch.zeros((cap_ori_len,), dtype=torch.bool, device=mask_device)
                )
                all_cap_feats_out.append(cap_feat)

            C, F, H, W = image.size()
            all_image_size.append((F, H, W))
            F_tokens, H_tokens, W_tokens = F // pF, H // pH, W // pW

            image = image.view(C, F_tokens, pF, H_tokens, pH, W_tokens, pW)
            image = image.permute(1, 3, 5, 2, 4, 6, 0).reshape(
                F_tokens * H_tokens * W_tokens, pF * pH * pW * C
            )
            image_ori_len = len(image)
            image_padding_len = (-image_ori_len) % SEQ_MULTI_OF

            image_ori_pos_ids = self.create_coordinate_grid(
                size=(F_tokens, H_tokens, W_tokens),
                start=(cap_ori_len + cap_padding_len + 1, 0, 0),
                device=device,
            ).flatten(0, 2)

            if image_padding_len > 0:
                pad_pos = (
                    self.create_coordinate_grid(size=(1, 1, 1), start=(0, 0, 0), device=device)
                    .flatten(0, 2)
                    .repeat(image_padding_len, 1)
                )
                all_image_pos_ids.append(torch.cat([image_ori_pos_ids, pad_pos], dim=0))
                all_image_pad_mask.append(
                    torch.cat(
                        [
                            torch.zeros((image_ori_len,), dtype=torch.bool, device=mask_device),
                            torch.ones((image_padding_len,), dtype=torch.bool, device=mask_device),
                        ],
                        dim=0,
                    )
                )
                all_image_out.append(
                    torch.cat([image, image[-1:].repeat(image_padding_len, 1)], dim=0)
                )
            else:
                all_image_pos_ids.append(image_ori_pos_ids)
                all_image_pad_mask.append(
                    torch.zeros((image_ori_len,), dtype=torch.bool, device=mask_device)
                )
                all_image_out.append(image)

        return (
            all_image_out,
            all_cap_feats_out,
            all_image_size,
            all_image_pos_ids,
            all_cap_pos_ids,
            all_image_pad_mask,
            all_cap_pad_mask,
        )

    ZImageTransformer2DModel.patchify_and_embed = _xla_safe_patchify

    def _xla_safe_forward(self, x, t, cap_feats, patch_size=2, f_patch_size=1):
        assert patch_size in self.all_patch_size
        assert f_patch_size in self.all_f_patch_size

        bsz = len(x)
        device = x[0].device
        t = t * self.t_scale
        t = self.t_embedder(t)

        (x, cap_feats, x_size, x_pos_ids, cap_pos_ids, x_inner_pad_mask, cap_inner_pad_mask) = (
            self.patchify_and_embed(x, cap_feats, patch_size, f_patch_size)
        )

        x_item_seqlens = [len(_) for _ in x]
        x_max_item_seqlen = max(x_item_seqlens)

        x = torch.cat(x, dim=0)
        x = self.all_x_embedder[f"{patch_size}-{f_patch_size}"](x)
        adaln_input = t.type_as(x)

        x_pad_mask = torch.cat(x_inner_pad_mask)
        if x_pad_mask.device != device:
            x_pad_mask = x_pad_mask.to(device)
        _x_total = sum(x_item_seqlens) if _cpu_backend_trace_env else x.shape[0]
        x = torch.where(
            x_pad_mask.unsqueeze(-1),
            self.x_pad_token.expand(_x_total, -1),
            x,
        )

        x = list(x.split(x_item_seqlens, dim=0))
        x_freqs_cis = list(
            self.rope_embedder(torch.cat(x_pos_ids, dim=0)).split(
                [len(_) for _ in x_pos_ids], dim=0
            )
        )
        x = _pad_seq(x, batch_first=True, padding_value=0.0)
        x_freqs_cis = _pad_seq(x_freqs_cis, batch_first=True, padding_value=0.0)
        _x_max_for_slice = x_max_item_seqlen if _cpu_backend_trace_env else x.shape[1]
        x_freqs_cis = x_freqs_cis[:, :_x_max_for_slice]

        x_attn_mask = torch.zeros((bsz, x_max_item_seqlen), dtype=torch.bool, device=device)
        for i, seq_len in enumerate(x_item_seqlens):
            x_attn_mask[i, :seq_len] = 1

        for layer in self.noise_refiner:
            x = layer(x, x_attn_mask, x_freqs_cis, adaln_input)

        _mark_every = _mark_step_every_n_layers()
        if _mark_every > 0:
            _mark_step()

        cap_item_seqlens = [len(_) for _ in cap_feats]
        cap_max_item_seqlen = max(cap_item_seqlens)

        cap_feats = torch.cat(cap_feats, dim=0)
        cap_feats = self.cap_embedder(cap_feats)

        cap_pad_mask = torch.cat(cap_inner_pad_mask)
        if cap_pad_mask.device != device:
            cap_pad_mask = cap_pad_mask.to(device)
        _cap_total = sum(cap_item_seqlens) if _cpu_backend_trace_env else cap_feats.shape[0]
        cap_feats = torch.where(
            cap_pad_mask.unsqueeze(-1),
            self.cap_pad_token.expand(_cap_total, -1),
            cap_feats,
        )

        cap_feats = list(cap_feats.split(cap_item_seqlens, dim=0))
        cap_freqs_cis = list(
            self.rope_embedder(torch.cat(cap_pos_ids, dim=0)).split(
                [len(_) for _ in cap_pos_ids], dim=0
            )
        )
        cap_feats = _pad_seq(cap_feats, batch_first=True, padding_value=0.0)
        cap_freqs_cis = _pad_seq(cap_freqs_cis, batch_first=True, padding_value=0.0)
        _cap_max_for_slice = cap_max_item_seqlen if _cpu_backend_trace_env else cap_feats.shape[1]
        cap_freqs_cis = cap_freqs_cis[:, :_cap_max_for_slice]

        cap_attn_mask = torch.zeros((bsz, cap_max_item_seqlen), dtype=torch.bool, device=device)
        for i, seq_len in enumerate(cap_item_seqlens):
            cap_attn_mask[i, :seq_len] = 1

        for layer in self.context_refiner:
            cap_feats = layer(cap_feats, cap_attn_mask, cap_freqs_cis)

        if _mark_every > 0:
            _mark_step()

        unified, unified_freqs_cis = [], []
        if _cpu_backend_trace_env:
            _mark_step()

        for i in range(bsz):
            x_len = x_item_seqlens[i]
            cap_len = cap_item_seqlens[i]
            if _cpu_backend_trace_env:
                _dim = self.dim
                _rope_dim = getattr(self, "head_dim", getattr(self, "_head_dim_for_rope", None))
                if _rope_dim is None:
                    _rope_dim = _dim // getattr(self, "n_heads", 1)

                unified_item = torch.zeros(
                    (x_len + cap_len, _dim),
                    dtype=x.dtype,
                    device=x.device,
                )
                unified_item[:x_len] = x[i, :x_len]
                unified_item[x_len : x_len + cap_len] = cap_feats[i, :cap_len]
                unified.append(unified_item)

                uni_fc_item = torch.zeros(
                    (x_len + cap_len, _rope_dim),
                    dtype=x_freqs_cis.dtype,
                    device=x_freqs_cis.device,
                )
                uni_fc_item[:x_len] = x_freqs_cis[i, :x_len]
                uni_fc_item[x_len : x_len + cap_len] = cap_freqs_cis[i, :cap_len]
                unified_freqs_cis.append(uni_fc_item)
            else:
                unified.append(torch.cat([x[i][:x_len], cap_feats[i][:cap_len]]))
                unified_freqs_cis.append(
                    torch.cat([x_freqs_cis[i][:x_len], cap_freqs_cis[i][:cap_len]])
                )

        unified_item_seqlens = [
            a + b for a, b in zip(cap_item_seqlens, x_item_seqlens, strict=False)
        ]
        unified_max_item_seqlen = max(unified_item_seqlens)

        unified = _pad_seq(unified, batch_first=True, padding_value=0.0)
        unified_freqs_cis = _pad_seq(unified_freqs_cis, batch_first=True, padding_value=0.0)

        unified_attn_mask = torch.zeros(
            (bsz, unified_max_item_seqlen), dtype=torch.bool, device=device
        )
        for i, seq_len in enumerate(unified_item_seqlens):
            unified_attn_mask[i, :seq_len] = 1

        for layer_idx, layer in enumerate(self.layers):
            unified = layer(unified, unified_attn_mask, unified_freqs_cis, adaln_input)
            if (
                _mark_every > 0
                and (layer_idx + 1) % _mark_every == 0
                and layer_idx != len(self.layers) - 1
            ):
                _mark_step()

        unified = self.all_final_layer[f"{patch_size}-{f_patch_size}"](unified, adaln_input)
        unified = list(unified.unbind(dim=0))
        x_out = self.unpatchify(unified, x_size, patch_size, f_patch_size)
        return x_out, {}

    ZImageTransformer2DModel.forward = _xla_safe_forward


def apply_patches(attention_backend: str = "native") -> None:
    global _PATCHES_APPLIED
    if _PATCHES_APPLIED:
        return

    import zimage.transformer as _tm
    from zimage.transformer import RopeEmbedder

    _tm.apply_rotary_emb = _apply_rotary_emb_real
    RopeEmbedder.__call__ = _rope_call_real

    _patch_transformer_forward()

    import utils.attention as _attn_mod

    if attention_backend == "nki":
        _attn_mod._ATTENTION_BACKENDS["nki"] = _nki_attention
        _attn_mod._ATTENTION_CONSTRAINTS["nki"] = []

    import torch.cuda as _cuda

    _orig = _cuda.empty_cache
    _cuda.empty_cache = lambda: _orig() if _cuda.is_available() else None

    _PATCHES_APPLIED = True
    logger.info(f"[neuron_utils] Patches applied (attention={attention_backend})")


def load_all_on_neuron(
    model_dir: str | Path,
    xla_device,
    dtype: torch.dtype = torch.bfloat16,
    verbose: bool = True,
) -> dict:
    import sys

    from safetensors.torch import load_file
    from transformers import AutoModel, AutoTokenizer

    sys.path.insert(0, str(Path(__file__).parent))

    from zimage.autoencoder import AutoencoderKL as LocalAutoencoderKL
    from zimage.scheduler import FlowMatchEulerDiscreteScheduler
    from zimage.transformer import ZImageTransformer2DModel

    from config import (
        DEFAULT_SCHEDULER_NUM_TRAIN_TIMESTEPS,
        DEFAULT_SCHEDULER_SHIFT,
        DEFAULT_SCHEDULER_USE_DYNAMIC_SHIFTING,
        DEFAULT_TRANSFORMER_CAP_FEAT_DIM,
        DEFAULT_TRANSFORMER_DIM,
        DEFAULT_TRANSFORMER_F_PATCH_SIZE,
        DEFAULT_TRANSFORMER_IN_CHANNELS,
        DEFAULT_TRANSFORMER_N_HEADS,
        DEFAULT_TRANSFORMER_N_KV_HEADS,
        DEFAULT_TRANSFORMER_N_LAYERS,
        DEFAULT_TRANSFORMER_N_REFINER_LAYERS,
        DEFAULT_TRANSFORMER_NORM_EPS,
        DEFAULT_TRANSFORMER_PATCH_SIZE,
        DEFAULT_TRANSFORMER_QK_NORM,
        DEFAULT_TRANSFORMER_T_SCALE,
        DEFAULT_VAE_IN_CHANNELS,
        DEFAULT_VAE_LATENT_CHANNELS,
        DEFAULT_VAE_NORM_NUM_GROUPS,
        DEFAULT_VAE_OUT_CHANNELS,
        DEFAULT_VAE_SCALING_FACTOR,
        ROPE_AXES_DIMS,
        ROPE_AXES_LENS,
        ROPE_THETA,
    )

    model_dir = Path(model_dir)

    def _log(msg):
        if verbose:
            logger.info(msg)

    _log("[load] Transformer …")
    t0 = time.time()
    transformer_dir = model_dir / "transformer"
    with open(transformer_dir / "config.json") as f:
        cfg = json.load(f)

    with torch.device("meta"):
        transformer = ZImageTransformer2DModel(
            all_patch_size=tuple(cfg.get("all_patch_size", DEFAULT_TRANSFORMER_PATCH_SIZE)),
            all_f_patch_size=tuple(cfg.get("all_f_patch_size", DEFAULT_TRANSFORMER_F_PATCH_SIZE)),
            in_channels=cfg.get("in_channels", DEFAULT_TRANSFORMER_IN_CHANNELS),
            dim=cfg.get("dim", DEFAULT_TRANSFORMER_DIM),
            n_layers=cfg.get("n_layers", DEFAULT_TRANSFORMER_N_LAYERS),
            n_refiner_layers=cfg.get("n_refiner_layers", DEFAULT_TRANSFORMER_N_REFINER_LAYERS),
            n_heads=cfg.get("n_heads", DEFAULT_TRANSFORMER_N_HEADS),
            n_kv_heads=cfg.get("n_kv_heads", DEFAULT_TRANSFORMER_N_KV_HEADS),
            norm_eps=cfg.get("norm_eps", DEFAULT_TRANSFORMER_NORM_EPS),
            qk_norm=cfg.get("qk_norm", DEFAULT_TRANSFORMER_QK_NORM),
            cap_feat_dim=cfg.get("cap_feat_dim", DEFAULT_TRANSFORMER_CAP_FEAT_DIM),
            rope_theta=cfg.get("rope_theta", ROPE_THETA),
            t_scale=cfg.get("t_scale", DEFAULT_TRANSFORMER_T_SCALE),
            axes_dims=cfg.get("axes_dims", ROPE_AXES_DIMS),
            axes_lens=cfg.get("axes_lens", ROPE_AXES_LENS),
        ).to(dtype)

    with open(transformer_dir / "diffusion_pytorch_model.safetensors.index.json") as f:
        index = json.load(f)
    shard_files = sorted(set(index["weight_map"].values()))
    state_dict: dict = {}
    for i, sf in enumerate(shard_files, 1):
        _log(f"  shard {i}/{len(shard_files)}: {sf}")
        shard = load_file(str(transformer_dir / sf), device="cpu")
        shard = {k: v.to(dtype) if v.dtype != dtype else v for k, v in shard.items()}
        state_dict.update(shard)
        del shard
        gc.collect()

    transformer.load_state_dict(state_dict, strict=False, assign=True)
    del state_dict
    gc.collect()

    _log("  → Neuron XLA …")
    transformer = transformer.to(xla_device)
    _mark_step()
    gc.collect()
    _log(f"  Transformer ready [{time.time() - t0:.1f}s]")

    _log("[load] VAE …")
    vae_dir = model_dir / "vae"
    with open(vae_dir / "config.json") as f:
        vcfg = json.load(f)

    vae = LocalAutoencoderKL(
        in_channels=vcfg.get("in_channels", DEFAULT_VAE_IN_CHANNELS),
        out_channels=vcfg.get("out_channels", DEFAULT_VAE_OUT_CHANNELS),
        down_block_types=tuple(vcfg.get("down_block_types", ("DownEncoderBlock2D",))),
        up_block_types=tuple(vcfg.get("up_block_types", ("UpDecoderBlock2D",))),
        block_out_channels=tuple(vcfg.get("block_out_channels", (64,))),
        layers_per_block=vcfg.get("layers_per_block", 1),
        latent_channels=vcfg.get("latent_channels", DEFAULT_VAE_LATENT_CHANNELS),
        norm_num_groups=vcfg.get("norm_num_groups", DEFAULT_VAE_NORM_NUM_GROUPS),
        scaling_factor=vcfg.get("scaling_factor", DEFAULT_VAE_SCALING_FACTOR),
        shift_factor=vcfg.get("shift_factor", None),
        use_quant_conv=vcfg.get("use_quant_conv", True),
        use_post_quant_conv=vcfg.get("use_post_quant_conv", True),
        mid_block_add_attention=vcfg.get("mid_block_add_attention", True),
    )
    vae_sd = {}
    for sf in sorted((model_dir / "vae").glob("*.safetensors")):
        vae_sd.update(load_file(str(sf), device="cpu"))
    vae.load_state_dict(vae_sd, strict=False)
    del vae_sd
    gc.collect()
    if os.environ.get("VAE_ON_CPU", "0") == "1":
        vae = vae.to(device="cpu", dtype=torch.float32)
        _log("  VAE ready (CPU, float32 — avoids 5M instruction limit)")
    else:
        vae = vae.to(device=xla_device, dtype=dtype)
        _log("  VAE ready (Neuron)")
    vae.eval()
    _mark_step()
    gc.collect()

    _log("[load] Text encoder (CPU — prompt encoding only) …")
    te_dir = model_dir / "text_encoder"
    text_encoder = AutoModel.from_pretrained(
        str(te_dir),
        torch_dtype=torch.float32,
        trust_remote_code=True,  # nosec B615 - local path, no remote download
    )
    text_encoder.to("cpu")
    text_encoder.eval()
    gc.collect()
    _log("  Text encoder ready (CPU, float32 — 5x faster than bfloat16 on CPU)")

    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    tok_dir = model_dir / "tokenizer"
    tokenizer = AutoTokenizer.from_pretrained(
        str(tok_dir) if tok_dir.exists() else str(te_dir),
        trust_remote_code=True,  # nosec B615 - local path, no remote download
    )

    with open(model_dir / "scheduler" / "scheduler_config.json") as f:
        scfg = json.load(f)
    scheduler = FlowMatchEulerDiscreteScheduler(
        num_train_timesteps=scfg.get("num_train_timesteps", DEFAULT_SCHEDULER_NUM_TRAIN_TIMESTEPS),
        shift=scfg.get("shift", DEFAULT_SCHEDULER_SHIFT),
        use_dynamic_shifting=scfg.get(
            "use_dynamic_shifting", DEFAULT_SCHEDULER_USE_DYNAMIC_SHIFTING
        ),
    )

    return {
        "transformer": transformer,
        "vae": vae,
        "text_encoder": text_encoder,
        "tokenizer": tokenizer,
        "scheduler": scheduler,
    }


def generate_on_neuron(
    transformer,
    vae,
    text_encoder,
    tokenizer,
    scheduler,
    prompt: str | list[str],
    height: int = 1024,
    width: int = 1024,
    num_inference_steps: int = 8,
    guidance_scale: float = 0.0,
    negative_prompt: str | list[str] | None = None,
    num_images_per_prompt: int = 1,
    seed: int = 42,
    max_sequence_length: int = 512,
    output_type: str = "pil",
    verbose: bool = True,
    vae_device: str = "auto",
) -> list:
    from tqdm import tqdm
    from zimage.pipeline import calculate_shift, retrieve_timesteps

    xla_device = next(transformer.parameters()).device
    dtype = next(transformer.parameters()).dtype

    vae_scale_factor = (
        2 ** (len(vae.config.block_out_channels) - 1)
        if hasattr(vae, "config") and hasattr(vae.config, "block_out_channels")
        else 8
    )
    vae_scale = vae_scale_factor * 2

    if isinstance(prompt, str):
        prompt = [prompt]
    batch_size = len(prompt)
    do_cfg = guidance_scale > 1.0

    global _PROMPT_EMBED_CACHE

    def _encode(prompts):
        cache_key = (tuple(prompts), max_sequence_length)
        if cache_key in _PROMPT_EMBED_CACHE:
            cached_embeds = _PROMPT_EMBED_CACHE[cache_key]
            return [e.to(xla_device) if e.device != xla_device else e for e in cached_embeds]
        formatted = []
        for p in prompts:
            formatted.append(
                tokenizer.apply_chat_template(
                    [{"role": "user", "content": p}],
                    tokenize=False,
                    add_generation_prompt=True,
                    enable_thinking=True,
                )
            )
        inputs = tokenizer(
            formatted,
            padding="max_length",
            max_length=max_sequence_length,
            truncation=True,
            return_tensors="pt",
        )
        ids = inputs.input_ids
        masks = inputs.attention_mask.bool()
        with torch.no_grad():
            embeds = (
                text_encoder(
                    input_ids=ids,
                    attention_mask=masks,
                    output_hidden_states=True,
                )
                .hidden_states[-2]
                .bfloat16()
            )
        fixed_cap_len = int(os.environ.get("ZIMAGE_FIXED_CAP_SEQ_LEN", "0"))
        result = []
        for i in range(len(prompts)):
            emb = embeds[i][masks[i]]
            if fixed_cap_len > 0:
                if emb.shape[0] > fixed_cap_len:
                    emb = emb[:fixed_cap_len]
                elif emb.shape[0] < fixed_cap_len:
                    pad_n = fixed_cap_len - emb.shape[0]
                    emb = torch.cat([emb, emb[-1:].repeat(pad_n, 1)], dim=0)
            else:
                pad = (-emb.shape[0]) % 32
                if pad:
                    emb = torch.cat([emb, emb[-1:].repeat(pad, 1)], dim=0)
            result.append(emb.to(xla_device))
        _PROMPT_EMBED_CACHE[cache_key] = result
        return result

    if verbose:
        logger.info("[gen] Encoding prompt …")
    prompt_embeds = _encode(prompt)
    neg_embeds = []
    if do_cfg:
        neg = (
            [""] * batch_size
            if negative_prompt is None
            else ([negative_prompt] if isinstance(negative_prompt, str) else negative_prompt)
        )
        neg_embeds = _encode(neg)

    if num_images_per_prompt > 1:
        prompt_embeds = [pe for pe in prompt_embeds for _ in range(num_images_per_prompt)]
        neg_embeds = [ne for ne in neg_embeds for _ in range(num_images_per_prompt)]

    actual_bs = batch_size * num_images_per_prompt

    height_latent = 2 * (height // vae_scale)
    width_latent = 2 * (width // vae_scale)
    image_seq_len = (height_latent // 2) * (width_latent // 2)

    from config import BASE_IMAGE_SEQ_LEN, BASE_SHIFT, MAX_IMAGE_SEQ_LEN, MAX_SHIFT

    mu = calculate_shift(
        image_seq_len,
        scheduler.config.get("base_image_seq_len", BASE_IMAGE_SEQ_LEN),
        scheduler.config.get("max_image_seq_len", MAX_IMAGE_SEQ_LEN),
        scheduler.config.get("base_shift", BASE_SHIFT),
        scheduler.config.get("max_shift", MAX_SHIFT),
    )
    scheduler.sigma_min = 0.0
    timesteps, num_inference_steps = retrieve_timesteps(
        scheduler,
        num_inference_steps,
        xla_device,
        sigmas=None,
        mu=mu,
    )

    shape = (actual_bs, transformer.in_channels, height_latent, width_latent)
    gen_cpu = torch.Generator("cpu").manual_seed(seed)
    latents = torch.randn(shape, generator=gen_cpu, device="cpu", dtype=torch.float32).to(
        xla_device
    )
    _mark_step()

    scheduler._step_index = 0

    if verbose:
        logger.info(f"[gen] Denoising {num_inference_steps} steps on Neuron …")
    for i, t in enumerate(tqdm(timesteps.cpu().tolist(), desc="Denoising", disable=not verbose)):
        t_tensor = torch.tensor([t], device=xla_device)
        if t == 0 and i == len(timesteps) - 1:
            continue

        timestep = t_tensor.expand(actual_bs)
        t_norm = (1000 - timestep) / 1000

        if do_cfg:
            lat_in = latents.to(dtype).repeat(2, 1, 1, 1).unsqueeze(2)
            cap_in = prompt_embeds + neg_embeds
            t_in = t_norm.repeat(2)
        else:
            lat_in = latents.to(dtype).unsqueeze(2)
            cap_in = prompt_embeds
            t_in = t_norm

        lat_list = list(lat_in.unbind(dim=0))
        with torch.no_grad():
            model_out, _ = transformer(lat_list, t_in, cap_in)
        if os.environ.get("SKIP_INTERMEDIATE_MARK_STEP") != "1":
            _mark_step()

        if do_cfg:
            pos, neg_ = model_out[:actual_bs], model_out[actual_bs:]
            noise_pred = torch.stack(
                [
                    p.float() + guidance_scale * (p.float() - n.float())
                    for p, n in zip(pos, neg_, strict=False)
                ],
                dim=0,
            )
        else:
            noise_pred = torch.stack([o.float() for o in model_out], dim=0)

        noise_pred = -noise_pred.squeeze(2)

        if not hasattr(scheduler, "_sigmas_py"):
            scheduler._sigmas_py = [float(x) for x in scheduler.sigmas.detach().cpu().tolist()]
        sigma = scheduler._sigmas_py[scheduler._step_index]
        next_idx = min(scheduler._step_index + 1, len(scheduler._sigmas_py) - 1)
        sigma_next = scheduler._sigmas_py[next_idx]
        dt = sigma_next - sigma
        if os.environ.get("FUSION_DT_XLA_INPUT") == "1":
            dt_xla = torch.tensor([float(dt)], device=latents.device, dtype=torch.float32)
            latents = latents.to(torch.float32) + dt_xla * noise_pred
        elif os.environ.get("FUSION_DT_COUNTER") == "1":
            if not hasattr(scheduler, "_all_sigmas_xla"):
                scheduler._all_sigmas_xla = scheduler.sigmas.to(latents.device).to(torch.float32)
                scheduler._step_counter_xla = torch.zeros(
                    (), device=latents.device, dtype=torch.long
                )
            sig_now = scheduler._all_sigmas_xla[scheduler._step_counter_xla]
            sig_next = scheduler._all_sigmas_xla[scheduler._step_counter_xla + 1]
            dt_xla = sig_next - sig_now
            latents = latents.to(torch.float32) + dt_xla * noise_pred
            scheduler._step_counter_xla = scheduler._step_counter_xla + 1
        else:
            latents = latents.to(torch.float32) + dt * noise_pred
        scheduler._step_index += 1
        _mark_step()

    if output_type == "latent":
        return latents

    shift = getattr(vae.config, "shift_factor", 0.0) or 0.0

    traced_vae_path = Path(os.environ.get("NEURON_VAE_PATH", "/app/compiled_models/vae_1024.pt"))
    use_traced_vae = traced_vae_path.exists() and os.environ.get("USE_TRACED_VAE", "1") == "1"

    vae_is_cpu = next(vae.parameters()).device.type == "cpu"
    if use_traced_vae:
        if verbose:
            logger.info(f"[gen] Using pre-traced VAE on Neuron ({traced_vae_path})…", flush=True)
        import sys as _sys

        _sys.stdout.flush()
        _mark_step()
        lat_cpu = latents.to(torch.float32).cpu()
        if verbose:
            logger.info(f"[gen] latents on CPU, shape={list(lat_cpu.shape)}", flush=True)
        _sys.stdout.flush()
        lat_in = lat_cpu / vae.config.scaling_factor + shift
        if verbose:
            logger.info("[gen] Running traced Neuron VAE decode…", flush=True)
        _sys.stdout.flush()
        t_vae = time.time()
        global _TRACED_VAE
        if "_TRACED_VAE" not in globals() or _TRACED_VAE is None:
            _TRACED_VAE = torch.jit.load(traced_vae_path)
            if verbose:
                logger.info("[gen] Loaded traced VAE NEFF", flush=True)
        with torch.no_grad():
            image = _TRACED_VAE(lat_in)
        if verbose:
            logger.info(f"[gen] Neuron VAE decode done in {time.time() - t_vae:.3f}s", flush=True)
        _sys.stdout.flush()
    elif vae_device == "cpu" or vae_is_cpu:
        if verbose:
            logger.info("[gen] Syncing XLA before VAE decode on CPU…", flush=True)
        import sys as _sys

        _sys.stdout.flush()
        _mark_step()
        if verbose:
            logger.info("[gen] mark_step done, transferring latents to CPU…", flush=True)
        _sys.stdout.flush()
        lat_cpu = latents.to(torch.float32).cpu()
        if verbose:
            logger.info(
                f"[gen] latents on CPU, shape={list(lat_cpu.shape)}, dtype={lat_cpu.dtype}",
                flush=True,
            )
        _sys.stdout.flush()
        vae_dtype = next(vae.parameters()).dtype
        lat_in = (lat_cpu.to(vae_dtype) / vae.config.scaling_factor) + shift
        if verbose:
            logger.info(f"[gen] Running VAE decode on CPU (vae_dtype={vae_dtype})…", flush=True)
        _sys.stdout.flush()
        t_vae = time.time()
        with torch.no_grad():
            image = vae.decode(lat_in, return_dict=False)[0]
        if verbose:
            logger.info(f"[gen] VAE decode done in {time.time() - t_vae:.2f}s", flush=True)
        _sys.stdout.flush()
    else:
        lat_in = (latents.to(dtype) / vae.config.scaling_factor) + shift
        with torch.no_grad():
            image = vae.decode(lat_in, return_dict=False)[0]
        _mark_step()

    if output_type == "pil":
        from PIL import Image

        image = (image / 2 + 0.5).clamp(0, 1)
        image = image.cpu().permute(0, 2, 3, 1).float().numpy()
        image = (image * 255).round().astype("uint8")
        return [Image.fromarray(img) for img in image]
    return image


def get_xla_device():
    import torch_neuronx  # noqa: F401 — registers Neuron XLA plugin
    import torch_xla.core.xla_model as xm

    return xm.xla_device()


def _mark_step():
    try:
        import torch_xla.core.xla_model as xm

        xm.mark_step()
    except Exception:
        pass


class FixedShapeTransformerWrapper(torch.nn.Module):
    def __init__(self, transformer: torch.nn.Module) -> None:
        super().__init__()
        self.transformer = transformer

    def forward(
        self,
        lat: torch.Tensor,
        timestep: torch.Tensor,
        cap_padded: torch.Tensor,
    ) -> torch.Tensor:
        out, _ = self.transformer([lat], timestep, [cap_padded])
        stacked = torch.stack(out)
        return stacked.squeeze(2)


def pad_caption(
    emb: torch.Tensor,
    max_seq_len: int = 512,
    device: torch.device | None = None,
) -> torch.Tensor:
    cap_dim = emb.shape[-1]
    out = torch.zeros(max_seq_len, cap_dim, dtype=emb.dtype)
    valid = min(emb.shape[0], max_seq_len)
    out[:valid] = emb[:valid]
    if device is not None:
        out = out.to(device)
    return out
