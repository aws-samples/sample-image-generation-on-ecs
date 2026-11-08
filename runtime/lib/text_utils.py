from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import torch


def pad_caption(
    caption_embs: torch.Tensor,
    emb_masks: torch.Tensor,
    max_length: int = 512,
) -> torch.Tensor:
    import torch

    valid_len = int(emb_masks[0].sum().item())
    padded = torch.zeros(1, max_length, caption_embs.shape[-1], dtype=caption_embs.dtype)
    padded[:, : min(valid_len, max_length), :] = caption_embs[:, : min(valid_len, max_length), :]
    return padded
