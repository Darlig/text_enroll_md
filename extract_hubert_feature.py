#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import argparse
from pathlib import Path
from typing import Optional, List
import re

import torch
import torch.nn as nn

# 建议用 soundfile 读 wav：pip install soundfile
import soundfile as sf

# from transformers import HubertModel
from model.TransformerKWSPhone_hubert_wenet import FrozenHubert


# class FrozenHubert(nn.Module):
#     """
#     输入: wav (B, L) float32/float16, 16kHz
#     输出: feats (B, T, D)  (T 为 HuBERT 帧数)
#     """
#     def __init__(self, model_name: str, device: Optional[str] = None):
#         super().__init__()
#         self.hubert = HubertModel.from_pretrained(model_name, use_safetensors=False)
#         self.hubert.eval()
#         for p in self.hubert.parameters():
#             p.requires_grad = False

#         self.device = device
#         if device is not None:
#             self.hubert.to(device)

#     @torch.no_grad()
#     def forward(self, wav_16k: torch.Tensor, mask: Optional[torch.Tensor] = None):
#         """
#         wav_16k: (B, L) float
#         mask: (B, L) 0/1, 1 表示有效采样点
#         """
#         attn_mask = None
#         if mask is not None:
#             attn_mask = mask.long()
#         out = self.hubert(input_values=wav_16k, attention_mask=attn_mask)
#         feats = out.last_hidden_state
#         return feats


def read_wav_16k_mono(path: str) -> torch.Tensor:
    """
    读 16kHz PCM wav，返回 float32 tensor (L,)
    """
    wav, sr = sf.read(path, dtype="float32", always_2d=False)
    if sr != 16000:
        raise ValueError(f"Expected 16kHz but got sr={sr} for: {path}")

    # 如果是多通道，转单通道（取平均）
    if wav.ndim == 2:
        wav = wav.mean(axis=1)

    if wav.ndim != 1:
        raise ValueError(f"Unexpected wav shape {wav.shape} for: {path}")

    return torch.from_numpy(wav)  # (L,)


def load_list(list_path: str) -> List[str]:
    items = []
    with open(list_path, "r", encoding="utf-8") as f:
        for line in f:
            s = line.strip()
            if not s or s.startswith("#"):
                continue
            s = s.split()
            items.append(s)
    return items


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--wav_list", required=True, help="输入音频列表文件，每行一个 wav 路径")
    ap.add_argument("--feat_list", required=True, help="输出特征列表文件，每行一个 feat 路径")
    ap.add_argument("--out_dir", required=True, help="特征保存目录（每条音频一个 .pt）")
    ap.add_argument("--model_name", required=True, help="Hubert 模型名或本地路径（如 chinese-hubert-large 路径）")
    ap.add_argument("--device", default="cuda", help="cuda / cpu / cuda:0 ...")
    ap.add_argument("--dtype", default="fp32", choices=["fp16", "fp32"], help="前向 dtype（建议 gpu 用 fp16）")
    ap.add_argument("--skip_if_exists", action="store_true", help="特征文件已存在则跳过")
    args = ap.parse_args()

    wav_paths = load_list(args.wav_list)
    if len(wav_paths) == 0:
        raise RuntimeError("wav_list is empty.")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    device = args.device
    use_fp16 = (args.dtype == "fp16") and ("cuda" in device)

    hubert = FrozenHubert(args.model_name, device=device)
    hubert.eval()

    # 注意：FrozenHubert 内部已经 no_grad，但这里再包一层更稳妥
    feat_paths_out = []

    for idx, wav_item in enumerate(wav_paths):
        uttid = wav_item[0]
        wav_path = wav_item[1]
        wav_path = wav_path.strip()
        if not wav_path:
            continue

        wav_p = Path(wav_path)
        #wav_path = os.path.realpath(wav_path)
        if not os.path.exists(wav_path):
        #if not wav_p.is_file():
            raise FileNotFoundError(f"Missing wav: 121{wav_path}121")

        wav_base = os.path.basename(wav_path)
        wav_base = re.sub('\.wav$', '', wav_base)
        out_feat = out_dir / f"{wav_base}.pt"

        if args.skip_if_exists and out_feat.exists():
            feat_paths_out.append(str(out_feat))
            continue

        wav = read_wav_16k_mono(str(wav_p))  # (L,)
        if wav.numel() < 3:
            raise ValueError(f"Too short wav (len={wav.numel()}): {wav_path}")

        # (B, L)
        wav = wav.unsqueeze(0)

        # mask: 1 表示有效采样点
        mask = torch.ones_like(wav, dtype=torch.long)

        wav = wav.to(device)
        mask = mask.to(device)

        if use_fp16:
            wav = wav.half()
        else:
            wav = wav.float()

        with torch.no_grad():
            feats = hubert(wav, mask=mask)  # (1, T, D)
            feats = feats.squeeze(0).contiguous().cpu()  # (T, D)

        # 保存成 .pt（torch tensor）
        torch.save(
            {
                "wav_path": str(wav_p),
                "feat": feats,          # (T, D)
                "feat_dim": feats.shape[-1],
                "num_frames": feats.shape[0],
                "model": args.model_name,
            },
            str(out_feat),
        )

        feat_paths_out.append([uttid, str(out_feat)])

        if (idx + 1) % 2000 == 0:
            print(f"[{idx+1}/{len(wav_paths)}] done")

    # 写 feat_list（顺序与输入一致）
    feat_list_path = Path(args.feat_list)
    feat_list_path.parent.mkdir(parents=True, exist_ok=True)
    with open(feat_list_path, "w", encoding="utf-8") as f:
        for p in feat_paths_out:
            f.write(' '.join(p) + "\n")

    print(f"Done. Wrote feat list: {feat_list_path}")
    print(f"Feats saved under: {out_dir}")


if __name__ == "__main__":
    main()
