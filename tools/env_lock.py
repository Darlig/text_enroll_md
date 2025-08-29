# tools/env_lock.py (py38 compatible)
import json
import os
import platform
import subprocess
import sys
from pathlib import Path
from typing import Optional, Tuple, List, Union, Dict, Any


def _run(cmd: List[str]) -> Tuple[int, str]:
    try:
        out = subprocess.check_output(cmd, stderr=subprocess.STDOUT, text=True)
        return 0, out
    except Exception as e:
        return 1, f"{e}"


def _detect_cuda_version() -> Optional[str]:
    # nvidia-smi
    code, out = _run(["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"])
    if code == 0 and out.strip():
        # driver != cuda version，但保留作参考
        drv = out.strip().splitlines()[0].strip()
        code2, out2 = _run(["nvcc", "--version"])
        if code2 == 0:
            return f"nvcc:{out2.strip().splitlines()[-1].strip()}, driver:{drv}"
        return f"driver:{drv}"
    # nvcc
    code, out = _run(["nvcc", "--version"])
    if code == 0:
        return out.strip().splitlines()[-1].strip()
    return None


def _conda_env_lock() -> Tuple[bool, Optional[str]]:
    # 优先用 conda-lock 风格；没有就退化为 conda env export
    # 1) conda 是否可用
    code, _ = _run(["conda", "--version"])
    if code != 0:
        return False, None
    # 2) 尝试导出无构建号的环境（更可移植）
    code, out = _run(["conda", "env", "export", "--no-builds"])
    if code == 0 and out.strip():
        return True, out
    # 3) 退化
    code, out = _run(["conda", "list", "--explicit"])
    if code == 0 and out.strip():
        return True, out
    return False, None


def _pip_env_lock() -> Optional[str]:
    code, out = _run([sys.executable, "-m", "pip", "freeze"])
    return out if code == 0 and out.strip() else None


def write_env_lock(run_dir: Union[str, os.PathLike], file_stem: str = "env.lock") -> Dict[str, str]:
    """
    在 run_dir 下写出：
      - env.lock.yml     (如果是 conda 环境可导出)
      - requirements.lock (pip 冻结)
      - env_meta.json    (python/平台/torch/cuda 等补充信息)
    返回写入的文件路径字典（不存在的键表示未写入）
    """
    run_dir = Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)

    written: Dict[str, str] = {}

    # 1) 基本元信息
    meta: Dict[str, Any] = {
        "python": sys.version.split()[0],
        "executable": sys.executable,
        "platform": {
            "system": platform.system(),
            "release": platform.release(),
            "machine": platform.machine(),
        },
        "cuda": _detect_cuda_version(),
        "torch": None,
        "torch_cuda": None,
    }
    try:
        import torch  # type: ignore

        meta["torch"] = torch.__version__
        meta["torch_cuda"] = {
            "is_available": torch.cuda.is_available(),
            "version": getattr(torch.version, "cuda", None),
        }
    except Exception:
        pass

    (run_dir / "env_meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    written["env_meta.json"] = str(run_dir / "env_meta.json")

    # 2) Conda（优先）
    ok, conda_text = _conda_env_lock()
    if ok and conda_text:
        # 简单判断内容是否是 YAML/explicit；统一存 .yml 也可
        suffix = ".yml" if ("name:" in conda_text or "channels:" in conda_text) else ".txt"
        p = run_dir / f"{file_stem}{suffix}"
        p.write_text(conda_text, encoding="utf-8")
        written["conda_lock"] = str(p)

    # 3) Pip 冻结（同时保留，即使有 conda）
    pip_text = _pip_env_lock()
    if pip_text:
        p = run_dir / "requirements.lock"
        p.write_text(pip_text, encoding="utf-8")
        written["pip_lock"] = str(p)

    return written