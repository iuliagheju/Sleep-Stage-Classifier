"""Environment and hardware introspection helpers."""
from __future__ import annotations

import os
import platform
from typing import Dict, List

import torch


def collect_hardware_summary() -> Dict[str, object]:
    cuda_available = torch.cuda.is_available()
    cuda_device = None
    if cuda_available:
        try:
            cuda_device = torch.cuda.get_device_name(0)
        except Exception:
            cuda_device = None
    return {
        "platform": platform.platform(),
        "python_version": platform.python_version(),
        "cpu_count": os.cpu_count() or 0,
        "cuda_available": bool(cuda_available),
        "cuda_device": cuda_device,
    }


def collect_package_versions(packages: List[str]) -> Dict[str, str]:
    try:
        from importlib import metadata
    except ImportError:  # pragma: no cover
        import importlib_metadata as metadata  # type: ignore
    versions: Dict[str, str] = {}
    for name in packages:
        try:
            versions[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            versions[name] = "unknown"
    return versions
