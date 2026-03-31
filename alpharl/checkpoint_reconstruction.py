from __future__ import annotations

import argparse
import json
import math
import random
import shutil
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

try:
    import torch
except ImportError:  # pragma: no cover - exercised in dependency checks
    torch = None

try:
    from safetensors import safe_open
    from safetensors.torch import save_file
except ImportError:  # pragma: no cover - exercised in dependency checks
    safe_open = None
    save_file = None


WEIGHT_INDEX_NAME = "model.safetensors.index.json"
SAFE_TENSORS_METADATA = {"format": "pt"}


@dataclass(frozen=True)
class ReconstructionConfig:
    base_model_path: Path
    trained_model_path: Path
    output_path: Path
    keep_ratio: float
    device: str = "cuda:0"
    svd_oversample: int = 16
    svd_niter: int = 2
    seed: int = 0


class SafeTensorReader:
    """Caches shard handles so tensors can be fetched by name efficiently."""

    def __init__(self, model_dir: Path, weight_map: dict[str, str]) -> None:
        self.model_dir = model_dir
        self.weight_map = weight_map
        self._contexts: dict[str, Any] = {}
        self._handles: dict[str, Any] = {}

    def get_tensor(self, tensor_name: str):
        shard_name = self.weight_map[tensor_name]
        handle = self._handles.get(shard_name)
        if handle is None:
            context = safe_open(str(self.model_dir / shard_name), framework="pt", device="cpu")
            handle = context.__enter__()
            self._contexts[shard_name] = context
            self._handles[shard_name] = handle
        return handle.get_tensor(tensor_name)

    def close(self) -> None:
        for context in self._contexts.values():
            context.__exit__(None, None, None)
        self._contexts.clear()
        self._handles.clear()

    def __enter__(self) -> "SafeTensorReader":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()


def require_dependencies() -> None:
    missing = []
    if torch is None:
        missing.append("torch")
    if safe_open is None or save_file is None:
        missing.append("safetensors")
    if missing:
        joined = ", ".join(missing)
        raise RuntimeError(
            f"Missing required dependencies: {joined}. "
            "Install them before running reconstruction."
        )


def read_weight_index(model_dir: Path) -> dict[str, Any]:
    index_path = model_dir / WEIGHT_INDEX_NAME
    if not index_path.exists():
        raise FileNotFoundError(f"Missing weight index: {index_path}")
    with index_path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def group_tensors_by_shard(weight_map: dict[str, str]) -> dict[str, list[str]]:
    grouped: dict[str, list[str]] = {}
    for tensor_name, shard_name in weight_map.items():
        grouped.setdefault(shard_name, []).append(tensor_name)
    return grouped


def validate_config(config: ReconstructionConfig) -> None:
    if not 0 < config.keep_ratio <= 1:
        raise ValueError("--keep-ratio must be in the interval (0, 1].")
    if config.svd_oversample < 0:
        raise ValueError("--svd-oversample must be non-negative.")
    if config.svd_niter < 0:
        raise ValueError("--svd-niter must be non-negative.")

    resolved_output = config.output_path.resolve()
    if resolved_output == config.base_model_path.resolve():
        raise ValueError("--output-path must be different from --base-model-path.")
    if resolved_output == config.trained_model_path.resolve():
        raise ValueError("--output-path must be different from --trained-model-path.")


def select_device(requested_device: str) -> str:
    require_dependencies()
    if requested_device.startswith("cuda"):
        if torch.cuda.is_available():
            try:
                torch.empty(0, device=requested_device)
                return requested_device
            except Exception:
                return "cpu"
        return "cpu"
    return requested_device


def seed_everything(seed: int) -> None:
    require_dependencies()
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def copy_support_files(base_model_path: Path, trained_model_path: Path, output_path: Path) -> list[str]:
    copied_files: list[str] = []
    output_path.mkdir(parents=True, exist_ok=True)

    for item in trained_model_path.iterdir():
        if not item.is_file():
            continue
        if item.name == WEIGHT_INDEX_NAME or item.name.endswith(".safetensors"):
            continue
        destination = output_path / item.name
        shutil.copy2(item, destination)
        copied_files.append(item.name)

    base_license = base_model_path / "LICENSE"
    output_license = output_path / "LICENSE"
    if base_license.exists() and not output_license.exists():
        shutil.copy2(base_license, output_license)
        copied_files.append(base_license.name)

    return sorted(set(copied_files))


def build_reconstructed_tensor(
    base_tensor,
    trained_tensor,
    keep_ratio: float,
    device: str,
    svd_oversample: int,
    svd_niter: int,
) -> tuple[Any, dict[str, Any]]:
    require_dependencies()

    start_time = time.perf_counter()
    stats: dict[str, Any] = {
        "shape": list(trained_tensor.shape),
        "dtype": str(trained_tensor.dtype).replace("torch.", ""),
    }

    if trained_tensor.ndim != 2 or not torch.is_floating_point(trained_tensor):
        stats.update(
            {
                "mode": "passthrough",
                "kept_rank": None,
                "relative_frobenius_error": 0.0,
                "elapsed_seconds": time.perf_counter() - start_time,
            }
        )
        return trained_tensor.contiguous(), stats

    base_fp32 = base_tensor.to(device=device, dtype=torch.float32)
    trained_fp32 = trained_tensor.to(device=device, dtype=torch.float32)
    delta = trained_fp32 - base_fp32
    rows, cols = delta.shape
    min_dim = min(rows, cols)
    kept_rank = max(1, math.ceil(keep_ratio * min_dim))
    q = min(min_dim, kept_rank + svd_oversample)

    with torch.inference_mode():
        left, singular_values, right = torch.svd_lowrank(delta, q=q, niter=svd_niter)
        left = left[:, :kept_rank]
        singular_values = singular_values[:kept_rank]
        right = right[:, :kept_rank]
        delta_reconstructed = (left * singular_values) @ right.transpose(0, 1)
        approximate = base_fp32 + delta_reconstructed

    delta_norm = torch.linalg.norm(delta).item()
    residual_norm = torch.linalg.norm(delta - delta_reconstructed).item()
    relative_error = 0.0 if delta_norm == 0.0 else residual_norm / delta_norm

    reconstructed = approximate.to(dtype=trained_tensor.dtype, device="cpu").contiguous()
    stats.update(
        {
            "mode": "svd",
            "kept_rank": kept_rank,
            "relative_frobenius_error": relative_error,
            "elapsed_seconds": time.perf_counter() - start_time,
        }
    )

    del base_fp32
    del trained_fp32
    del delta
    del left
    del singular_values
    del right
    del delta_reconstructed
    del approximate
    if device.startswith("cuda"):
        torch.cuda.empty_cache()

    return reconstructed, stats


def write_weight_index(output_path: Path, trained_index: dict[str, Any]) -> None:
    index_path = output_path / WEIGHT_INDEX_NAME
    with index_path.open("w", encoding="utf-8") as handle:
        json.dump(trained_index, handle, indent=2)
        handle.write("\n")


def write_reconstruction_stats(
    output_path: Path,
    config: ReconstructionConfig,
    selected_device: str,
    copied_files: list[str],
    tensor_stats: dict[str, dict[str, Any]],
    elapsed_seconds: float,
) -> None:
    matrix_tensor_stats = [stats for stats in tensor_stats.values() if stats["mode"] == "svd"]
    payload = {
        "base_model_path": str(config.base_model_path),
        "trained_model_path": str(config.trained_model_path),
        "output_path": str(config.output_path),
        "keep_ratio": config.keep_ratio,
        "requested_device": config.device,
        "selected_device": selected_device,
        "svd_oversample": config.svd_oversample,
        "svd_niter": config.svd_niter,
        "seed": config.seed,
        "copied_support_files": copied_files,
        "total_tensors": len(tensor_stats),
        "matrix_tensors": len(matrix_tensor_stats),
        "passthrough_tensors": len(tensor_stats) - len(matrix_tensor_stats),
        "elapsed_seconds": elapsed_seconds,
        "tensors": tensor_stats,
    }
    stats_path = output_path / "reconstruction_stats.json"
    with stats_path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)
        handle.write("\n")


def reconstruct_checkpoint(config: ReconstructionConfig) -> None:
    require_dependencies()
    validate_config(config)
    seed_everything(config.seed)

    base_index = read_weight_index(config.base_model_path)
    trained_index = read_weight_index(config.trained_model_path)
    base_weight_map = base_index["weight_map"]
    trained_weight_map = trained_index["weight_map"]
    if set(base_weight_map) != set(trained_weight_map):
        raise ValueError("Base and trained checkpoints must contain the same tensor keys.")

    selected_device = select_device(config.device)
    grouped_trained_shards = group_tensors_by_shard(trained_weight_map)

    config.output_path.mkdir(parents=True, exist_ok=True)
    copied_files = copy_support_files(
        base_model_path=config.base_model_path,
        trained_model_path=config.trained_model_path,
        output_path=config.output_path,
    )

    started_at = time.perf_counter()
    tensor_stats: dict[str, dict[str, Any]] = {}

    with SafeTensorReader(config.base_model_path, base_weight_map) as base_reader, SafeTensorReader(
        config.trained_model_path, trained_weight_map
    ) as trained_reader:
        for shard_name, tensor_names in grouped_trained_shards.items():
            output_tensors = {}
            for tensor_name in tensor_names:
                base_tensor = base_reader.get_tensor(tensor_name)
                trained_tensor = trained_reader.get_tensor(tensor_name)
                reconstructed_tensor, stats = build_reconstructed_tensor(
                    base_tensor=base_tensor,
                    trained_tensor=trained_tensor,
                    keep_ratio=config.keep_ratio,
                    device=selected_device,
                    svd_oversample=config.svd_oversample,
                    svd_niter=config.svd_niter,
                )
                output_tensors[tensor_name] = reconstructed_tensor
                tensor_stats[tensor_name] = stats

            save_file(
                output_tensors,
                str(config.output_path / shard_name),
                metadata=SAFE_TENSORS_METADATA,
            )

    write_weight_index(config.output_path, trained_index)
    write_reconstruction_stats(
        output_path=config.output_path,
        config=config,
        selected_device=selected_device,
        copied_files=copied_files,
        tensor_stats=tensor_stats,
        elapsed_seconds=time.perf_counter() - started_at,
    )


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Reconstruct an approximate checkpoint by keeping the top-k singular "
            "subspace of each 2D parameter update."
        )
    )
    parser.add_argument("--base-model-path", type=Path, required=True)
    parser.add_argument("--trained-model-path", type=Path, required=True)
    parser.add_argument("--output-path", type=Path, required=True)
    parser.add_argument("--keep-ratio", type=float, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--svd-oversample", type=int, default=16)
    parser.add_argument("--svd-niter", type=int, default=2)
    parser.add_argument("--seed", type=int, default=0)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_argument_parser()
    args = parser.parse_args(argv)

    config = ReconstructionConfig(
        base_model_path=args.base_model_path,
        trained_model_path=args.trained_model_path,
        output_path=args.output_path,
        keep_ratio=args.keep_ratio,
        device=args.device,
        svd_oversample=args.svd_oversample,
        svd_niter=args.svd_niter,
        seed=args.seed,
    )
    reconstruct_checkpoint(config)
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
