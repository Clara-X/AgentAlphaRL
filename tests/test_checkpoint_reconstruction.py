from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

try:
    import torch
    from safetensors import safe_open
    from safetensors.torch import save_file
except ImportError:  # pragma: no cover - exercised when dependencies are missing
    torch = None
    safe_open = None
    save_file = None

from alpharl.checkpoint_reconstruction import ReconstructionConfig, reconstruct_checkpoint


def write_index(model_dir: Path, weight_map: dict[str, str], total_size: int) -> None:
    payload = {"metadata": {"total_size": total_size}, "weight_map": weight_map}
    (model_dir / "model.safetensors.index.json").write_text(
        json.dumps(payload, indent=2) + "\n",
        encoding="utf-8",
    )


@unittest.skipIf(torch is None or safe_open is None or save_file is None, "requires torch and safetensors")
class ReconstructionCheckpointTest(unittest.TestCase):
    def test_reconstruct_checkpoint_preserves_layout_and_support_files(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            base_dir = root / "base"
            trained_dir = root / "trained"
            output_dir = root / "approx"
            base_dir.mkdir()
            trained_dir.mkdir()

            base_weight = torch.tensor(
                [
                    [1.0, 0.0, 0.0, 0.0],
                    [0.0, 1.0, 0.0, 0.0],
                    [0.0, 0.0, 1.0, 0.0],
                    [0.0, 0.0, 0.0, 1.0],
                ],
                dtype=torch.bfloat16,
            )
            delta = torch.tensor(
                [
                    [2.0, 0.0, 0.0, 0.0],
                    [4.0, 0.0, 0.0, 0.0],
                    [6.0, 0.0, 0.0, 0.0],
                    [8.0, 0.0, 0.0, 0.0],
                ],
                dtype=torch.bfloat16,
            )
            trained_weight = base_weight + delta
            base_norm = torch.tensor([1.0, 2.0, 3.0, 4.0], dtype=torch.bfloat16)
            trained_norm = torch.tensor([5.0, 6.0, 7.0, 8.0], dtype=torch.bfloat16)

            base_shard_1 = {"linear.weight": base_weight}
            base_shard_2 = {"norm.weight": base_norm}
            trained_shard_1 = {"norm.weight": trained_norm}
            trained_shard_2 = {"linear.weight": trained_weight}

            save_file(base_shard_1, str(base_dir / "model-00001-of-00002.safetensors"), metadata={"format": "pt"})
            save_file(base_shard_2, str(base_dir / "model-00002-of-00002.safetensors"), metadata={"format": "pt"})
            save_file(
                trained_shard_1,
                str(trained_dir / "model-00001-of-00002.safetensors"),
                metadata={"format": "pt"},
            )
            save_file(
                trained_shard_2,
                str(trained_dir / "model-00002-of-00002.safetensors"),
                metadata={"format": "pt"},
            )

            element_size = torch.tensor([], dtype=torch.bfloat16).element_size()
            total_size = (base_weight.numel() + base_norm.numel()) * element_size
            write_index(
                base_dir,
                {
                    "linear.weight": "model-00001-of-00002.safetensors",
                    "norm.weight": "model-00002-of-00002.safetensors",
                },
                total_size=total_size,
            )
            write_index(
                trained_dir,
                {
                    "linear.weight": "model-00002-of-00002.safetensors",
                    "norm.weight": "model-00001-of-00002.safetensors",
                },
                total_size=total_size,
            )

            (trained_dir / "config.json").write_text('{"model_type": "toy"}\n', encoding="utf-8")
            (trained_dir / "README.md").write_text("trained model\n", encoding="utf-8")
            (trained_dir / "added_tokens.json").write_text("{}\n", encoding="utf-8")
            (base_dir / "LICENSE").write_text("license text\n", encoding="utf-8")

            reconstruct_checkpoint(
                ReconstructionConfig(
                    base_model_path=base_dir,
                    trained_model_path=trained_dir,
                    output_path=output_dir,
                    keep_ratio=0.25,
                    device="cpu",
                    svd_oversample=1,
                    svd_niter=2,
                    seed=7,
                )
            )

            output_index = json.loads((output_dir / "model.safetensors.index.json").read_text(encoding="utf-8"))
            self.assertEqual(
                output_index["weight_map"],
                {
                    "linear.weight": "model-00002-of-00002.safetensors",
                    "norm.weight": "model-00001-of-00002.safetensors",
                },
            )

            with safe_open(str(output_dir / "model-00002-of-00002.safetensors"), framework="pt", device="cpu") as handle:
                reconstructed_weight = handle.get_tensor("linear.weight")
            with safe_open(str(output_dir / "model-00001-of-00002.safetensors"), framework="pt", device="cpu") as handle:
                reconstructed_norm = handle.get_tensor("norm.weight")

            self.assertEqual(reconstructed_weight.dtype, torch.bfloat16)
            self.assertTrue(torch.equal(reconstructed_weight, trained_weight))
            self.assertTrue(torch.equal(reconstructed_norm, trained_norm))

            self.assertTrue((output_dir / "config.json").exists())
            self.assertTrue((output_dir / "README.md").exists())
            self.assertTrue((output_dir / "added_tokens.json").exists())
            self.assertTrue((output_dir / "LICENSE").exists())

            stats = json.loads((output_dir / "reconstruction_stats.json").read_text(encoding="utf-8"))
            self.assertEqual(stats["selected_device"], "cpu")
            self.assertEqual(stats["matrix_tensors"], 1)
            self.assertEqual(stats["passthrough_tensors"], 1)
            self.assertEqual(stats["tensors"]["linear.weight"]["kept_rank"], 1)
            self.assertAlmostEqual(stats["tensors"]["linear.weight"]["relative_frobenius_error"], 0.0, places=6)
            self.assertEqual(stats["tensors"]["norm.weight"]["mode"], "passthrough")


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
