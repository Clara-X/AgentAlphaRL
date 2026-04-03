import argparse
import importlib
import json
import os

from huggingface_hub import snapshot_download


DEFAULT_BASE_REPO = "Qwen/Qwen2.5-32B"
DEFAULT_FULL_REPO = "BytedTsinghua-SIA/DAPO-Qwen-32B"


def ensure_model(repo_id, local_dir, token=None, local_files_only=False):
    os.makedirs(local_dir, exist_ok=True)
    print(f"🔽 Resolving {repo_id} -> {local_dir}")
    kwargs = {
        "repo_id": repo_id,
        "local_dir": local_dir,
        "resume_download": True,
        "local_files_only": local_files_only,
    }
    if token:
        kwargs["token"] = token
    snapshot_download(**kwargs)
    print(f"✅ Ready: {local_dir}")


def persist_tokenizer_fix(model_dir):
    config_path = os.path.join(model_dir, "tokenizer_config.json")
    if not os.path.exists(config_path):
        return

    with open(config_path, "r", encoding="utf-8") as f:
        tokenizer_config = json.load(f)

    if tokenizer_config.get("fix_mistral_regex") is True:
        return

    tokenizer_config["fix_mistral_regex"] = True
    with open(config_path, "w", encoding="utf-8") as f:
        json.dump(tokenizer_config, f, ensure_ascii=False, indent=2)
        f.write("\n")
    print(f"✅ Persisted fix_mistral_regex=True in {config_path}")


def import_runtime_modules():
    modules = {}
    for name in ("torch", "transformers", "vllm"):
        modules[name] = importlib.import_module(name)
        print(f"✅ Imported {name} {getattr(modules[name], '__version__', 'unknown')}")
    return modules


def smoke_test(model_paths):
    import_runtime_modules()
    from transformers import AutoConfig

    for label, model_path in model_paths:
        if not model_path:
            continue
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"{label} path does not exist: {model_path}")
        cfg = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
        print(
            f"✅ AutoConfig[{label}] model_type={cfg.model_type} "
            f"layers={getattr(cfg, 'num_hidden_layers', 'n/a')} "
            f"hidden={getattr(cfg, 'hidden_size', 'n/a')}"
        )


def parse_args():
    parser = argparse.ArgumentParser(description="Prepare local checkpoints for the DAPO-Qwen-32B workflow.")
    parser.add_argument("--base_repo", default=DEFAULT_BASE_REPO, help="Base model repo id.")
    parser.add_argument("--full_repo", default=DEFAULT_FULL_REPO, help="DAPO full model repo id.")
    parser.add_argument(
        "--base_dir",
        default="./dapo32b/base",
        help="Local directory for the base model.",
    )
    parser.add_argument(
        "--full_dir",
        default="./dapo32b/full",
        help="Local directory for the DAPO full model.",
    )
    parser.add_argument("--hf_token", default=None, help="Optional Hugging Face token.")
    parser.add_argument(
        "--skip-download",
        action="store_true",
        help="Skip snapshot_download and only run smoke checks.",
    )
    parser.add_argument(
        "--local-files-only",
        action="store_true",
        help="Resolve models from local cache only.",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    if not args.skip_download:
        ensure_model(
            repo_id=args.base_repo,
            local_dir=args.base_dir,
            token=args.hf_token,
            local_files_only=args.local_files_only,
        )
        ensure_model(
            repo_id=args.full_repo,
            local_dir=args.full_dir,
            token=args.hf_token,
            local_files_only=args.local_files_only,
        )
    else:
        print("⏭️  Skipping download phase.")




    print("🎉 DAPO-Qwen-32B preparation complete.")


if __name__ == "__main__":
    main()
