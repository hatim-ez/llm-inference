#!/usr/bin/env python3
"""Download model weights from Hugging Face."""

import argparse
import os
import sys
from pathlib import Path

from huggingface_hub import snapshot_download


def main():
    parser = argparse.ArgumentParser(
        description="Download LLM model weights from Hugging Face"
    )
    parser.add_argument(
        "--model",
        type=str,
        default="meta-llama/Llama-3.2-11B-Vision-Instruct",
        help="Model name on Hugging Face",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="/home/ubuntu/models/llama-3.2-11b-vision",
        help="Directory to save model weights",
    )
    parser.add_argument(
        "--token",
        type=str,
        default=None,
        help="Hugging Face token (or set HF_TOKEN env var)",
    )
    parser.add_argument(
        "--revision",
        type=str,
        default="main",
        help="Model revision/branch",
    )

    args = parser.parse_args()

    # Get token from args or environment
    token = args.token or os.environ.get("HF_TOKEN")

    print(f"Downloading model: {args.model}")
    print(f"Output directory: {args.output_dir}")

    # Create output directory
    output_path = Path(args.output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    try:
        # Download model
        snapshot_download(
            repo_id=args.model,
            local_dir=args.output_dir,
            revision=args.revision,
            token=token,
            resume_download=True,
        )
        print(f"\nModel downloaded successfully to: {args.output_dir}")

        # List downloaded files
        print("\nDownloaded files:")
        total_size = 0
        for path in sorted(output_path.rglob("*")):
            if path.is_file():
                size_mb = path.stat().st_size / (1024 * 1024)
                total_size += size_mb
                print(f"  {path.relative_to(output_path)}: {size_mb:.2f} MB")

        print(f"\nTotal size: {total_size / 1024:.2f} GB")

    except Exception as e:
        print(f"Error downloading model: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
