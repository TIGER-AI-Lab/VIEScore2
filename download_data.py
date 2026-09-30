#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Data Download Helper for VIEScore2 SFT Construction

Downloads and prepares the following datasets:
- ImagenWorld-annotated-set (from HuggingFace)
- RichHF-18K (from GitHub)
- ImageReward (from HuggingFace)

Usage:
    python download_data.py --datasets all
    python download_data.py --datasets imagenworld richhf
    python download_data.py --datasets imagenworld --unzip
"""

import argparse
import subprocess
import os
from pathlib import Path
from zipfile import ZipFile
import logging

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent
DATA_DIR = PROJECT_ROOT / "data"

# Dataset configurations
DATASETS = {
    "imagenworld": {
        "name": "ImagenWorld-annotated-set",
        "repo_id": os.environ.get("IMAGENWORLD_REPO", "<official ImagenWorld HF repo id>"),  # set IMAGENWORLD_REPO to the dataset's official HF id
        "type": "huggingface",
        "description": "Multi-task human ratings for image generation/editing",
    },
    "richhf": {
        "name": "RichHF-18K",
        "repo_id": "Exploration/richhf_18k_with_images",
        "type": "huggingface",
        "description": "Fine-grained human feedback with heatmaps and images",
    },
    "imagereward": {
        "name": "ImageRewardDB",
        "repo_id": "THUDM/ImageRewardDB",
        "type": "huggingface-datasets",
        "description": "Human preference rankings for generated images",
    },
    "imagenhub": {
        "name": "ImagenHub",
        "repo_id": "ImagenHub/Text_to_Image",
        "type": "huggingface-datasets",
        "description": "Benchmark prompts for text-to-image generation",
    },
    "coco": {
        "name": "COCO",
        "repo_id": "detection-datasets/coco",
        "type": "huggingface-datasets",
        "description": "Real-world photos for no-error baseline samples",
    },
}


def ensure_dirs():
    """Create necessary directories."""
    DATA_DIR.mkdir(exist_ok=True, parents=True)


def check_git_lfs():
    """Check if git-lfs is installed."""
    try:
        result = subprocess.run(
            ["git", "lfs", "version"],
            capture_output=True,
            text=True
        )
        return result.returncode == 0
    except FileNotFoundError:
        return False


def download_imagenworld(unzip: bool = True):
    """Download ImagenWorld-annotated-set from HuggingFace."""
    logger.info("Downloading ImagenWorld-annotated-set...")

    try:
        from huggingface_hub import snapshot_download
    except ImportError:
        logger.error("huggingface_hub not installed. Install with: pip install huggingface_hub")
        return False

    target_dir = DATA_DIR / "ImagenWorld-annotated-set"

    try:
        local_path = snapshot_download(
            repo_id=DATASETS["imagenworld"]["repo_id"],
            repo_type="dataset",
            local_dir=str(target_dir),
            local_dir_use_symlinks=False,
        )
        logger.info(f"Downloaded to: {local_path}")

        # Unzip if requested
        if unzip:
            logger.info("Extracting zip files...")
            for split in ["train", "test"]:
                split_dir = target_dir / split
                if not split_dir.exists():
                    continue

                for zip_file in split_dir.glob("*.zip"):
                    extract_dir = split_dir / zip_file.stem
                    extract_dir.mkdir(exist_ok=True)

                    logger.info(f"Extracting {zip_file.name}...")
                    with ZipFile(zip_file, "r") as zf:
                        zf.extractall(extract_dir)

        return True

    except Exception as e:
        logger.error(f"Failed to download ImagenWorld: {e}")
        return False


def download_richhf():
    """Download RichHF-18K from HuggingFace."""
    logger.info("Downloading RichHF-18K from HuggingFace...")

    try:
        from huggingface_hub import snapshot_download
    except ImportError:
        logger.error("huggingface_hub not installed. Install with: pip install huggingface_hub")
        return False

    target_dir = DATA_DIR / "richhf-18k"

    # Check if already downloaded
    if target_dir.exists() and any(target_dir.glob("*.parquet")):
        logger.info("RichHF-18K already downloaded")
        return True

    try:
        local_path = snapshot_download(
            repo_id=DATASETS["richhf"]["repo_id"],
            repo_type="dataset",
            local_dir=str(target_dir),
            local_dir_use_symlinks=False,
        )
        logger.info(f"Downloaded to: {local_path}")
        return True

    except Exception as e:
        logger.error(f"Failed to download RichHF-18K: {e}")
        return False


def download_imagereward():
    """Download ImageReward dataset (pre-cached by HuggingFace datasets)."""
    logger.info("Downloading ImageReward dataset...")

    try:
        from datasets import load_dataset
    except ImportError:
        logger.error("datasets library not installed. Install with: pip install datasets")
        return False

    try:
        # This will cache the dataset
        ds = load_dataset(DATASETS["imagereward"]["repo_id"], split="train")
        logger.info(f"ImageReward dataset loaded and cached ({len(ds)} samples)")
        logger.info("Dataset is cached in ~/.cache/huggingface/datasets/")
        return True

    except Exception as e:
        logger.error(f"Failed to download ImageReward: {e}")
        return False


def download_imagenhub():
    """Download ImagenHub prompts (pre-cached by HuggingFace datasets)."""
    logger.info("Downloading ImagenHub prompts...")

    try:
        from datasets import load_dataset
    except ImportError:
        logger.error("datasets library not installed. Install with: pip install datasets")
        return False

    try:
        ds = load_dataset(DATASETS["imagenhub"]["repo_id"], split="eval")
        logger.info(f"ImagenHub prompts loaded and cached ({len(ds)} samples)")
        return True

    except Exception as e:
        logger.error(f"Failed to download ImagenHub: {e}")
        return False


def download_pickapic_metadata():
    """
    Download Pick-a-Pic v1 metadata for joining with RichHF-18K.

    RichHF-18K images come from Pick-a-Pic, and filenames can be used
    to retrieve the original prompts.
    """
    logger.info("Downloading Pick-a-Pic metadata for RichHF prompt joining...")

    try:
        from datasets import load_dataset
    except ImportError:
        logger.error("datasets library not installed")
        return False

    try:
        # Pick-a-Pic v1 dataset
        ds = load_dataset("yuvalkirstain/PickaPic", split="train")
        logger.info(f"Pick-a-Pic v1 loaded ({len(ds)} samples)")

        # Save mapping file for later use
        mapping_file = DATA_DIR / "pickapic_filename_to_prompt.json"

        import json
        mapping = {}
        for ex in ds:
            if "image_0_uid" in ex:
                mapping[ex["image_0_uid"]] = ex.get("caption", "")
            if "image_1_uid" in ex:
                mapping[ex["image_1_uid"]] = ex.get("caption", "")

        with open(mapping_file, "w") as f:
            json.dump(mapping, f)

        logger.info(f"Saved filename-to-prompt mapping to {mapping_file}")
        return True

    except Exception as e:
        logger.error(f"Failed to download Pick-a-Pic: {e}")
        return False


def show_status():
    """Show download status of all datasets."""
    logger.info("=" * 50)
    logger.info("Dataset Status")
    logger.info("=" * 50)

    # ImagenWorld
    iw_path = DATA_DIR / "ImagenWorld-annotated-set"
    iw_train = iw_path / "train"
    if iw_train.exists():
        task_count = sum(1 for t in ["TIG", "TIE", "SRIG", "SRIE", "MRIG", "MRIE"]
                        if (iw_train / t).exists())
        logger.info(f"ImagenWorld: DOWNLOADED ({task_count}/6 tasks found)")
    else:
        logger.info("ImagenWorld: NOT DOWNLOADED")

    # RichHF-18K
    richhf_path = DATA_DIR / "richhf-18k"
    parquet_files = list(richhf_path.glob("**/*.parquet")) if richhf_path.exists() else []
    if parquet_files:
        logger.info(f"RichHF-18K: DOWNLOADED ({len(parquet_files)} parquet files)")
    elif (richhf_path / "train.tfrecord").exists():
        logger.info("RichHF-18K: DOWNLOADED (legacy tfrecord format)")
    else:
        logger.info("RichHF-18K: NOT DOWNLOADED")

    # ImageReward (cached by datasets)
    try:
        from datasets import load_dataset
        ds = load_dataset(DATASETS["imagereward"]["repo_id"], split="train")
        logger.info(f"ImageReward: CACHED ({len(ds)} samples)")
    except Exception:
        logger.info("ImageReward: NOT DOWNLOADED")

    # ImagenHub
    try:
        from datasets import load_dataset
        ds = load_dataset(DATASETS["imagenhub"]["repo_id"], split="eval")
        logger.info(f"ImagenHub: CACHED ({len(ds)} samples)")
    except Exception:
        logger.info("ImagenHub: NOT DOWNLOADED")

    # Pick-a-Pic mapping
    mapping_file = DATA_DIR / "pickapic_filename_to_prompt.json"
    if mapping_file.exists():
        logger.info("Pick-a-Pic mapping: AVAILABLE")
    else:
        logger.info("Pick-a-Pic mapping: NOT AVAILABLE")

    logger.info("=" * 50)


def main():
    parser = argparse.ArgumentParser(
        description="Download datasets for VIEScore2 SFT construction"
    )
    parser.add_argument(
        "--datasets",
        nargs="+",
        default=["status"],
        choices=["all", "imagenworld", "richhf", "imagereward", "imagenhub",
                 "coco", "pickapic", "status"],
        help="Which datasets to download"
    )
    parser.add_argument(
        "--unzip",
        action="store_true",
        default=True,
        help="Unzip downloaded archives (default: True)"
    )
    parser.add_argument(
        "--no-unzip",
        action="store_false",
        dest="unzip",
        help="Don't unzip downloaded archives"
    )

    args = parser.parse_args()

    ensure_dirs()

    datasets = args.datasets

    if "status" in datasets:
        show_status()
        if len(datasets) == 1:
            return

    if "all" in datasets:
        datasets = ["imagenworld", "richhf", "imagereward", "imagenhub", "coco"]

    results = {}

    if "imagenworld" in datasets:
        results["imagenworld"] = download_imagenworld(unzip=args.unzip)

    if "richhf" in datasets:
        results["richhf"] = download_richhf()

    if "imagereward" in datasets:
        results["imagereward"] = download_imagereward()

    if "imagenhub" in datasets:
        results["imagenhub"] = download_imagenhub()

    if "pickapic" in datasets:
        results["pickapic"] = download_pickapic_metadata()

    # Summary
    if results:
        logger.info("=" * 50)
        logger.info("Download Summary")
        logger.info("=" * 50)
        for name, success in results.items():
            status = "SUCCESS" if success else "FAILED"
            logger.info(f"  {name}: {status}")


if __name__ == "__main__":
    main()
