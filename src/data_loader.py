"""
Loader for the DOHA parsed-cases Parquet shards.

Usage
-----
    from src.data_loader import load_cases
    df = load_cases()

Or from the command line (builds/refreshes the local cache):
    python -m src.data_loader
    python -m src.data_loader --rebuild
    python -m src.data_loader --raw-dir /path/to/doha_parsed_cases

Design notes
------------
- The two raw shards (`all_cases_1.parquet`, `all_cases_2.parquet`) are
  committed in the *separate* `doha` repo (https://github.com/TeneikaAskew/doha),
  cloned as a sibling folder next to this repo (see Challenge-Project-Overview.md
  "Getting Started" -> step 1). We deliberately do NOT copy them into this repo:
  combined they are ~170MB, which is unfriendly to git and would duplicate data
  that already lives, version-controlled, in the doha repo.
- This module merges both shards into a single cached Parquet file under
  `data/processed/all_cases.parquet` so every notebook/script in this repo
  loads instantly after the first run, instead of re-reading/re-concatenating
  the raw shards every time. Only one person needs to trigger the first build;
  after that, the cache lives on each machine and reloads are near-instant.
- No cleaning or deduplication is performed here on purpose (see data/README.md
  for known data-quality quirks) -- EDA should happen on the data as shipped.
  The only columns added beyond the raw shards are `row_id` (a stable synthetic
  index) and `source_shard` (which raw file the row came from). Both are pure
  plumbing -- no existing values are modified, nothing is dropped or reordered
  beyond simple concatenation -- added so that rows/splits can be referenced
  reproducibly later (e.g. for a stable train/val/test split).
- The cache is automatically rebuilt if the raw shard files change (detected via
  size + mtime), so nobody has to remember to pass --rebuild after a `git pull`
  in the doha repo picks up new/updated shards.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import pandas as pd

# Repo layout: this file lives in <project_repo>/src/data_loader.py
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_RAW_DIR = PROJECT_ROOT.parent / "doha" / "doha_parsed_cases"
DEFAULT_CACHE_DIR = PROJECT_ROOT / "data" / "processed"
DEFAULT_CACHE_PATH = DEFAULT_CACHE_DIR / "all_cases.parquet"

RAW_DIR_ENV_VAR = "DOHA_RAW_DIR"
SHARD_GLOB = "all_cases_*.parquet"


class RawShardsNotFoundError(FileNotFoundError):
    """Raised when the raw DOHA parquet shards can't be located."""


def _resolve_raw_dir(raw_dir: str | Path | None) -> Path:
    if raw_dir is not None:
        return Path(raw_dir)
    env_dir = os.environ.get(RAW_DIR_ENV_VAR)
    if env_dir:
        return Path(env_dir)
    return DEFAULT_RAW_DIR


def _find_raw_shards(raw_dir: Path) -> list[Path]:
    shards = sorted(raw_dir.glob(SHARD_GLOB)) if raw_dir.exists() else []
    if not shards:
        raise RawShardsNotFoundError(
            f"No DOHA parquet shards found in '{raw_dir}'.\n\n"
            "Expected files like 'all_cases_1.parquet' and 'all_cases_2.parquet'.\n"
            "Fix by either:\n"
            "  1) Cloning the doha repo as a sibling folder next to this repo:\n"
            "       git clone https://github.com/TeneikaAskew/doha.git\n"
            f"     (expected at: {DEFAULT_RAW_DIR})\n"
            f"  2) Setting the {RAW_DIR_ENV_VAR} environment variable to point at\n"
            "     wherever you have doha_parsed_cases/ checked out."
        )
    return shards


def _raw_signature(shards: list[Path]) -> dict:
    """Cheap fingerprint of the raw shards (size + mtime) to detect staleness."""
    return {
        p.name: {"size": p.stat().st_size, "mtime": p.stat().st_mtime}
        for p in shards
    }


def _load_manifest(manifest_path: Path) -> dict | None:
    if not manifest_path.exists():
        return None
    try:
        return json.loads(manifest_path.read_text())
    except (json.JSONDecodeError, OSError):
        return None


def _build_cache(raw_dir: Path, cache_path: Path, manifest_path: Path) -> pd.DataFrame:
    shards = _find_raw_shards(raw_dir)
    print(f"[data_loader] Building cache from {len(shards)} raw shard(s) in '{raw_dir}':")

    frames = []
    for shard_path in shards:
        df = pd.read_parquet(shard_path)
        df["source_shard"] = shard_path.name
        print(f"  - {shard_path.name}: {len(df):,} rows")
        frames.append(df)

    merged = pd.concat(frames, ignore_index=True)
    merged.insert(0, "row_id", range(len(merged)))

    cache_path.parent.mkdir(parents=True, exist_ok=True)
    # gzip (not the pyarrow/pandas default of snappy) to match the compression the raw
    # shards already use -- keeps the on-disk cache roughly the same size as the raw
    # shards combined (~170MB) instead of ~300MB+ with snappy's lower compression ratio.
    merged.to_parquet(cache_path, index=False, compression="gzip")

    manifest = {
        "built_from": _raw_signature(shards),
        "raw_dir": str(raw_dir),
        "n_rows": len(merged),
        "n_cols": len(merged.columns),
        "columns": list(merged.columns),
        "cache_path": str(cache_path),
    }
    manifest_path.write_text(json.dumps(manifest, indent=2))

    print(f"[data_loader] Cached {len(merged):,} rows x {len(merged.columns)} cols -> '{cache_path}'")
    return merged


def load_cases(
    raw_dir: str | Path | None = None,
    cache_path: str | Path | None = None,
    force_rebuild: bool = False,
) -> pd.DataFrame:
    """
    Load the merged DOHA parsed-cases dataset (both shards, concatenated).

    On the first call on a given machine, this reads the two raw Parquet shards
    from the sibling `doha` repo clone (or wherever `raw_dir` / DOHA_RAW_DIR
    points), concatenates them, and writes a local cache to
    `data/processed/all_cases.parquet`. Every subsequent call (by anyone, in any
    notebook/script, in this repo) just reads that cache directly -- no
    re-parsing of the raw shards needed.

    The cache is automatically rebuilt if the raw shard files have changed
    (different size/mtime) since it was last built, or if `force_rebuild=True`.

    Parameters
    ----------
    raw_dir : path to the folder containing all_cases_*.parquet.
        Defaults to the DOHA_RAW_DIR env var, then '../doha/doha_parsed_cases'
        (i.e. a sibling clone of https://github.com/TeneikaAskew/doha).
    cache_path : where to read/write the merged cache. Defaults to
        'data/processed/all_cases.parquet' inside this repo.
    force_rebuild : if True, always rebuild the cache from the raw shards.

    Returns
    -------
    pd.DataFrame with every row from both shards, unmodified, plus two added
    columns:
        - row_id: stable synthetic integer index (position after concat)
        - source_shard: which raw file the row came from
    No cleaning or deduplication is applied -- see data/README.md for known
    data-quality quirks (duplicate rows, messy dates, etc.) to handle in EDA.
    """
    resolved_raw_dir = _resolve_raw_dir(raw_dir)
    resolved_cache_path = Path(cache_path) if cache_path else DEFAULT_CACHE_PATH
    manifest_path = resolved_cache_path.parent / "manifest.json"

    if not force_rebuild and resolved_cache_path.exists():
        manifest = _load_manifest(manifest_path)
        try:
            shards = _find_raw_shards(resolved_raw_dir)
            current_sig = _raw_signature(shards)
        except RawShardsNotFoundError:
            # Raw shards aren't available (e.g. doha repo not cloned) but we
            # already have a cache from a previous run -- just use it.
            print(f"[data_loader] Raw shards not found; using existing cache '{resolved_cache_path}'.")
            return pd.read_parquet(resolved_cache_path)

        if manifest is not None and manifest.get("built_from") == current_sig:
            return pd.read_parquet(resolved_cache_path)

        print("[data_loader] Raw shards changed (or no manifest found); rebuilding cache...")

    return _build_cache(resolved_raw_dir, resolved_cache_path, manifest_path)


def _main() -> int:
    parser = argparse.ArgumentParser(
        description="Build/refresh the local cache of the merged DOHA parsed-cases dataset."
    )
    parser.add_argument("--raw-dir", type=str, default=None, help="Folder containing all_cases_*.parquet")
    parser.add_argument("--cache-path", type=str, default=None, help="Output path for the merged cache")
    parser.add_argument("--rebuild", action="store_true", help="Force rebuild even if a valid cache exists")
    args = parser.parse_args()

    try:
        df = load_cases(raw_dir=args.raw_dir, cache_path=args.cache_path, force_rebuild=args.rebuild)
    except RawShardsNotFoundError as exc:
        print(f"[data_loader] ERROR: {exc}", file=sys.stderr)
        return 1

    print("\nSummary:")
    print(f"  Rows: {len(df):,}")
    print(f"  Columns: {list(df.columns)}")
    print(f"  case_type counts:\n{df['case_type'].value_counts().to_string()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(_main())
