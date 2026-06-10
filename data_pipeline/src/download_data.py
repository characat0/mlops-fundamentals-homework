"""
Download the Spotify songs dataset from Kaggle and verify its integrity.

Authentication resolution order (first match wins):
    1. KAGGLE_API_TOKEN environment variable (recommended, post-2025).
    2. File at ~/.kaggle/access_token (single-token format).
    3. Legacy file at ~/.kaggle/kaggle.json (KAGGLE_USERNAME + KAGGLE_KEY).

Loads credentials from a local .env file (project root) if present,
so the secrets stay out of the shell and the repository.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
import shutil
import subprocess
import sys
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DEST = REPO_ROOT / "songs.csv"
EXPECTED_MD5 = "0e71e2c46244acac485bd8c245aa6e56"

KAGGLE_DATASET = "serkantysz/550k-spotify-songs-audio-lyrics-and-genres"


def _load_dotenv() -> None:
    """Load variables from a .env file in the data_pipeline directory if present."""
    try:
        from dotenv import load_dotenv
    except ImportError:
        logger.debug("python-dotenv not installed; skipping .env loading")
        return
    env_path = REPO_ROOT / ".env"
    if env_path.is_file():
        load_dotenv(env_path, override=False)
        logger.info("Loaded environment variables from %s", env_path)
    else:
        logger.debug("No .env file at %s", env_path)


def _file_md5(path: Path, chunk_size: int = 1024 * 1024) -> str:
    md5 = hashlib.md5()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(chunk_size), b""):
            md5.update(chunk)
    return md5.hexdigest()


def _resolve_credentials() -> str | None:
    """Returns a short human-readable description of the auth source, or None.

    The actual credentials are read by the Kaggle CLI itself, so this
    function only reports which method will be used.
    """
    if os.environ.get("KAGGLE_API_TOKEN"):
        return "KAGGLE_API_TOKEN env var"

    access_token = Path.home() / ".kaggle" / "access_token"
    if access_token.is_file() and access_token.stat().st_size > 0:
        return f"file {access_token}"

    legacy = Path.home() / ".kaggle" / "kaggle.json"
    if legacy.is_file():
        try:
            data = json.loads(legacy.read_text())
            if data.get("username") and data.get("key"):
                return f"legacy file {legacy}"
        except (json.JSONDecodeError, OSError):
            pass

    return None


def _run_kaggle_download(dest_dir: Path) -> Path:
    """Invoke `kaggle datasets download` and return the resulting CSV path."""
    cmd = [
        "kaggle", "datasets", "download",
        "-d", KAGGLE_DATASET,
        "-p", str(dest_dir),
        "--unzip",
    ]
    logger.info("Running: %s", " ".join(cmd))
    subprocess.run(cmd, check=True)
    return dest_dir / "songs.csv"


def verify(dest: Path) -> bool:
    if not dest.is_file():
        logger.error("File not found: %s", dest)
        return False
    digest = _file_md5(dest)
    if digest != EXPECTED_MD5:
        logger.error(
            "MD5 mismatch.\n  expected: %s\n  got:      %s",
            EXPECTED_MD5, digest,
        )
        return False
    logger.info("MD5 OK: %s", digest)
    return True


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--dest", type=Path, default=DEFAULT_DEST,
        help=f"Destination path for songs.csv (default: {DEFAULT_DEST})",
    )
    parser.add_argument(
        "--check-only", action="store_true",
        help="Only verify the existing file at --dest; do not download.",
    )
    args = parser.parse_args()

    _load_dotenv()

    if args.check_only:
        return 0 if verify(args.dest) else 1

    auth = _resolve_credentials()
    if auth is None:
        logger.error(
            "No Kaggle credentials found. Provide one of:\n"
            "  - KAGGLE_API_TOKEN in .env or environment\n"
            "  - file at ~/.kaggle/access_token\n"
            "  - legacy file at ~/.kaggle/kaggle.json (username + key)\n"
            "See .env.example for details."
        )
        return 1
    logger.info("Auth source: %s", auth)

    args.dest.parent.mkdir(parents=True, exist_ok=True)
    csv_path = _run_kaggle_download(args.dest.parent)
    if csv_path.resolve() != args.dest.resolve():
        shutil.move(str(csv_path), args.dest)
    logger.info("Saved dataset to %s", args.dest)

    if not verify(args.dest):
        logger.error(
            "Downloaded file failed integrity check. "
            "Re-run with --check-only after removing the bad file."
        )
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
