"""Fetch immutable public data artifacts from a checksum-pinned TOML manifest."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import tomllib
import urllib.request
from collections.abc import Iterable, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path, PurePosixPath
from urllib.parse import urlparse


@dataclass(frozen=True, slots=True)
class DataArtifact:
    name: str
    url: str
    relative_path: str
    sha256: str
    bytes: int
    source_repository: str
    source_commit: str
    license: str
    training_status: str
    description: str = ""

    def __post_init__(self) -> None:
        if not self.name or not self.url:
            raise ValueError("artifact name and URL are required")
        path = PurePosixPath(self.relative_path)
        if path.is_absolute() or ".." in path.parts or not path.parts:
            raise ValueError(f"artifact {self.name!r} has unsafe relative_path")
        if len(self.sha256) != 64 or any(char not in "0123456789abcdef" for char in self.sha256):
            raise ValueError(f"artifact {self.name!r} has invalid SHA-256")
        if self.bytes <= 0:
            raise ValueError(f"artifact {self.name!r} byte count must be positive")
        if self.training_status not in {"approved", "review_required", "evaluation_only"}:
            raise ValueError(f"artifact {self.name!r} has invalid training_status")
        if urlparse(self.url).scheme not in {"https", "file"}:
            raise ValueError(f"artifact {self.name!r} URL must use HTTPS or file scheme")


@dataclass(frozen=True, slots=True)
class DataManifest:
    schema_version: int
    artifacts: tuple[DataArtifact, ...]
    path: Path
    sha256: str


@dataclass(frozen=True, slots=True)
class FetchResult:
    artifact: DataArtifact
    path: Path
    downloaded: bool


def load_data_manifest(path: str | Path) -> DataManifest:
    manifest_path = Path(path)
    raw = manifest_path.read_bytes()
    document = tomllib.loads(raw.decode("utf-8"))
    if document.get("schema_version") != 1:
        raise ValueError("data manifest schema_version must be 1")
    raw_artifacts = document.get("artifact")
    if not isinstance(raw_artifacts, list) or not raw_artifacts:
        raise ValueError("data manifest requires at least one [[artifact]]")
    artifacts = tuple(DataArtifact(**values) for values in raw_artifacts)
    names = [artifact.name for artifact in artifacts]
    paths = [artifact.relative_path for artifact in artifacts]
    if len(names) != len(set(names)):
        raise ValueError("artifact names must be unique")
    if len(paths) != len(set(paths)):
        raise ValueError("artifact relative paths must be unique")
    return DataManifest(
        schema_version=1,
        artifacts=artifacts,
        path=manifest_path,
        sha256=hashlib.sha256(raw).hexdigest(),
    )


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify_artifact(path: str | Path, artifact: DataArtifact) -> None:
    candidate = Path(path)
    actual_size = candidate.stat().st_size
    if actual_size != artifact.bytes:
        raise ValueError(
            f"{artifact.name}: expected {artifact.bytes} bytes, found {actual_size} at {candidate}"
        )
    actual_sha = sha256_file(candidate)
    if actual_sha != artifact.sha256:
        raise ValueError(
            f"{artifact.name}: checksum mismatch at {candidate}; "
            f"expected {artifact.sha256}, found {actual_sha}"
        )


def fetch_artifacts(
    manifest: DataManifest,
    *,
    output_root: str | Path,
    names: Iterable[str] | None = None,
    verify_only: bool = False,
) -> tuple[FetchResult, ...]:
    """Download selected artifacts atomically or verify existing snapshots."""

    root = Path(output_root).resolve()
    requested = None if names is None else set(names)
    known = {artifact.name for artifact in manifest.artifacts}
    if requested is not None and not requested <= known:
        raise ValueError(f"unknown artifact name(s): {sorted(requested - known)}")
    selected = tuple(
        artifact
        for artifact in manifest.artifacts
        if requested is None or artifact.name in requested
    )
    if not selected:
        raise ValueError("no data artifacts selected")
    root.mkdir(parents=True, exist_ok=True)
    results: list[FetchResult] = []

    for artifact in selected:
        destination = (root / artifact.relative_path).resolve()
        if root != destination and root not in destination.parents:
            raise ValueError(f"artifact {artifact.name!r} resolves outside output root")
        if destination.exists():
            verify_artifact(destination, artifact)
            results.append(FetchResult(artifact, destination, downloaded=False))
            continue
        if verify_only:
            raise FileNotFoundError(f"missing artifact {artifact.name!r}: {destination}")
        destination.parent.mkdir(parents=True, exist_ok=True)
        temporary = destination.with_name(destination.name + ".partial")
        if temporary.exists():
            temporary.unlink()
        request = urllib.request.Request(
            artifact.url,
            headers={"User-Agent": "amp-challenge-data-fetch/0.1"},
        )
        try:
            with (
                urllib.request.urlopen(request, timeout=60) as response,
                temporary.open("wb") as handle,
            ):
                while chunk := response.read(1024 * 1024):
                    handle.write(chunk)
            verify_artifact(temporary, artifact)
            temporary.replace(destination)
        except Exception:
            if temporary.exists():
                temporary.unlink()
            raise
        results.append(FetchResult(artifact, destination, downloaded=True))

    lock = {
        "schema_version": 1,
        "manifest": str(manifest.path),
        "manifest_sha256": manifest.sha256,
        "artifacts": [
            {
                **asdict(result.artifact),
                "resolved_path": str(result.path),
            }
            for result in results
        ],
    }
    (root / "snapshot.lock.json").write_text(
        json.dumps(lock, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return tuple(results)


def _default_output_root() -> Path | None:
    raw = os.environ.get("AMP_DATA_ROOT")
    return None if not raw else Path(raw)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path("configs/data/starter_snapshots.toml"),
    )
    parser.add_argument("--output-root", type=Path, default=_default_output_root())
    parser.add_argument("--artifact", action="append", dest="artifacts")
    parser.add_argument("--verify-only", action="store_true")
    parser.add_argument("--list", action="store_true", dest="list_only")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    manifest = load_data_manifest(args.manifest)
    if args.list_only:
        for artifact in manifest.artifacts:
            print(
                f"{artifact.name}\t{artifact.training_status}\t{artifact.bytes}\t"
                f"{artifact.relative_path}"
            )
        return 0
    if args.output_root is None:
        raise SystemExit("--output-root or AMP_DATA_ROOT is required")
    results = fetch_artifacts(
        manifest,
        output_root=args.output_root,
        names=args.artifacts,
        verify_only=args.verify_only,
    )
    print(
        json.dumps(
            {
                "artifacts": len(results),
                "downloaded": sum(result.downloaded for result in results),
                "output_root": str(args.output_root),
                "manifest_sha256": manifest.sha256,
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
