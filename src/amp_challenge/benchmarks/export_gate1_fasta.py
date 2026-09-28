"""CLI shim for exporting the frozen Gate-1 ESM2 input FASTA."""

from __future__ import annotations

from amp_challenge.benchmarks.esm_oof import export_main

if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(export_main())
