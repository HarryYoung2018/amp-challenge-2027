"""Label-free ESM2 residue/contact features; compatible with the pinned Python 3.10 worker.

This is a data adapter, not a fitting or scientific acceptance interface. NPY
payloads never require pickle. Means are computed from saved float32 residue
tokens using float64 accumulation, then rounded once to float32.
"""

from __future__ import annotations

import hashlib
import io
import json
from dataclasses import asdict
from pathlib import Path

import numpy as np

from amp_challenge.representations.laplacian import (
    LaplacianLogSpectralDensityConfig,
    laplacian_log_spectral_densities,
)

ALPHABET = frozenset("ACDEFGHIKLMNPQRSTVWY")
CONFIG = {
    "schema_version": 1,
    "model": "esm2_t6_8M_UR50D",
    "representation_layer": 6,
    "embedding_dimension": 320,
    "batch_size": 128,
    "maximum_rows": 1113,
    "minimum_length": 8,
    "maximum_length": 50,
    "seed": 20260909,
    "length_divisor": 50.0,
    "mean_accumulator": "float64_then_float32",
    "maximum_output_bytes": 256 * 1024**2,
    "comparison_rtol": 1e-5,
    "tokens_atol": 1e-5,
    "contacts_atol": 2e-6,
    "spectral_atol": 1e-6,
    "laplacian": asdict(LaplacianLogSpectralDensityConfig()),
}
MODEL_SHA256 = "46f002a9870c9bdecd0ea887acb1f9a38a6b561e8f8bf8a6990b679b9d31b928"
CONTACT_SHA256 = "8f7a4557d57713b97ba0e484303007efb7230d25299c0ac47a0a1b12a87bbb9d"
LOCK_SHA256 = "aaf37baa3adf5070dd3090c43527daa2696741a7e3057c72c78d51a80af9e234"
NAMESPACE_RECEIPT_SHA256 = "c430c8704c853a90fd325765251ede189e0edfdb7226922f5234c1d095c09662"
ARRAY_NAMES = (
    "lengths",
    "token_offsets",
    "contact_offsets",
    "residue_tokens",
    "contacts",
    "esm_mean",
    "esm_length",
    "spectral",
    "esm_length_spectral",
)


def canonical_json(value: object) -> bytes:
    return (
        json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n"
    ).encode()


def digest(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def file_digest(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def read_sequences(payload: bytes) -> list[dict[str, str]]:
    """Reject extra fields (including namespace/fold) at the model boundary."""
    if len(payload) > 1024**2:
        raise ValueError("sequence input exceeds the bounded input size")
    rows = [json.loads(line) for line in payload.splitlines()]
    if not 1 <= len(rows) <= CONFIG["maximum_rows"]:
        raise ValueError("sequence count outside predeclared bound")
    seen = set()
    for row in rows:
        if not isinstance(row, dict) or set(row) != {"sequence_id", "sequence"}:
            raise ValueError("model input must contain only sequence_id and sequence")
        sequence = row["sequence"]
        if (
            not isinstance(sequence, str)
            or not CONFIG["minimum_length"] <= len(sequence) <= CONFIG["maximum_length"]
            or not set(sequence) <= ALPHABET
        ):
            raise ValueError("noncanonical or out-of-bounds peptide")
        sequence_id = digest(sequence.encode("ascii"))
        if row["sequence_id"] != sequence_id or sequence_id in seen:
            raise ValueError("sequence ID mismatch or duplicate")
        seen.add(sequence_id)
    return rows


def project_namespaces(receipt: Path, generator: Path, oracle: Path) -> tuple[bytes, dict]:
    """Project only the two authenticated label-free corpora, never assay files."""
    receipt_bytes = receipt.read_bytes()
    if digest(receipt_bytes) != NAMESPACE_RECEIPT_SHA256:
        raise ValueError("namespace receipt pin mismatch")
    accepted = json.loads(receipt_bytes)
    pins = accepted["twins"][0]["semantic_sha256"]
    rows, provenance = [], []
    bindings = {}
    required = {
        "schema_version",
        "sequence_id",
        "sequence",
        "namespace",
        "generator_fold",
        "homology_component_id",
        "union_component_id",
    }
    for namespace, path in (("generator", generator), ("oracle", oracle)):
        payload = path.read_bytes()
        if digest(payload) != pins[f"{namespace}_corpus.jsonl"]:
            raise ValueError("namespace corpus does not match accepted receipt")
        bindings[namespace] = {"path": str(path.resolve()), "sha256": digest(payload)}
        for line in payload.splitlines():
            row = json.loads(line)
            if set(row) != required or row["namespace"] != namespace:
                raise ValueError("unexpected namespace corpus schema")
            rows.append({key: row[key] for key in ("sequence_id", "sequence")})
            provenance.append({key: row[key] for key in required - {"sequence"}})
    rows.sort(key=lambda row: row["sequence_id"])
    provenance.sort(key=lambda row: row["sequence_id"])
    payload = b"".join(canonical_json(row) for row in rows)
    read_sequences(payload)
    if len(rows) != 1113:
        raise ValueError("accepted namespace projection must have 1113 sequences")
    return payload, {
        "schema_version": 1,
        "namespace_receipt_sha256": digest(receipt_bytes),
        "sources": bindings,
        "sequence_input_sha256": digest(payload),
        "rows": provenance,
    }


def derived_arrays(rows: list[dict], tokens: np.ndarray, contacts: np.ndarray) -> dict:
    """Reconstruct all features from compact, residue-only model outputs."""
    lengths = np.asarray([len(row["sequence"]) for row in rows], dtype="<i8")
    token_offsets = np.concatenate((np.zeros(1, dtype="<i8"), np.cumsum(lengths)))
    contact_offsets = np.concatenate((np.zeros(1, dtype="<i8"), np.cumsum(lengths**2)))
    if tokens.dtype != np.dtype("<f4") or tokens.shape != (int(token_offsets[-1]), 320):
        raise ValueError("residue tokens have wrong dtype/shape/alignment")
    if contacts.dtype != np.dtype("<f4") or contacts.shape != (int(contact_offsets[-1]),):
        raise ValueError("contacts have wrong dtype/shape/alignment")
    if not np.isfinite(tokens).all() or not np.isfinite(contacts).all():
        raise ValueError("nonfinite model output")
    if np.any(contacts < 0) or np.any(contacts > 1):
        raise ValueError("model contact probabilities outside [0,1]")
    means, matrices = [], []
    for index, length in enumerate(lengths):
        means.append(
            tokens[token_offsets[index] : token_offsets[index + 1]].mean(axis=0, dtype=np.float64)
        )
        matrices.append(
            contacts[contact_offsets[index] : contact_offsets[index + 1]].reshape(length, length)
        )
    mean = np.asarray(means, dtype="<f4")
    esm_length = np.column_stack((mean, lengths / CONFIG["length_divisor"])).astype("<f4")
    spectral = laplacian_log_spectral_densities(matrices).astype("<f8")
    combined = np.column_stack((esm_length, spectral)).astype("<f8")
    return dict(
        zip(
            ARRAY_NAMES,
            (
                lengths,
                token_offsets,
                contact_offsets,
                tokens,
                contacts,
                mean,
                esm_length,
                spectral,
                combined,
            ),
            strict=True,
        )
    )


def extract(model, alphabet, torch, rows: list[dict], device: str = "cuda") -> dict:
    """Batched official ESM2 forward; no BOS/EOS/padding residues are retained."""
    if not alphabet.prepend_bos or not alphabet.append_eos:
        raise ValueError("expected ESM2 BOS/EOS alphabet")
    model.eval()
    converter = alphabet.get_batch_converter()
    residue_chunks, contact_chunks = [], []
    with torch.inference_mode():
        for offset in range(0, len(rows), CONFIG["batch_size"]):
            batch = rows[offset : offset + CONFIG["batch_size"]]
            labels, sequences, ids = converter(
                [(row["sequence_id"], row["sequence"]) for row in batch]
            )
            if labels != [row["sequence_id"] for row in batch] or sequences != [
                row["sequence"] for row in batch
            ]:
                raise ValueError("batch converter changed input order/content")
            for index, row in enumerate(batch):
                expected = [
                    alphabet.cls_idx,
                    *[alphabet.get_idx(aa) for aa in row["sequence"]],
                    alphabet.eos_idx,
                ]
                observed = ids[index].tolist()
                if observed[: len(expected)] != expected or any(
                    value != alphabet.padding_idx for value in observed[len(expected) :]
                ):
                    raise ValueError("BOS/residue/EOS/padding token alignment mismatch")
            result = model(ids.to(device), repr_layers=[6], return_contacts=True)
            hidden = result["representations"][6]
            contact = result["contacts"]
            longest = max(len(row["sequence"]) for row in batch)
            if tuple(hidden.shape) != (len(batch), longest + 2, 320) or tuple(contact.shape) != (
                len(batch),
                longest,
                longest,
            ):
                raise ValueError("ESM2 token/contact tensor dimensions mismatch")
            for index, row in enumerate(batch):
                length = len(row["sequence"])
                residue_chunks.append(hidden[index, 1 : length + 1].float().cpu().numpy().copy())
                contact_chunks.append(
                    contact[index, :length, :length].float().cpu().numpy().copy().reshape(-1)
                )
    return derived_arrays(rows, np.concatenate(residue_chunks), np.concatenate(contact_chunks))


def publish(output: Path, payload: bytes, arrays: dict, identity: dict) -> dict:
    """Exclusive new directory; COMPLETE is last. Failed runs remain tombstones."""
    rows = read_sequences(payload)
    reconstructed = derived_arrays(rows, arrays["residue_tokens"], arrays["contacts"])
    if set(arrays) != set(ARRAY_NAMES) or any(
        not np.array_equal(arrays[key], reconstructed[key]) for key in ARRAY_NAMES
    ):
        raise ValueError("derived feature arrays do not reconstruct")
    output.mkdir(mode=0o700)  # no exist_ok, rename, replacement, or cleanup
    inventory = {}
    total = 0
    for name, array in arrays.items():
        stream = io.BytesIO()
        np.save(stream, array, allow_pickle=False)
        data = stream.getvalue()
        total += len(data)
        if total > CONFIG["maximum_output_bytes"]:
            raise ValueError("feature output size bound exceeded")
        path = output / f"{name}.npy"
        with path.open("xb") as target:
            target.write(data)
        path.chmod(0o444)
        inventory[path.name] = {
            "sha256": digest(data),
            "bytes": len(data),
            "shape": list(array.shape),
            "dtype": array.dtype.str,
        }
    with (output / "sequences.jsonl").open("xb") as target:
        target.write(payload)
    (output / "sequences.jsonl").chmod(0o444)
    manifest = {
        "schema_version": 1,
        "artifact": "real_peptide_esm_contact_features_v1",
        "qualification": "label_free_data_preparation_only_not_scientific_evidence",
        "production_input_eligible": False,
        "scientific_evidence_accepted": False,
        "rows": len(rows),
        "config": CONFIG,
        "config_sha256": digest(canonical_json(CONFIG)),
        "sequence_input_sha256": digest(payload),
        "sequence_order_sha256": digest(canonical_json([row["sequence_id"] for row in rows])),
        "arrays": inventory,
        "identity": identity,
    }
    manifest_bytes = canonical_json(manifest)
    for name, data in (
        ("manifest.json", manifest_bytes),
        ("COMPLETE", canonical_json({"manifest_sha256": digest(manifest_bytes)})),
    ):
        with (output / name).open("xb") as target:
            target.write(data)
        (output / name).chmod(0o444)
    output.chmod(0o555)
    return manifest


def load_features(output: Path, expected_manifest_sha256: str) -> tuple[list[dict], dict, dict]:
    """Validate externally pinned payloads and independently reconstruct derived arrays."""
    manifest_bytes = (output / "manifest.json").read_bytes()
    if digest(manifest_bytes) != expected_manifest_sha256:
        raise ValueError("feature manifest pin mismatch")
    marker = json.loads((output / "COMPLETE").read_bytes())
    if marker != {"manifest_sha256": expected_manifest_sha256}:
        raise ValueError("incomplete feature artifact")
    manifest = json.loads(manifest_bytes)
    if manifest["config"] != CONFIG or manifest["config_sha256"] != digest(canonical_json(CONFIG)):
        raise ValueError("feature config mismatch")
    if (
        manifest["production_input_eligible"] is not False
        or manifest["scientific_evidence_accepted"] is not False
    ):
        raise ValueError("feature artifact cannot carry promotion authority")
    expected_names = {f"{name}.npy" for name in ARRAY_NAMES}
    if set(manifest["arrays"]) != expected_names or {
        path.name for path in output.iterdir()
    } != expected_names | {"sequences.jsonl", "manifest.json", "COMPLETE"}:
        raise ValueError("unexpected feature payload inventory")
    payload = (output / "sequences.jsonl").read_bytes()
    rows = read_sequences(payload)
    if (
        digest(payload) != manifest["sequence_input_sha256"]
        or manifest["rows"] != len(rows)
        or manifest["sequence_order_sha256"]
        != digest(canonical_json([row["sequence_id"] for row in rows]))
    ):
        raise ValueError("feature input sequence/order mismatch")
    arrays = {}
    for name in ARRAY_NAMES:
        data = (output / f"{name}.npy").read_bytes()
        binding = manifest["arrays"][f"{name}.npy"]
        if len(data) != binding["bytes"] or digest(data) != binding["sha256"]:
            raise ValueError("feature array bytes mismatch")
        array = np.load(io.BytesIO(data), allow_pickle=False)
        if list(array.shape) != binding["shape"] or array.dtype.str != binding["dtype"]:
            raise ValueError("feature array metadata mismatch")
        arrays[name] = array
    expected = derived_arrays(rows, arrays["residue_tokens"], arrays["contacts"])
    for name in ARRAY_NAMES:
        matches = np.array_equal(arrays[name], expected[name])
        if name in {"spectral", "esm_length_spectral"}:
            matches = np.allclose(arrays[name], expected[name], atol=1e-12, rtol=1e-12)
        if not matches:
            raise ValueError(f"feature array does not reconstruct: {name}")
    return rows, arrays, manifest
