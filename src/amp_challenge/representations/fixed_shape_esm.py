"""Explicit fixed tensor layout; no changes to the historical ESM extractor."""

from __future__ import annotations

import numpy as np

from amp_challenge.representations.peptide_esm import (
    canonical_json,
    derived_arrays,
    digest,
    read_sequences,
)

FIXED_LAYOUT = {
    "artifact": "esm_fixed_shape_layout_v2",
    "batch_rows": 128,
    "token_positions": 52,
    "filler_sequence": "ACDEFGHI",
    "filler_role": "computational_padding_only_never_candidate_or_training_rows",
    "output": "requested_residues_and_contacts_only",
}
FIXED_LAYOUT_SHA256 = digest(canonical_json(FIXED_LAYOUT))


def extract_fixed_shape(model, alphabet, torch, rows: list[dict], device: str = "cuda") -> dict:
    """Always forward128x52 tokens; filler outputs cannot enter saved arrays.

    Fixed shape removes the observed dynamic batch-size/length change. Numerical
    invariance still requires empirical qualification; it is not assumed from
    this implementation alone. The checkpoint and float32 runtime are unchanged.
    """
    rows = read_sequences(b"".join(canonical_json(row) for row in rows))
    if not alphabet.prepend_bos or not alphabet.append_eos:
        raise ValueError("expected ESM2 BOS/EOS alphabet")
    model.eval()
    converter = alphabet.get_batch_converter()
    residue_chunks, contact_chunks = [], []
    with torch.inference_mode():
        for offset in range(0, len(rows), 128):
            batch = rows[offset : offset + 128]
            inputs = [(row["sequence_id"], row["sequence"]) for row in batch]
            inputs.extend(("computational-filler", "ACDEFGHI") for _ in range(128 - len(batch)))
            labels, sequences, converted = converter(inputs)
            if labels != [item[0] for item in inputs] or sequences != [item[1] for item in inputs]:
                raise ValueError("fixed-shape converter changed input order/content")
            if converted.ndim != 2 or converted.shape[0] != 128 or converted.shape[1] > 52:
                raise ValueError("fixed-shape converter dimensions differ")
            ids = torch.full((128, 52), alphabet.padding_idx, dtype=torch.long)
            ids[:, : converted.shape[1]] = converted
            for index, (_, sequence) in enumerate(inputs):
                expected = [
                    alphabet.cls_idx,
                    *[alphabet.get_idx(aa) for aa in sequence],
                    alphabet.eos_idx,
                ]
                observed = ids[index].tolist()
                if observed[: len(expected)] != expected or any(
                    token != alphabet.padding_idx for token in observed[len(expected) :]
                ):
                    raise ValueError("fixed-shape BOS/residue/EOS/padding alignment differs")
            result = model(ids.to(device), repr_layers=[6], return_contacts=True)
            hidden, contact = result["representations"][6], result["contacts"]
            if tuple(hidden.shape) != (128, 52, 320) or tuple(contact.shape) != (128, 50, 50):
                raise ValueError("fixed-shape ESM tensor dimensions differ")
            for index, row in enumerate(batch):
                length = len(row["sequence"])
                residue_chunks.append(hidden[index, 1 : length + 1].float().cpu().numpy().copy())
                contact_chunks.append(
                    contact[index, :length, :length].float().cpu().numpy().copy().reshape(-1)
                )
    return derived_arrays(rows, np.concatenate(residue_chunks), np.concatenate(contact_chunks))
