# Training data, external databases and filters

Every dataset that entered a trained model in this repository is openly licensed
and pinned by SHA-256. The normalized training projections are shipped by value
under `checkpoints/competition/training/`, so this disclosure can be checked
directly rather than taken on trust.

## Sources that entered training

| Source | License | Pinned identity | Used for |
| --- | --- | --- | --- |
| AMP-Diffusion starter-kit experimental MIC table (Torres et al., *Cell Biomaterials* 2025) | CC-BY-4.0 | `3da5adb7814f00b5f331eedd3dc909c55ee66cc1561e1bf33d5c1753d30eb82d`, 46,598 bytes, source commit `1a862af9078e6b55c87d1fa576f3da81851ba94b` | Reward ensemble |
| DRAMP 2.0 general AMP deposit | CC0-1.0 | `b726a633634257e382545db5da62df75b5c284c03e49a26e03d8d5fc96fbad41`, 1,356,701 bytes, `doi:10.6084/m9.figshare.8006309.v1` | Reward ensemble, generator corpus |
| ESM-2 `esm2_t6_8M_UR50D` pretrained weights | MIT (Meta AI) | checkpoint `46f002a9870c9bdecd0ea887acb1f9a38a6b561e8f8bf8a6990b679b9d31b928`, bundled under `checkpoints/competition/embedding/` | Frozen sequence encoder, never fine-tuned |

The two approved sources are normalized and combined into a single
homology-and-study union panel. The deployed reward ensemble is fit on
**2,492 assay contexts over 952 distinct peptides** across seven bacterial
targets; its training inventory digest is
`d3eecbf3014fd78cf7021818466893d292315b6e90e77cea85d4e1fbd5bec520`, recorded in
`checkpoints/competition/oracle/report.json` and shipped as
`checkpoints/competition/training/oracle_examples.jsonl`.

The ten diffusion generators were trained on a corpus of **698 unique peptides**.
Each generator's exact weighted training projection is shipped as
`checkpoints/competition/training/generator-NN.jsonl`.

## Sources deliberately NOT used

`configs/data/starter_snapshots.toml` pins several further starter-kit and
database artifacts as `review_required`. They were downloaded and audited for
schema, overlap and licensing, then **rejected for training** because their
upstream redistribution terms are ambiguous. This includes DBAASP-derived
tables, HydrAMP splits, GRAMPA, dbAMP, APD3 and AMP Scanner exports. Changing
a status flag alone will not admit them: preparation fails without a written
decision and a source-specific adapter.

`data/antibacterial.fasta` is the organizer's reference file, reproduced
byte-for-byte from `szczurek-lab/amp-challenge-2027` at commit
`5c8a5d8e2551c8cf572d3d3bfcfe7633b109d91e` under BSD-3-Clause. It is used
**only** as a novelty and compliance reference, never as training data.

## Filters applied

Normalization is conservative and quarantines anything it cannot represent
faithfully:

- only the 20 canonical amino acids; length 8-50 for challenge use;
- modified, cyclized, disulfide-bonded, lipidated and noncanonical peptides are
  quarantined rather than coerced;
- MIC right-censoring is preserved (`>64` is never rewritten as `64`);
- mass-to-molar unit conversion uses the free-termini molecular weight;
- rows whose chemistry or measurement context cannot be interpreted are logged
  in a rejection ledger instead of being silently dropped.

Held-out evaluation uses **homology- and study-grouped folds**, never random
assay-row splits: no sequence or study/homology component may cross a fold.

## What the model does not predict

The reward ensemble predicts a binary per-target activity probability only. It
explicitly does **not** model continuous MIC, hemolysis or MDR-strain-specific
activity; those endpoints are listed as unsupported in the oracle report. All
activity numbers in this repository are internal model predictions, not measured
potency, selectivity or safety.
