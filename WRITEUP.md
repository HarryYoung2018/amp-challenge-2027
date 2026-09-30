# AMP Challenge 2027 — submission writeup

**Repository:** https://github.com/HarryYoung2018/amp-challenge-2027 (public, MIT)
**Weights release:** `v1.0.0`
**Attachments:** `library.fasta` (50,000), `top.fasta` (ranked 100)

---

## Abstract

We design antimicrobial peptides with a **frozen-mixture discrete diffusion
search under an explicit distribution-distance budget**.

Ten categorical discrete-diffusion generators over the twenty canonical amino
acids (six Transformer layers, width 128, four heads, feed-forward width 384,
64 diffusion levels, 1,015,572 parameters each) are trained on a 698-peptide
corpus. Search proceeds by evolutionary remasking: a parent peptide is
partially remasked at a scheduled noise level and denoised into children.
Candidates are scored by a **target-specific reward ensemble** over frozen
ESM-2 (`esm2_t6_8M_UR50D`) mean-pooled residue embeddings — a random forest, a
LightGBM classifier and a cosine-similarity neighbour estimator, equally
weighted, fitted separately for each of seven bacterial targets.

The distinguishing element is the **update contract**. Any change to a sampling
policy must pass two independent limits. *Locally*, the updated policy's
conditional distribution at forward-noised replay anchors must stay within
total variation 0.05 of the previous policy, with backtracking interpolation
until it does. *Globally*, an updated component enters the sampling mixture with
at most 1/20 of the total mass, and exactly one component is drawn per complete
generation trajectory — so for whole-peptide distributions `P` and `Q`,
`TV((1-a)P + aQ, P) = a <= 0.05` holds exactly, independent of any sampled probe.
Re-drawing a component at each diffusion step would void that bound.

We submit the **no-update frozen mixture** as the primary entry. In our own
matched three-seed search comparison the updating variant did not beat the
frozen control (0.878091 vs 0.879012 eligible paid top-100 mean), and a tuned
genetic algorithm beat both (0.883499). We therefore ship the simpler policy
rather than selecting the nominally higher single-seed number after the fact.
The total-variation updating run is published beside it for completeness.

All reported activity values are **internal model predictions**, not measured
potency, selectivity or safety. No wet-lab validation is claimed.

---

## Top-100 selection and ranking

The ranked list is produced by a greedy, diversity-penalised, multi-objective
portfolio selection over the scored 50,000-member library
(`choose_portfolio` in `src/amp_challenge/workflows/competition_generate.py`).

**Three utilities** are computed per candidate from the seven per-target
predicted activity probabilities:

| Utility | Definition |
| --- | --- |
| `broad` | mean over all seven targets |
| `gram_positive` | mean over the Gram-positive panel |
| `gram_negative` | mean over the Gram-negative panel |

**Rank cycle.** Ranks are assigned by cycling objectives `(broad,
gram_positive, broad, gram_negative)`, so the final list is 50% broad-spectrum,
25% Gram-positive-led and 25% Gram-negative-led throughout the ranking, not
only at the top.

**Diversity penalty.** At each rank the score is
`utility - 0.10 x max_cosine_similarity_to_already_selected`, computed on
L2-normalized ESM-2 embeddings. This spreads the portfolio across the embedding
space instead of returning one motif family.

**Hard eligibility.** A candidate is admitted only if its maximum normalized
indel (Levenshtein) similarity against every record in
`data/antibacterial.fasta` is at most 0.8. Candidates above the threshold are
permanently removed from the pool. Ties break on ascending candidate index for
determinism.

**Per-sequence provenance** is shipped in `top_metadata.json`: rank, sequence,
which objective selected it, all seven per-target predicted probabilities, the
diversity-penalised utility at selection time, and its maximum reference
similarity ratio. Note that `internal_utility` is the *penalised* selection
score, not a raw activity estimate.

**Achieved properties** (independently re-verified from a fresh clone, not
reusing the generator's own validator):

| Property | No-update control | Updating total variation |
| --- | ---: | ---: |
| Library records / unique | 50,000 / 50,000 | 50,000 / 50,000 |
| Top-100 records / unique | 100 / 100 | 100 / 100 |
| Exact matches to reference set | 0 | 0 |
| Max top-100 reference similarity | 0.6512 | 0.6667 |
| Top-100 mean predicted activity | 0.878970 | 0.879875 |
| Top-100 lower decile | 0.867099 | 0.866718 |
| Top-100 mean pairwise indel similarity | 0.514635 | 0.513374 |

---

## Training data, external databases and filters

Every dataset that entered a trained model is openly licensed and pinned by
SHA-256. Normalized training projections are shipped **by value** in the
repository under `checkpoints/competition/training/`, so the disclosure is
checkable rather than asserted.

| Source | License | Used for |
| --- | --- | --- |
| AMP-Diffusion starter-kit experimental MIC table (Torres et al., *Cell Biomaterials* 2025) | CC-BY-4.0 | Reward ensemble |
| DRAMP 2.0 general AMP deposit (`doi:10.6084/m9.figshare.8006309.v1`) | CC0-1.0 | Reward ensemble, generator corpus |
| ESM-2 `esm2_t6_8M_UR50D` pretrained weights | MIT (Meta AI) | Frozen encoder, never fine-tuned |

The reward ensemble is fit on **2,492 assay contexts over 952 distinct
peptides** across seven targets. The ten generators are trained on **698 unique
peptides**.

**Deliberately excluded.** DBAASP-derived tables, HydrAMP splits, GRAMPA,
dbAMP, APD3 and AMP Scanner exports were downloaded, audited for schema,
overlap and licensing, and then **rejected for training** because their upstream
redistribution terms are ambiguous. They are pinned as `review_required` in
`configs/data/starter_snapshots.toml`, and preparation fails if a status flag is
changed without a written decision and a source-specific adapter.

`data/antibacterial.fasta` is the organizer's reference file, reproduced
byte-for-byte under BSD-3-Clause, used **only** for novelty and compliance
checks — never as training data.

**Filters applied.** Only the 20 canonical amino acids; length 8-50; modified,
cyclized, disulfide-bonded, lipidated and noncanonical peptides quarantined
rather than coerced; MIC right-censoring preserved (`>64` is never rewritten as
`64`); mass-to-molar conversion via free-termini molecular weight; uninterpretable
rows logged in a rejection ledger rather than silently dropped. Held-out
evaluation uses **homology- and study-grouped folds**, never random assay-row
splits.

**Not modelled.** The reward predicts a binary per-target activity probability
only. Continuous MIC, hemolysis and MDR-strain-specific activity are explicit
unsupported endpoints.

---

## Reproduction

```bash
git clone https://github.com/HarryYoung2018/amp-challenge-2027.git
cd amp-challenge-2027
uv sync
uv run generate
```

No arguments required; every argument has a default; the default seed is **42**.
This writes `generate/library.fasta`, `generate/top.fasta`,
`generate/top_metadata.json` and `generate/manifest.json`. The packaged
no-argument run reproduces the shipped `submission/no-update-control/` files
byte for byte, verified by SHA-256 in `MANIFEST.json`.

A CUDA GPU is used by default. `--device cpu` is supported, but cross-device
byte identity is not asserted.

The 2.5 GB weight bundle for the updating run's 641-component mixture and its
63 adaptive update records are attached to release `v1.0.0` rather than tracked
in Git; the primary entry needs none of it.

## Honest limitations

- Reported activity is a **model prediction** on a binary per-target scale.
- Grouped out-of-fold, the reward ensemble does **not** clearly beat a plain
  descriptor logistic regression (ensemble Brier 0.2007 / log loss 0.5857 /
  ROC-AUC 0.6751 / AP 0.8261; descriptor baseline 0.1966 / 0.5807 / 0.6843 /
  0.8185). It is ahead on average precision and behind on the other three.
  Treat it as a usable ranking surrogate, not a validated oracle.
- The measured data do not cover all seven reward targets equally, and contain
  no *Acinetobacter baumannii* context.
- The whole-output 5% bound applies to the sampling **policy mixture**, not to
  the post-hoc filtered and ranked 50,000-member library.
- The submitted library's peptide lengths span only 27-29 residues. This is
  within the 8-50 requirement but is a narrow band, reflecting the length
  distribution the search converged to.
