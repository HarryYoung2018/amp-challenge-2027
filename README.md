# AMP Challenge 2027 — frozen-mixture diffusion search under a distance budget

Submission entry for the [AMP Challenge 2027](https://github.com/szczurek-lab/amp-challenge-2027).
See [WRITEUP.md](WRITEUP.md) for the full submission writeup,
[ABSTRACT.md](ABSTRACT.md) for the method summary, and
[DATA_DISCLOSURE.md](DATA_DISCLOSURE.md) for the full training-data disclosure.

All activity values below are **internal model predictions**, not measured
potency, selectivity or safety. No wet-lab validation is claimed.

## Quick start

```bash
uv sync
uv run generate
```

That command needs no arguments and every argument has a default. It writes
`generate/library.fasta` (50,000 peptides), `generate/top.fasta` (ranked top
100), `generate/top_metadata.json` and `generate/manifest.json`, using the
bundled no-update frozen mixture and fixed seed **42**. Repeated runs on the
same device reproduce byte-identical output. A CUDA GPU is used by default;
`--device cpu` works but cross-device byte identity is not asserted.

To verify against the organizer's own checker, run `scripts/verify_submission.py`
from the starter kit against this repository. It is not vendored here because it
depends on the GPL-licensed `Levenshtein` package; this repository uses
MIT-licensed RapidFuzz for the same normalized-indel ratio.

The pre-generated outputs are also committed under `submission/`, so the
libraries can be inspected without running the model.

## What is in this repository

| Path | Contents |
| --- | --- |
| `submission/no-update-control/` | **Primary submission.** 50,000-peptide library and ranked top 100 from the frozen no-update mixture. |
| `submission/updating-total-variation/` | Alternative submission from the total-variation updating run (63 accepted policy updates). |
| `checkpoints/competition/evolutionary/` | The ten frozen generator policies used by the default entry point. |
| `checkpoints/competition/oracle/` | Target-specific reward ensemble and its grouped out-of-fold report. |
| `checkpoints/competition/embedding/` | Pretrained ESM-2 `esm2_t6_8M_UR50D` encoder (frozen). |
| `checkpoints/competition/training/` | Exact training projections for every trained model. |
| `src/amp_challenge/` | Generation, search, reward and selection code. |
| `MANIFEST.json`, `CHECKSUMS.sha256` | Identities for every shipped file. |

The 2.5 GB weight bundle for the updating run's 640-component mixture is
attached to the tagged GitHub release rather than committed, together with that
run's 63 adaptive update records.

## Method

Ten categorical discrete-diffusion generators (six Transformer layers, width
128, four heads, feed-forward width 384, 64 diffusion levels, 1,015,572
parameters each) are trained on a 698-peptide corpus. Search is evolutionary
remasking: a parent is partially remasked at a scheduled noise level and
denoised into children, which are scored by a target-specific reward ensemble
over frozen mean-pooled ESM-2 embeddings (random forest, LightGBM and
cosine-neighbour predictor, equally weighted, one model set per target).

Policy updates are constrained twice over:

- **Locally**, an updated policy's conditional distribution at forward-noised
  replay anchors must stay within total variation 0.05 of the previous policy;
  candidates are interpolated back toward the old policy until they pass.
- **Globally**, an updated component enters the sampling mixture with at most
  1/20 of the mass, and exactly one component is drawn per complete generation
  trajectory. This makes `TV(P_new, P_old) <= 1/20` hold exactly for the
  whole-peptide distribution, independent of any sampled probe.

## Results

Both libraries pass the organizer's checks against all 39,448 reference records
with **zero issues**: 50,000 unique sequences, canonical alphabet, length 8-50,
and no top-100 sequence above 80% Levenshtein identity to any reference.

| | No-update control (default) | Updating total variation |
| --- | ---: | ---: |
| Library size | 50,000 | 50,000 |
| Accepted policy updates | 0 | 63 |
| Top-100 mean predicted activity | 0.878970 | 0.879875 |
| Top-100 lower decile | 0.867099 | 0.866718 |
| Top-100 mean pairwise Indel similarity | 0.514635 | 0.513374 |
| Audit issues | 0 | 0 |

**Why the no-update mixture is the default.** The updating variant is ahead by
0.0009 on top-100 mean and behind on the lower decile — well inside run-to-run
noise. In our own matched three-seed search comparison the updating variant did
not beat the no-update control on the primary endpoint. We therefore ship the
simpler, fully frozen policy as the primary deliverable and publish the updating
variant beside it rather than selecting the nominally higher number after the
fact.

## Honest limitations

- Reported activity is a **model prediction**, on a binary per-target
  probability scale. Continuous MIC, hemolysis and MDR-strain-specific activity
  are explicitly unsupported endpoints.
- Grouped out-of-fold, the reward ensemble does **not** clearly beat a plain
  descriptor logistic-regression baseline (ensemble Brier 0.2007 / log loss
  0.5857 / ROC-AUC 0.6751 / AP 0.8261; descriptor baseline 0.1966 / 0.5807 /
  0.6843 / 0.8185). It is ahead on average precision and behind on the other
  three. Treat it as a usable ranking surrogate, not a validated oracle.
- The reward averages seven bacterial targets, and the available measured data
  do not cover every one of them equally.
- The whole-output 5% bound applies to the sampling policy mixture. It is **not**
  claimed for the post-hoc selected 50,000-member library, which is filtered and
  ranked after generation.

## License

MIT (see [LICENSE](LICENSE)). Third-party components and their terms are listed
in [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md).
