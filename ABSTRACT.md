# Abstract

We design antimicrobial peptides with a **frozen-mixture discrete diffusion search**
under an explicit, auditable distribution-distance budget.

Ten categorical discrete-diffusion generators over the twenty canonical amino
acids are trained on public antibacterial peptide data. Search proceeds by
evolutionary remasking: a parent peptide is partially remasked at a scheduled
noise level and denoised into children. Candidates are scored by a
**target-specific reward ensemble** built on fixed ESM-2 (`esm2_t6_8M_UR50D`)
mean-pooled residue embeddings, combining a random forest, a gradient-boosted
tree model and a cosine-similarity neighbour estimator per bacterial target.

The distinguishing element is the **update contract**. Any change to a sampling
policy must pass two independent limits. First, a *local* limit: the updated
policy's conditional distribution at forward-noised replay anchors must stay
within a total-variation ball of radius 0.05 of the previous policy, with
backtracking interpolation until it does. Second, a *global* limit: an updated
component may enter the sampling mixture with at most 1/20 of the total mass, so
that for whole-peptide distributions `P` and `Q`,
`TV((1-a)P + aQ, P) <= a <= 0.05` holds exactly and independently of any sampled
probe. One mixture component is chosen per complete generation trajectory, which
is what makes that bound valid.

We release two audited 50,000-peptide libraries from a matched pair of runs that
differ only in whether policy updates were admitted. The **no-update frozen
mixture is the default entry point**: across our own matched three-seed
comparison the updating variant did not beat it, so we ship the simpler policy
as the primary deliverable and the updating variant as a fully documented
alternative. Both libraries pass the organizer's sequence, uniqueness and
reference-identity checks with zero issues.

All reported activity values are **internal model predictions**, not measured
potency, selectivity or safety. No wet-lab validation is claimed.
