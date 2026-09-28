# Third-party notices

## AMP Challenge organizer reference

`data/antibacterial.fasta` is reproduced byte-for-byte from
[`szczurek-lab/amp-challenge-2027`](https://github.com/szczurek-lab/amp-challenge-2027)
at commit `5c8a5d8e2551c8cf572d3d3bfcfe7633b109d91e`, under the BSD 3-Clause
terms reproduced in full below and kept verbatim as `LICENSE.starter-kit`.

The organizer's `scripts/verify_submission.py` is deliberately **not** vendored
here: it depends on the GPL-licensed `Levenshtein` package, which this
MIT-licensed repository avoids. Run it from the starter kit against this
repository instead.

BSD 3-Clause License

Copyright (c) 2026, Ewa Szczurek lab

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:

1. Redistributions of source code must retain the above copyright notice,
   this list of conditions and the following disclaimer.

2. Redistributions in binary form must reproduce the above copyright notice,
   this list of conditions and the following disclaimer in the documentation
   and/or other materials provided with the distribution.

3. Neither the name of the copyright holder nor the names of its contributors
   may be used to endorse or promote products derived from this software without
   specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

## Research integrations and dependencies

No Equiformer v3, AMP-Diffusion, or OmegAMP source code, datasets, checkpoints,
or weights are vendored at this stage. Their papers and repositories are cited
as research references. `integrations/omegamp/reviewed_bridge.py` is a
repository-authored compatibility and verification layer for the public
[OmegAMP](https://github.com/szczurek-lab/OmegAMP) API pinned in its integration
manifest; the upstream project declares the MIT license.

Runtime packages are resolved from their publishers by `uv.lock`, not copied
into this source tree. In particular, the organizer's normalized-indel ratio is
implemented with MIT-licensed RapidFuzz rather than the GPL-licensed
python-Levenshtein package. The quality-constrained final-portfolio baseline
uses BSD-3-Clause SciPy's `milp` interface and its bundled MIT-licensed HiGHS
solver; their exact resolved versions are retained in `uv.lock` and the solver
versions used by a constrained run are written to its summary.

## Note on ESM-2

`checkpoints/competition/embedding/` contains the pretrained ESM-2
