# Product

<!-- impeccable:product-schema 1 -->

## Platform

web

## Stack

Single static HTML file (inline CSS/JS, no build step). Deploy target: GitHub Pages, served
directly from the `pardhuvarmax/lbsm-research` repository.

## Users

- **Primary: ML/statistics researchers and reviewers** evaluating the LBSM hypothesis,
  methodology, and experimental results. They expect rigor, real figures/data, reproducibility
  cues, and accurate framing of what is proven versus still open.
- **Secondary: broader technical audience** (ML/software engineers, curious technical readers)
  following the narrative and visuals without needing full statistical fluency. The site should
  stay accessible to this audience without diluting rigor for the primary audience.

## Product Purpose

A project site for LBSM ("Latent Behavioral Structure in Low-Dimensional Statistical
Manifolds") — a statistical ML research program testing whether latent behavioral structure and
its temporal evolution in adaptive, autonomous, sequential systems occupies a structured
low-dimensional manifold rather than arbitrary high-dimensional space. The site's job is to
present the hypothesis, methodology, and experimental artifacts (notebooks, figures, reports,
datasets) so a visitor can understand and evaluate the research. **Research is ongoing** — the
site must present current findings and artifacts as work-in-progress, not as a finished or final
result.

## Positioning

Reinforcement-guided synthetic agents (`AdaptiveAgent`, a hidden 4-state Markov chain emitting
6-D observable telemetry) are the project's controlled Phase 1 instantiation of the hypothesis —
chosen because their regimes are known by construction, giving ground truth to validate the
manifold-learning/HMM/drift/RL methodology against before applying it to real sequential systems
without ground truth (e.g. planetary telemetry). The site should make this Phase 1 framing
explicit rather than presenting the synthetic-agent results as the full scope of the hypothesis.

## Operating Context

- Pipeline: simulation → telemetry → manifold learning → three parallel analysis branches (HMM
  regime recovery, drift/regime-shift detection, RL-driven behavioral adaptation) → evaluation.
- 7 populated Jupyter notebooks (`01_telemetry_generation` through `07_final_experiment_analysis`),
  each with processed datasets under `data/processed/nbNN/` and a PDF report under
  `outputs/reports/nbNN/`, plus a unified report (`outputs/reports/lbsm_unified_report.pdf`).
  Notebooks 06–07 per README are populated (CLAUDE.md's "not started" note for 06/07 is stale
  relative to README — verify against the live repo before citing notebook status precisely).
- Generated figures live under `outputs/figures/nbNN/`, one directory per notebook, catalogued in
  `outputs/figures/readme.md` and the unified report's Appendix A.
- A TMLR paper draft exists (`paper/latex-tmlr/main.tex`, with a `main-accepted.pdf` build
  artifact) and a Zenodo-archived research artifact (DOI `10.5281/zenodo.22108702`, see
  `CITATION.cff`) — see Evidence on Hand for how the site should frame these given ongoing status.

## Capabilities and Constraints

- Single self-contained HTML file: inline CSS/JS, no external build tooling, no backend, no
  server-side rendering.
- Must link out to real repository artifacts (GitHub-hosted notebooks, datasets, reports) rather
  than reproducing or fabricating their content on the page.
- Deploys via GitHub Pages from this repository — links to in-repo files should resolve as GitHub
  blob/raw URLs or relative paths compatible with Pages serving.
- Open/undecided: exact repository visibility and final Pages URL path; not needed to build the
  page itself.

## Brand Commitments

- Project name/wordmark: "LBSM Research," owl illustration mark (`arts/lbsm-01.png`, 2392×1080,
  black background) — currently used as the GitHub README header and the unified report's cover
  page. `arts/lbsm-02.png` is a softer-contrast alternate rendering of the same mark, not
  currently used anywhere.
- License: MIT (`LICENSE`).
- These are existing brand assets to treat as evidence, not a confirmed visual direction for the
  new site — the visual-world decision (reuse, adapt, or depart from this mark) is a new-work
  decision, not settled here.

## Evidence on Hand

- `README.md` — existing narrative copy (hypothesis, research direction, status) and a working
  badge-linked index of all 7 notebooks, their processed datasets, and their PDF reports.
- `outputs/figures/nbNN/*.png` — real generated figures per notebook (7–12 per notebook),
  authoritative visual evidence the site can draw on.
- `outputs/reports/nbNN/lbsm_notebookNN.pdf` and `outputs/reports/lbsm_unified_report.pdf` — real
  per-notebook and unified PDF reports.
- `paper/latex-tmlr/main-accepted.pdf` and `CITATION.cff` (Zenodo DOI) exist in-repo, but per the
  user, research is still ongoing — the site should not present the paper or DOI as evidence of a
  finished/published result. Treat their inclusion and framing (e.g. "draft," "preprint," "in
  progress") as an open decision for the build step, not a settled claim.
- No testimonials, case studies, press, benchmarks against competing methods, or pricing exist —
  do not fabricate any.

## Product Principles

1. Present an ongoing hypothesis-testing research program, not a finished result — every claim of
   status (what's proven, what's Phase 1, what's still open) must match the current repo state.
2. Ground every visual and narrative claim in a real artifact already in this repo (notebook,
   figure, report, dataset) — never invent results, metrics, or endorsements.
3. Serve the researcher audience's need for rigor and reproducibility cues first, while keeping
   the narrative legible to a technical non-specialist skimming the page.
4. Stay a single, dependency-free, static HTML file — no build step, no backend, trivially
   deployable to GitHub Pages.
5. Make the Phase 1 synthetic-agent framing explicit rather than letting it read as the full
   scope of the hypothesis.

## Accessibility & Inclusion

No project-specific accessibility requirement has been established beyond standard web
accessibility practice (semantic structure, color contrast, keyboard navigability) for a static
informational page.
