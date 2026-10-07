# LBSM — Elevation Plan (C+ → A)

Companion to `LBSM_numerical_errors.md`. That document is a correctness punch list; it caps out around **B−/B+**. This one specifies what closes the remaining gap.

---

## 0. Reachable ceiling

| Grade | What it requires | Reachable? |
|---|---|---|
| A+ | Field-shifting: a result that changes how people model latent state in sequential systems | No. Not this project's shape. |
| A | Novel supported claim + real-world evidence + measurable theoretical grounding + baselines | **Yes**, with Phases 1–5. |
| A− | The above minus one component (typically the theory or the baselines) | **Yes**, with Phases 1–4. |
| B+ | Correct, honest, well-executed methods validation | Phases 1–2 alone. |

Target: **A−, with A available if Phase 5 lands.** Plan below is sequenced so each phase is independently valuable and each gate is a real stop/go.

---

## 1. Gap analysis against the current rubric

| Dimension | Now | Blocker | Closed by |
|---|---|---|---|
| Numerical correctness | 3.0 | 48 registered errors | Phase 1 |
| Experimental design | 5.0 | No criterion can fail; generator unvalidated | Phase 1, 2 |
| Theoretical framing | 4.5 | "Manifold" is a metaphor; no intrinsic-dimension estimate; hypothesis's high-dimensionality clause untested | **Phase 3** |
| Reproducibility | 6.5 | NB01–04 unpinned; NB05 artifacts unregenerable | Phase 1 |
| Novelty / contribution | 7.0 | No baselines; N=1 finding, not a characterized regime | **Phase 4** |
| Real-world evidence | — | NB08 empty | **Phase 5** |
| Self-criticism | 8.5 | Already strong — preserve it | — |
| Engineering | 8.5 | Already strong | — |

The two starred rows are what the error register does not address and what separates B+ from A.

---

## Phase 1 — Correctness and provenance (≈1–2 weeks)

Everything in `LBSM_numerical_errors.md`, plus:

**1.1 Resolve A1 (the AR(1) carryover).** This is a fork, not a fix. Pick one and commit:

- *(a) Keep it, document it.* Declare regime transitions as continuous relaxations rather than instantaneous jumps. Defensible and arguably closer to a genuine manifold than a clean mixture. Requires: an explicit generative equation in the paper matching the code, a stated relaxation time per regime, and rewriting NB01 §1.7.1's causal claim.
- *(b) Remove it.* Reset `_prev_telemetry` on transition or reseed from the new regime's mean. Regenerate everything. Cleaner mixture semantics, but weakens the manifold narrative further.

Recommendation: **(a)**, with (b) run as an ablation. The relaxation transients are the most manifold-like structure in the whole dataset, and they are the mechanism behind NB07's finding.

**1.2 Provenance discipline.** Every table gets a header stating whether it is computed from configuration constants or emitted data. This alone prevents the B1–B3 class of error recurring.

**1.3 Reproducibility hardening.** Pin NB01–04 to the same environment as NB05–07. Regenerate or document the loss of `q_tables.npy`. Resolve the 15,108/15,110 drift. Add the distributional regression test (A5).

**Gate 1:** every number in the compiled report traces to a script that reproduces it, and `pytest` fails if the generator drifts from spec.

---

## Phase 2 — Falsifiability (≈3–5 days)

**2.1 Retire the checklist device or repair it.** Currently 41 criteria, 12 failures, verdict never below STRONG. It reads as manufactured rigor.

Replace with a smaller set — **5 to 8 criteria total across the project** — each of which:
- is stated *before* the experiment, in a preregistration file committed ahead of the run;
- has a threshold justified by something external (a published baseline, a chance rate, a theoretical bound) rather than chosen near the observed value;
- has a stated consequence: "if this fails, hypothesis component X is not supported."

**2.2 Add at least one criterion that can kill the hypothesis.** Candidate: *if intrinsic dimension of the telemetry manifold is not measurably below ambient dimension after controlling for the number of regimes, LBSM's core claim is unsupported.* Right now nothing in the project can return that verdict.

**2.3 Fix the statistical validity items.** Replace the LDA leakage baseline with blocked/grouped CV (split by agent or by contiguous time blocks). Drop the pooled-units Pearson r. Either use NB04's train/test split or delete it.

**Gate 2:** a reviewer can point to a specific number that, had it come out differently, would have falsified a stated claim.

---

## Phase 3 — Make "manifold" measurable (≈1–2 weeks) ★

This is the single largest gap. The thesis is about *low-dimensional statistical manifolds* and the project never estimates dimension.

**3.1 Intrinsic dimension estimation.** Add to NB02 (or a new NB02b):
- Two-NN estimator (Facco et al., 2017)
- Levina–Bickel MLE
- Correlation dimension
- PCA participation ratio

Report ID globally, per regime, and along transition segments. The key comparison: is ID of the *transition transients* higher than ID of the regime cores? If the relaxation paths trace a genuine continuous structure, this is where it shows up, and it is measurable.

**3.2 Test the hypothesis's own high-dimensionality clause.** The stated hypothesis says structure emerges "despite high-dimensional observable dynamics." Six features is not high-dimensional. Two options:
- Extend the generator to emit 30–100 correlated derived channels (rolling statistics, ratios, lagged features) from the same 4-state latent process, and show ID recovers the low-dimensional truth as ambient dimension grows.
- Or narrow the hypothesis to drop the clause.

The first is far stronger and is cheap — it reuses the existing generator.

**3.3 Settle the naming.** One expansion of LBSM, used in README, repo description, every `src/` docstring header, and the paper. Currently three variants are in circulation, and "State Machine" and "Structure" are different claims.

**3.4 Decide what "manifold" means operationally** and state it in one paragraph in the paper. If the answer is "a low-ID set with local Euclidean structure," say so and measure it. If the honest answer is "a Gaussian mixture with correlated transients," say that instead and drop the manifold language. Either is fine; the current ambiguity is not.

**Gate 3:** "low-dimensional manifold" appears in the paper only where an estimator backs it.

---

## Phase 4 — Baselines and the regime map (≈2–3 weeks) ★

**4.1 Comparison baselines.** A methods paper with no alternatives compared is not competitive. Minimum set:
- **HSMM** (explicit duration modeling) — the theoretically correct response to short, non-geometric dwell times, and a direct test of whether the BIC pathology is a dwell-modeling failure.
- **HDP-HMM / sticky HDP-HMM** — infers state count directly, sidestepping the model-selection step entirely. If it recovers 4, that is a strong result against the BIC story; if it also over-segments, the finding generalizes beyond BIC.
- **Switching linear dynamical system** — handles within-regime autocorrelation natively, which is exactly what A1 produces.
- **Changepoint detection** (PELT / BOCPD) as a non-parametric floor.

**4.2 Turn NB07's single finding into a phase diagram.** You already have checkpointed parallel grid infrastructure — reuse it. Sweep the *generator* parameters, not just the fitting parameters:

| Axis | Range |
|---|---|
| Regime separation (scale μ spread) | 0.25× → 4× |
| Dwell time (scale T diagonal) | 2 → 50 steps |
| ρ (carryover strength) | 0.0 → 0.9 |
| Ambient dimension | 6 → 100 |
| N, T | existing grid |

Report **where recovery succeeds and where it fails** as a surface. This converts "BIC over-selected in our setup" into "here is the region of generator space where Gaussian-HMM order selection is reliable, and here is where it is not." That is a citable contribution and it is the cheapest available route to A, because the infrastructure exists.

**4.3 Add the effective-sample-size hypothesis (H4 in the register).** With dwell ≈ 2.8 and AR(1) emissions, BIC's `ln N` penalty is computed on a sample size the data does not have. Test ESS-corrected BIC, ICL, and cross-validated held-out likelihood against standard BIC across the grid.

**Gate 4:** the BIC result is stated as a characterized region, not an anecdote, and at least three alternative model classes have been compared on the same data.

---

## Phase 5 — NB08 real-world validation (≈3–4 weeks)

The existing `docs/NB08_REAL_WORLD_GENERALIZATION_PLAN.md` is strong and already handles labels, seasonality, single-sequence statistics, and covariance inheritance. Additions:

**5.1 Lead with Sojourner engineering telemetry** (`MPFR-M-RVRENG-2/3-EDR/RDR-V1.0`), not REMS. Fault counters, motor and wheel faults, thermal and power channels — this is an autonomous sequential system emitting behavioral telemetry, and it is genuinely high-dimensional, which makes it the first real test of the hypothesis's own high-dimensionality clause. REMS is the better *positive control* (known diurnal and Ls structure = external ground truth for a method that claims to find latent structure), not the primary case.

**5.2 Preregister the success criteria before touching the data.** Without labels, define in advance:
- held-out predictive likelihood vs. baselines from Phase 4
- stability of recovered segmentation across resamples (ARI between refits)
- alignment with documented mission events at above-surrogate rates
- ID estimates from Phase 3 applied to real telemetry

**5.3 Surrogate-data controls.** Phase-randomized and AR-matched surrogates, to establish that recovered structure is not an artifact of autocorrelation alone. Given A1, this is essential rather than optional.

**Gate 5:** a stated criterion was met or not met on data with no labels, and the answer was written down before the run.

---

## Phase 0 — Language boundary: R and Python, deliberately bilingual

Applies across all phases. Set this up first; it makes Phases 3 and 4 cheaper.

### Why not a full R rewrite

R does not buy numerical precision over Python. Both use IEEE 754 doubles; NumPy and R call the same LAPACK/BLAS underneath. None of the 48 registered errors is a floating-point error — they are transcription, provenance, and scope errors. A rewrite would not have caught one of them, and rewriting NB05–07 risks the strongest work in the project to fix the weakest.

What *does* address the observed failure mode is **differential testing**: an independent second implementation of the key statistics, diffed automatically.

### 0.1 Differential verification layer (highest value, ~3 days)

`r/statistics/` and `r/reports/` already exist and are half-built. Complete them as a verification layer:

Recompute in R, from the canonical emitted dataset, and diff against the Python values in CI:

| Statistic | Python source | R check |
|---|---|---|
| Transition matrix MAE / max / Frobenius | NB03 | base R |
| Fisher separability ratios | NB01 | base R — **and assert they are computed from data, not constants** |
| Centroid distance ratio (C4) | NB01 | base R |
| Silhouette (PCA and UMAP) | NB02 | `cluster::silhouette` |
| ARI, NMI | NB02, NB03 | `mclust::adjustedRandIndex` |
| Stationary distribution | NB01, NB03 | `expm`, eigendecomposition |
| Per-regime means and SDs | NB01 Table 1 | `dplyr` |
| Confusion matrix, precision/recall/F1, AUC | NB04 | `caret`, `pROC` |

Any disagreement beyond `1e-10` fails the build. This directly targets the error class you actually have — a wrong number can no longer sit unnoticed next to a correct table.

### 0.2 Where R is genuinely specialised — assign these to R outright

**Model selection and mixture inference (Phase 4)**
- `mclust` — ICL natively alongside BIC. ICL is currently absent from the project and is a direct response to the BIC over-selection finding. Also gives EM-based mixtures with a well-tested full covariance parameterization.

**Duration-modelled sequence models (Phase 4, highest priority baseline)**
- `mhsmm` — hidden semi-Markov models with explicit dwell-duration distributions. More mature than any Python HSMM package. This is the theoretically correct answer to dwell ≈ 2.8 steps and the single most informative Phase 4 baseline: if HSMM's order selection is stable where the HMM's is not, the BIC pathology is a dwell-modelling failure rather than a manifold effect.
- `depmixS4` — constrained HMM/mixture fitting with formal LR tests between nested specifications, which the Python stack does not offer cleanly.

**Changepoint detection (Phase 4 baseline)**
- `changepoint` (PELT) and `bcp` (Bayesian changepoint). Canonical implementations; the non-parametric floor your regime-recovery methods must beat.

**Intrinsic dimension (Phase 3 — core of the manifold claim)**
- `intRinsic` — TWO-NN and Hidalgo, maintained by the authors of the method. Better provenance than a reimplementation.
- `intrinsicDimension` — Levina–Bickel MLE, correlation dimension, local PCA estimators.
- Use these as the *primary* ID estimators; treat any Python implementation as the cross-check, not the reverse.

**Formal statistical inference (Phase 2)**
- Hypothesis testing, multiple-comparison correction (`p.adjust`), effect sizes with CIs (`effectsize`), bootstrap (`boot`), surrogate-data tests. Better-vetted and better-documented than ad hoc SciPy usage, which matters when a criterion has to survive review.
- `blockCV` / grouped resampling for the leakage fix in 2.3 — blocked and grouped CV are first-class in R.

**Publication figures — R is the sole source for figures in the final paper**

Scope: this rule applies to the **final paper only**. Notebook visualizations stay in Python (matplotlib/seaborn/plotly) exactly as they are — they are the working record of the analysis, they are already extensive, and there is no reason to touch them. The existing figures across NB01–07 remain as-is.

The boundary is: **if it goes in the paper, it is generated in R.** Nothing else changes.

Rationale beyond aesthetics: the caption/data contradictions in the error register (C4 tortuosity sign, C5 speed direction, C6 porosity maximum) arose because figures were produced separately from the numbers describing them. Regenerating the paper's figure set in R from the canonical dataset closes that gap at the point where it matters most — the version reviewers read.

- `ggplot2` for all paper figures. One shared theme object (`theme_lbsm()`) in `r/visualization/` so the paper is typographically consistent without per-figure styling.
- `patchwork` for multi-panel composition.
- `gt` for publication tables — generated from data, never hand-typed. This alone would have prevented the entire B1–B3 class of error.
- **Captions are computed, not written.** Any numeric value in a paper caption must be an inline `r` expression pulling from the same object that produced the plot. If a caption states a correlation, a ratio, or a direction, it is derived — never typed. This makes C4, C5 and C6 impossible to reproduce in the paper.
- Vector output (`ggsave` to PDF/SVG) at fixed dimensions per figure class, so nothing is rescaled after the fact.
- Colour: one palette defined once, colourblind-safe (`viridis` or Okabe–Ito), with the four regime colours fixed in a single named vector. The current hex codes live in `behavior_profiles.py`; mirror them into `r/visualization/` from the canonical artifact's metadata rather than duplicating the literals — that keeps notebook and paper figures visually consistent without hand-copying.
- Every paper figure script writes a sidecar `.parquet` of exactly the data plotted, so any figure can be audited against its source without rerunning the pipeline.

**Existing scaffolding to complete:** `r/visualization/manifold_figures.R`, `publication_plots.R`, `statistical_heatmaps.R` already exist. These become the paper's figure pipeline. Only the subset of figures selected for the paper needs porting — typically 10–15 of the 100+ generated across the notebooks, so the actual cost is small.

**Reporting**
- `r/reports/*.Rmd` → knit the statistical appendix directly from the data. Any number in the appendix becomes an inline `r` expression, never a typed literal. This makes the transcription errors structurally impossible rather than merely corrected.

### 0.3 Where Python stays

- The generator (`src/simulation/`) — single source of truth, do not port.
- NB05–07 grid infrastructure, checkpointing, parallel execution, audit logging.
- UMAP (`umap-learn` is the reference implementation) — computes the embedding; R plots it.
- `pyhsmm` for sticky HDP-HMM — canonical, no real R equivalent.
- `hmmlearn` for the primary HMM fits, so the existing NB03/NB07 results remain comparable.
- Notebook visualizations — matplotlib, seaborn, plotly, all of it. These are the working record of the analysis and stay in Python. The R rule in 0.2 applies to the paper's figure set only.

### 0.4 Interchange rules (non-negotiable)

1. **Parquet or Feather only.** Never CSV. CSV round-trips silently lose float64 precision unless handled deliberately, and would manufacture exactly the cross-notebook drift already logged as A4 (15,108 vs 15,110).
2. **One canonical dataset artifact.** Generated once in Python, written once, read by both languages. Neither language regenerates it.
3. **One seed source**, recorded in the artifact's metadata alongside package versions for both stacks.
4. **Every table in the paper carries a provenance tag** naming the script and language that produced it (extends item 1.2).

Setup cost: roughly one week. It reduces Phase 3 and Phase 4 effort by more than that, and makes the Phase 1 error class non-recurring.

---

## Sequencing and effort

| Phase | Effort | Cumulative grade |
|---|---|---|
| 0 — Language boundary + differential checks | ~1 wk | (enabler) |
| 1 — Correctness | 1–2 wk | B− |
| 2 — Falsifiability | 3–5 d | B |
| 3 — Manifold made measurable ★ | 1–2 wk | B+ |
| 4 — Baselines + phase diagram ★ | 2–3 wk | **A−** |
| 5 — NB08 real-world | 3–4 wk | **A** |

Phase 0 runs first and pays for itself. Phases 1–2 can run in parallel with the Phase 3 framing rewrite, since both touch the same prose. Phase 4 depends on Phase 1's A1 decision. Phase 5 depends on Phase 3's ID tooling.

Realistic total: **9–13 weeks** to A−, **13–17** to A.

---

## What would still be missing at A

Honest accounting of why A+ stays out of reach:

- **The core methods finding is negative and diagnostic**, not generative. "BIC over-selects under these conditions" is useful and citable; it does not change practice on its own.
- **No new estimator or algorithm.** An A+ version would propose a method that *fixes* the problem it identifies — e.g. a dwell-aware order-selection criterion validated across the Phase 4 grid — rather than characterizing it.
- **Single-author scope.** A+ typically implies multi-dataset, multi-domain generalization.

If you want the A+ path: **Phase 6 would be proposing and validating a corrected order-selection criterion** for autocorrelated, short-dwell, manifold-structured emissions, benchmarked on the Phase 4 phase diagram plus at least two external datasets. That is a separate paper, and it is the natural sequel — the current work is exactly the diagnostic groundwork such a proposal would need to cite.

---

## Minimum viable A−

If time-constrained, the shortest defensible path:

1. Phase 0.1 and 0.4 (differential checks + interchange rules — cheap, and prevents Phase 1 from having to be redone)
2. Phase 1 in full (non-negotiable — nothing else counts without it)
3. Phase 2.2 and 2.3 only (one falsifiable criterion, fix the leakage)
4. Phase 3.1 and 3.3 (intrinsic dimension via `intRinsic` + naming)
5. Phase 4.1 restricted to HSMM (`mhsmm`) and ICL (`mclust`) only — the two highest-information baselines
6. Phase 4.2 restricted to two axes: dwell × separation
7. Defer Phase 5, and scope the paper explicitly as a synthetic characterization study with real-world validation stated as follow-up

That is roughly 6–7 weeks and lands at a solid **A−/B+ boundary** with claims that survive review.
