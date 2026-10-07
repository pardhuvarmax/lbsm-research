"""
behavior_profiles.py
====================
Latent Behavioral State Machine (LBSM) — Research Implementation
-----------------------------------------------------------------
Defines the statistical ground truth for each hidden behavioral regime.

Each profile encodes:
  - mean telemetry vector (μ)
  - per-feature standard deviations (σ)
  - temporal correlation hints (for realistic time-series generation)

Reference
---------
"Latent Behavioral Structure in Low-Dimensional Statistical Manifolds"
Section 3.1 — Behavioral Regime Parameterization
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Tuple

import numpy as np


# ---------------------------------------------------------------------------
# Telemetry feature order (canonical)
# ---------------------------------------------------------------------------
TELEMETRY_FEATURES: Tuple[str, ...] = (
    "latency",        # ms   — response / action latency
    "entropy",        # bits — policy / action-distribution entropy
    "reward",         # a.u. — instantaneous reward signal
    "memory_usage",   # MB   — working-set memory footprint
    "error_rate",     # [0,1]— proportion of erroneous actions
    "action_freq",    # Hz   — actions per second
)

N_FEATURES = len(TELEMETRY_FEATURES)


# ---------------------------------------------------------------------------
# Data container
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class BehaviorProfile:
    """Immutable statistical descriptor for one hidden behavioral regime.

    Parameters
    ----------
    name : str
        Human-readable regime label (e.g. ``"stable"``).
    means : np.ndarray  shape (N_FEATURES,)
        Expected value of each telemetry feature under this regime.
    stds : np.ndarray   shape (N_FEATURES,)
        Standard deviation of each telemetry feature.
    autocorr : float | np.ndarray
        AR(1) autocorrelation coefficient(s) for temporal smoothing (ρ ∈ [0, 1)),
        one per feature (broadcast from a scalar if a single float is given).
        Higher → smoother trajectories within the regime.
    description : str
        Plain-language characterisation (used in paper figures / captions).
    color : str
        Matplotlib color used for consistent visualisation across notebooks.
    """

    name: str
    means: np.ndarray
    stds: np.ndarray
    autocorr: np.ndarray = field(default_factory=lambda: np.full(6, 0.3))
    description: str = ""
    color: str = "#333333"

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "autocorr", np.broadcast_to(np.asarray(self.autocorr, dtype=np.float64), (N_FEATURES,)).copy()
        )

    # ------------------------------------------------------------------ #
    # Derived helpers
    # ------------------------------------------------------------------ #
    def sample(
        self,
        n: int = 1,
        rng: np.random.Generator | None = None,
        prev: np.ndarray | None = None,
    ) -> np.ndarray:
        """Draw *n* i.i.d. (or AR-1 correlated) samples from this regime.

        Parameters
        ----------
        n   : number of time steps to sample
        rng : numpy random Generator (reproducibility)
        prev: previous telemetry vector for AR-1 seeding (shape N_FEATURES)

        Returns
        -------
        samples : np.ndarray  shape (n, N_FEATURES)
        """
        if rng is None:
            rng = np.random.default_rng()

        noise = rng.normal(loc=0.0, scale=1.0, size=(n, N_FEATURES))
        samples = np.empty((n, N_FEATURES))

        x_prev = prev if prev is not None else self.means.copy()

        for t in range(n):
            eps = self.stds * noise[t]
            x_ar1 = self.autocorr * x_prev + (1.0 - self.autocorr) * self.means + eps
            # Clip physically impossible values
            x_ar1 = np.clip(x_ar1, a_min=_FEATURE_LOWER_BOUNDS, a_max=_FEATURE_UPPER_BOUNDS)
            samples[t] = x_ar1
            x_prev = x_ar1

        return samples

    def mahalanobis(self, x: np.ndarray) -> float:
        """Compute Mahalanobis distance from *x* to this profile's centroid
        (diagonal covariance assumption for efficiency)."""
        delta = (x - self.means) / (self.stds + 1e-9)
        return float(np.sqrt(np.dot(delta, delta)))

    def __repr__(self) -> str:  # pragma: no cover
        mu_str = np.array2string(self.means, precision=2, suppress_small=True)
        return f"BehaviorProfile(name={self.name!r}, μ={mu_str})"


# ---------------------------------------------------------------------------
# Physical bounds per feature  (lower, upper)
# ---------------------------------------------------------------------------
_FEATURE_LOWER_BOUNDS = np.array([0.0,  0.0, -10.0,   0.0, 0.0, 0.0])
_FEATURE_UPPER_BOUNDS = np.array([2000.0, 10.0, 100.0, 4096.0, 1.0, 200.0])


# ---------------------------------------------------------------------------
# Regime definitions
# ---------------------------------------------------------------------------
#
#  Feature order:  latency | entropy | reward | memory_usage | error_rate | action_freq
#                  (ms)      (bits)    (a.u.)   (MB)           [0,1]        (Hz)
#
BEHAVIOR_PROFILES: Dict[str, BehaviorProfile] = {

    # ------------------------------------------------------------------ #
    "stable": BehaviorProfile(
        name="stable",
        #                lat    ent   rew   mem   err   afreq
        means=np.array([50.0,  0.8,  8.5,  120.0, 0.02, 15.0]),
        stds =np.array([8.0,   0.12, 0.6,   15.0, 0.008, 2.0]),
        autocorr=0.70,
        description=(
            "Agent operates in a well-converged policy. "
            "Low latency, near-zero error rate, high reward, "
            "low entropy (deterministic action selection)."
        ),
        color="#2ecc71",
    ),

    # ------------------------------------------------------------------ #
    "exploratory": BehaviorProfile(
        name="exploratory",
        #                lat    ent   rew   mem   err   afreq
        means=np.array([120.0, 2.8,  5.0,  200.0, 0.10, 25.0]),
        stds =np.array([ 30.0, 0.55, 1.2,   35.0, 0.04, 6.0]),
        autocorr=0.40,
        description=(
            "Agent actively explores the action space. "
            "High entropy reflects stochastic policy; "
            "reward is moderate; memory grows with exploration buffers."
        ),
        color="#3498db",
    ),

    # ------------------------------------------------------------------ #
    "adaptive": BehaviorProfile(
        name="adaptive",
        #                lat    ent   rew   mem   err   afreq
        means=np.array([85.0,  1.6,  6.8,  160.0, 0.06, 20.0]),
        stds =np.array([20.0,  0.35, 1.0,   25.0, 0.025, 4.0]),
        autocorr=0.55,
        description=(
            "Transitional regime: agent integrates exploratory observations "
            "into an updated policy. Moderate entropy and latency; "
            "reward climbing; adaptation manifests as smooth drift in manifold."
        ),
        color="#9b59b6",
    ),

    # ------------------------------------------------------------------ #
    "unstable": BehaviorProfile(
        name="unstable",
        #                lat    ent   rew    mem    err    afreq
        means=np.array([350.0, 4.5,  1.5,  380.0,  0.35, 50.0]),
        stds =np.array([120.0, 1.20, 2.5,   80.0,  0.15, 18.0]),
        autocorr=0.10,
        description=(
            "Catastrophic / chaotic regime. High latency spikes, "
            "near-maximum entropy, frequent errors, elevated memory pressure, "
            "and bursts of poorly-directed action. "
            "Trajectories exhibit no stable attractor in latent space."
        ),
        color="#e74c3c",
    ),
}


# Convenience ordered list (used for consistent legend ordering)
PROFILE_NAMES: Tuple[str, ...] = ("stable", "exploratory", "adaptive", "unstable")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def get_profile(name: str) -> BehaviorProfile:
    """Retrieve a :class:`BehaviorProfile` by regime name.

    Raises
    ------
    KeyError
        If *name* is not a recognised regime.
    """
    if name not in BEHAVIOR_PROFILES:
        valid = list(BEHAVIOR_PROFILES.keys())
        raise KeyError(f"Unknown regime {name!r}. Valid options: {valid}")
    return BEHAVIOR_PROFILES[name]


def profile_distance_matrix() -> np.ndarray:
    """Return pairwise Euclidean distances between profile centroids.

    Returns
    -------
    D : np.ndarray  shape (4, 4)
        ``D[i, j]`` = ||μ_i − μ_j||₂  (un-normalised feature space).
    """
    k = len(PROFILE_NAMES)
    D = np.zeros((k, k))
    means = np.stack([BEHAVIOR_PROFILES[n].means for n in PROFILE_NAMES])
    for i in range(k):
        for j in range(k):
            D[i, j] = np.linalg.norm(means[i] - means[j])
    return D


def regime_separability_ratio() -> Dict[str, float]:
    """Compute a simple between-class / within-class variance ratio per feature.

    Higher values indicate that a feature is more discriminative across regimes
    (analogous to a univariate Fisher criterion).

    Returns
    -------
    ratios : dict  feature_name → separability ratio
    """
    means_matrix = np.stack([BEHAVIOR_PROFILES[n].means for n in PROFILE_NAMES])
    stds_matrix  = np.stack([BEHAVIOR_PROFILES[n].stds  for n in PROFILE_NAMES])

    grand_mean = means_matrix.mean(axis=0)
    between_var = np.mean((means_matrix - grand_mean) ** 2, axis=0)
    within_var  = np.mean(stds_matrix ** 2, axis=0)

    ratios = {
        feat: float(between_var[i] / (within_var[i] + 1e-12))
        for i, feat in enumerate(TELEMETRY_FEATURES)
    }
    return ratios