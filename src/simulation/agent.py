"""
agent.py
========
Latent Behavioral State Machine (LBSM) — Research Implementation
-----------------------------------------------------------------
Defines an adaptive agent whose observable telemetry is governed by a
hidden discrete-state Markov chain over behavioral regimes.

Hidden state sequence:   s_t  ∈  {stable, exploratory, adaptive, unstable}
Observable emissions:    x_t  ~  𝒩(μ_{s_t}, Σ_{s_t})   (diagonal covariance)

The transition structure encodes domain knowledge:
  stable      → exploratory  (curiosity-driven drift)
  exploratory → adaptive     (policy integration)
  adaptive    → stable       (convergence)
  any state   → unstable     (small shock probability)
  unstable    → any state    (recovery / persistence)

Reference
---------
"Latent Behavioral Structure in Low-Dimensional Statistical Manifolds"
Section 3.2 — Agent Dynamics & Emission Model
"""

from __future__ import annotations

import uuid
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from .behavior_profiles import (
    BEHAVIOR_PROFILES,
    PROFILE_NAMES,
    TELEMETRY_FEATURES,
    BehaviorProfile,
    get_profile,
)


# ---------------------------------------------------------------------------
# Default transition matrix  P[i, j] = P(s_{t+1}=j | s_t=i)
# Rows correspond to PROFILE_NAMES order: stable(0) exploratory(1) adaptive(2) unstable(3)
# ---------------------------------------------------------------------------
DEFAULT_TRANSITION_MATRIX = np.array(
    #  stable  explor  adapt   unstab
    [[ 0.75,   0.15,   0.05,   0.05 ],   # from stable
     [ 0.05,   0.55,   0.30,   0.10 ],   # from exploratory
     [ 0.30,   0.10,   0.50,   0.10 ],   # from adaptive
     [ 0.10,   0.10,   0.10,   0.70 ]],  # from unstable
    dtype=np.float64,
)

# Map regime name → integer index
_STATE_INDEX: Dict[str, int] = {name: i for i, name in enumerate(PROFILE_NAMES)}
_INDEX_STATE: Dict[int, str] = {i: name for i, name in enumerate(PROFILE_NAMES)}


# ---------------------------------------------------------------------------
# Agent
# ---------------------------------------------------------------------------
class AdaptiveAgent:
    """Simulates a single adaptive agent emitting behavioural telemetry.

    The agent maintains a *hidden* discrete state (the behavioral regime) that
    evolves according to a Markov chain and emits observable telemetry drawn
    from the corresponding :class:`~simulation.behavior_profiles.BehaviorProfile`.

    Parameters
    ----------
    agent_id : str | None
        Unique identifier.  Auto-generated (UUID4 prefix) if *None*.
    initial_state : str
        Starting regime.  Defaults to ``"stable"``.
    transition_matrix : np.ndarray | None
        Row-stochastic (4×4) transition matrix.  Uses
        :data:`DEFAULT_TRANSITION_MATRIX` if *None*.
    rng_seed : int | None
        Seed for the internal :class:`numpy.random.Generator`.
    """

    def __init__(
        self,
        agent_id: Optional[str] = None,
        initial_state: str = "stable",
        transition_matrix: Optional[np.ndarray] = None,
        rng_seed: Optional[int] = None,
    ) -> None:
        self.agent_id: str = agent_id or ("agent_" + uuid.uuid4().hex[:8])
        self._rng: np.random.Generator = np.random.default_rng(rng_seed)

        # Validate / store transition matrix
        if transition_matrix is None:
            self._T = DEFAULT_TRANSITION_MATRIX.copy()
        else:
            self._T = np.asarray(transition_matrix, dtype=np.float64)
            self._validate_transition_matrix(self._T)

        # Hidden state
        if initial_state not in _STATE_INDEX:
            raise ValueError(
                f"Unknown initial_state {initial_state!r}. "
                f"Valid: {list(_STATE_INDEX.keys())}"
            )
        self._state_idx: int = _STATE_INDEX[initial_state]

        # Running telemetry buffer (populated by step / simulate)
        self._history: List[Dict] = []

        # AR-1 carry-over: last emitted telemetry vector
        self._prev_telemetry: Optional[np.ndarray] = None

    # ------------------------------------------------------------------ #
    # Properties
    # ------------------------------------------------------------------ #
    @property
    def current_state(self) -> str:
        """Current hidden behavioral regime name."""
        return _INDEX_STATE[self._state_idx]

    @property
    def current_profile(self) -> BehaviorProfile:
        """Statistical profile of the current hidden state."""
        return BEHAVIOR_PROFILES[self.current_state]

    @property
    def history(self) -> pd.DataFrame:
        """Full telemetry history as a :class:`pandas.DataFrame`."""
        return pd.DataFrame(self._history)

    # ------------------------------------------------------------------ #
    # Core dynamics
    # ------------------------------------------------------------------ #
    def _transition(self) -> None:
        """Sample the next hidden state from the Markov transition row."""
        probs = self._T[self._state_idx]
        self._state_idx = int(
            self._rng.choice(len(PROFILE_NAMES), p=probs)
        )

    def _emit(self) -> np.ndarray:
        """Sample one telemetry observation from the current profile.

        Uses AR-1 temporal correlation seeded by the previous emission.
        """
        profile = self.current_profile
        sample = profile.sample(n=1, rng=self._rng, prev=self._prev_telemetry)
        vec = sample[0]
        self._prev_telemetry = vec
        return vec

    def step(self, timestep: int) -> Dict:
        """Advance the agent by one discrete time step.

        Sequence: emit telemetry → transition hidden state.

        Parameters
        ----------
        timestep : int
            Global simulation clock tick (stored in history).

        Returns
        -------
        record : dict
            One row of telemetry (agent_id, timestep, hidden_state, features…).
        """
        # 1. Emit observable telemetry under current hidden state
        telemetry_vec = self._emit()

        # 2. Build record
        record: Dict = {
            "agent_id":    self.agent_id,
            "timestep":    timestep,
            "hidden_state": self.current_state,
        }
        for feat, val in zip(TELEMETRY_FEATURES, telemetry_vec):
            record[feat] = float(val)

        self._history.append(record)

        # 3. Stochastic state transition (for *next* step)
        prev_state_idx = self._state_idx
        self._transition()
        if self._state_idx != prev_state_idx:
            # Don't let the outgoing regime's emission leak into the
            # incoming regime's AR(1) seed.
            self._prev_telemetry = None

        return record

    def simulate(self, n_steps: int, start_timestep: int = 0) -> pd.DataFrame:
        """Run the agent for *n_steps* steps.

        Parameters
        ----------
        n_steps        : number of discrete time steps
        start_timestep : offset added to the step index (for multi-segment runs)

        Returns
        -------
        df : pd.DataFrame  shape (n_steps, 2 + N_FEATURES)
        """
        for t in range(start_timestep, start_timestep + n_steps):
            self.step(t)
        return self.history.tail(n_steps).reset_index(drop=True)

    def reset(
        self,
        initial_state: str = "stable",
        clear_history: bool = True,
    ) -> None:
        """Reset the agent to a clean initial condition.

        Parameters
        ----------
        initial_state : regime to begin from after reset
        clear_history : if True, wipe the telemetry history buffer
        """
        if initial_state not in _STATE_INDEX:
            raise ValueError(f"Unknown state {initial_state!r}.")
        self._state_idx = _STATE_INDEX[initial_state]
        self._prev_telemetry = None
        if clear_history:
            self._history.clear()

    # ------------------------------------------------------------------ #
    # Introspection helpers
    # ------------------------------------------------------------------ #
    def state_distribution(self) -> Dict[str, float]:
        """Empirical fraction of time spent in each regime (over history)."""
        if not self._history:
            return {name: 0.0 for name in PROFILE_NAMES}
        states = [r["hidden_state"] for r in self._history]
        total = len(states)
        return {
            name: states.count(name) / total
            for name in PROFILE_NAMES
        }

    def transition_counts(self) -> np.ndarray:
        """Empirical (4×4) transition count matrix from stored history.

        Useful for verifying that the simulated chain matches
        :attr:`_T` asymptotically.
        """
        k = len(PROFILE_NAMES)
        counts = np.zeros((k, k), dtype=int)
        states = [r["hidden_state"] for r in self._history]
        for s_from, s_to in zip(states[:-1], states[1:]):
            counts[_STATE_INDEX[s_from], _STATE_INDEX[s_to]] += 1
        return counts

    def stationary_distribution(self) -> np.ndarray:
        """Theoretical stationary distribution π of the Markov chain.

        Computed as the left eigenvector of :attr:`_T` corresponding to
        eigenvalue 1 (normalised to sum to 1).

        Returns
        -------
        pi : np.ndarray  shape (4,)  with PROFILE_NAMES order
        """
        # Solve π^T T = π^T  ⟺  (T^T − I) π = 0
        A = (self._T.T - np.eye(len(PROFILE_NAMES)))
        # Replace last equation with normalisation constraint
        A[-1, :] = 1.0
        b = np.zeros(len(PROFILE_NAMES))
        b[-1] = 1.0
        pi = np.linalg.solve(A, b)
        return np.clip(pi, 0.0, 1.0)

    # ------------------------------------------------------------------ #
    # Dunder
    # ------------------------------------------------------------------ #
    def __repr__(self) -> str:  # pragma: no cover
        return (
            f"AdaptiveAgent(id={self.agent_id!r}, "
            f"state={self.current_state!r}, "
            f"history_len={len(self._history)})"
        )

    # ------------------------------------------------------------------ #
    # Private validators
    # ------------------------------------------------------------------ #
    @staticmethod
    def _validate_transition_matrix(T: np.ndarray) -> None:
        k = len(PROFILE_NAMES)
        if T.shape != (k, k):
            raise ValueError(f"Transition matrix must be ({k},{k}), got {T.shape}.")
        row_sums = T.sum(axis=1)
        if not np.allclose(row_sums, 1.0, atol=1e-6):
            raise ValueError(
                f"All rows must sum to 1. Got row sums: {row_sums}."
            )
        if (T < 0).any():
            raise ValueError("Transition probabilities must be non-negative.")


# ---------------------------------------------------------------------------
# Factory helpers
# ---------------------------------------------------------------------------
def make_agent(
    agent_id: Optional[str] = None,
    initial_state: str = "stable",
    rng_seed: Optional[int] = None,
    **kwargs,
) -> AdaptiveAgent:
    """Convenience factory wrapping :class:`AdaptiveAgent`."""
    return AdaptiveAgent(
        agent_id=agent_id,
        initial_state=initial_state,
        rng_seed=rng_seed,
        **kwargs,
    )


def make_agent_pool(
    n_agents: int,
    initial_states: Optional[Sequence[str]] = None,
    base_seed: int = 42,
) -> List[AdaptiveAgent]:
    """Create a heterogeneous pool of agents.

    Parameters
    ----------
    n_agents       : number of agents
    initial_states : per-agent starting regimes (cycled if shorter than n_agents);
                     defaults to cycling through all four regimes.
    base_seed      : seeds are  base_seed, base_seed+1, …

    Returns
    -------
    agents : list of :class:`AdaptiveAgent`
    """
    if initial_states is None:
        initial_states = list(PROFILE_NAMES)

    agents = []
    for i in range(n_agents):
        state = initial_states[i % len(initial_states)]
        agent = AdaptiveAgent(
            agent_id=f"agent_{i:04d}",
            initial_state=state,
            rng_seed=base_seed + i,
        )
        agents.append(agent)
    return agents