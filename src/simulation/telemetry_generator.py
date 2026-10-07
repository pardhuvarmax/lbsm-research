"""
telemetry_generator.py
======================
Latent Behavioral State Machine (LBSM) — Research Implementation
-----------------------------------------------------------------
Orchestrates large-scale telemetry simulation across a pool of
:class:`~simulation.agent.AdaptiveAgent` instances and exports the
resulting dataset for downstream analysis (manifold learning, RL, PCA).

Typical usage
-------------
>>> from simulation.telemetry_generator import TelemetryGenerator
>>> gen = TelemetryGenerator(n_agents=10, n_timesteps=1000, seed=0)
>>> df = gen.run()
>>> gen.save("data/telemetry.csv")

Reference
---------
"Latent Behavioral Structure in Low-Dimensional Statistical Manifolds"
Section 4 — Experimental Setup & Dataset Construction
"""

from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import numpy as np
import pandas as pd

from .agent import AdaptiveAgent, make_agent_pool
from .behavior_profiles import PROFILE_NAMES, TELEMETRY_FEATURES

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Generator
# ---------------------------------------------------------------------------
class TelemetryGenerator:
    """Simulate a heterogeneous pool of adaptive agents and collect telemetry.

    Parameters
    ----------
    n_agents     : int
        Number of parallel agents.
    n_timesteps  : int
        Number of discrete simulation steps per agent.
    initial_states : list[str] | None
        Per-agent starting regimes (cycled).  Defaults to round-robin over
        all four behavioral regimes for diversity.
    seed         : int
        Master RNG seed.  Per-agent seeds are derived as ``seed + i``.
    verbose      : bool
        Emit progress logs via the standard :mod:`logging` module.
    """

    def __init__(
        self,
        n_agents: int = 10,
        n_timesteps: int = 1_000,
        initial_states: Optional[Sequence[str]] = None,
        seed: int = 42,
        verbose: bool = True,
    ) -> None:
        self.n_agents = n_agents
        self.n_timesteps = n_timesteps
        self.seed = seed
        self.verbose = verbose

        self._agents: List[AdaptiveAgent] = make_agent_pool(
            n_agents=n_agents,
            initial_states=list(initial_states) if initial_states else None,
            base_seed=seed,
        )

        self._telemetry_df: Optional[pd.DataFrame] = None

    # ------------------------------------------------------------------ #
    # Simulation
    # ------------------------------------------------------------------ #
    def run(self, reset_agents: bool = False) -> pd.DataFrame:
        """Execute the simulation for all agents.

        Parameters
        ----------
        reset_agents : if True, reset each agent before simulating
                       (useful for re-running with different parameters).

        Returns
        -------
        df : pd.DataFrame
            Shape ``(n_agents × n_timesteps, 3 + N_FEATURES)``.
            Columns: ``agent_id``, ``timestep``, ``hidden_state``,
            plus one column per telemetry feature.
        """
        t0 = time.perf_counter()
        if self.verbose:
            logger.info(
                "Starting simulation: %d agents × %d timesteps = %d observations",
                self.n_agents, self.n_timesteps, self.n_agents * self.n_timesteps,
            )

        records: List[pd.DataFrame] = []

        for idx, agent in enumerate(self._agents):
            if reset_agents:
                agent.reset(clear_history=True)

            agent_df = agent.simulate(self.n_timesteps, start_timestep=0)
            records.append(agent_df)

            if self.verbose and (idx + 1) % max(1, self.n_agents // 5) == 0:
                logger.info("  Simulated %d / %d agents …", idx + 1, self.n_agents)

        self._telemetry_df = pd.concat(records, ignore_index=True)
        self._add_metadata_columns()

        elapsed = time.perf_counter() - t0
        if self.verbose:
            logger.info(
                "Simulation complete in %.2f s. Dataset shape: %s",
                elapsed, self._telemetry_df.shape,
            )

        return self._telemetry_df

    # ------------------------------------------------------------------ #
    # Post-processing
    # ------------------------------------------------------------------ #
    def _add_metadata_columns(self) -> None:
        """Attach derived columns that simplify downstream analysis."""
        df = self._telemetry_df

        # Integer regime label (stable=0, exploratory=1, adaptive=2, unstable=3)
        _label_map = {name: i for i, name in enumerate(PROFILE_NAMES)}
        df["state_label"] = df["hidden_state"].map(_label_map).astype(np.int8)

        # Composite anomaly indicator: error_rate > 0.2  OR  latency > 300 ms
        df["is_anomaly"] = (
            (df["error_rate"] > 0.20) | (df["latency"] > 300.0)
        ).astype(np.int8)

        # Z-score normalised features (useful for direct manifold input)
        for feat in TELEMETRY_FEATURES:
            mu  = df[feat].mean()
            std = df[feat].std(ddof=1) + 1e-9
            df[f"{feat}_z"] = (df[feat] - mu) / std

    # ------------------------------------------------------------------ #
    # IO
    # ------------------------------------------------------------------ #
    def save(
        self,
        path: str | Path = "telemetry.csv",
        include_z_scores: bool = True,
    ) -> Path:
        """Persist the telemetry dataset to a CSV file.

        Parameters
        ----------
        path             : output path (parent dirs created automatically)
        include_z_scores : whether to include the ``*_z`` normalised columns

        Returns
        -------
        resolved : Path  — absolute path of the written file
        """
        if self._telemetry_df is None:
            raise RuntimeError("No data available. Call .run() first.")

        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        df = self._telemetry_df
        if not include_z_scores:
            z_cols = [c for c in df.columns if c.endswith("_z")]
            df = df.drop(columns=z_cols)

        df.to_csv(path, index=False)

        if self.verbose:
            logger.info("Telemetry saved → %s  (%d rows, %d cols)",
                        path.resolve(), len(df), df.shape[1])
        return path.resolve()

    @classmethod
    def load(cls, path: str | Path) -> pd.DataFrame:
        """Load a previously saved telemetry CSV.

        Returns
        -------
        df : pd.DataFrame
        """
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Telemetry file not found: {path}")
        df = pd.read_csv(path)
        logger.info("Loaded telemetry from %s  (%d rows)", path, len(df))
        return df

    # ------------------------------------------------------------------ #
    # Introspection / statistics
    # ------------------------------------------------------------------ #
    @property
    def data(self) -> pd.DataFrame:
        """The current telemetry DataFrame (None until :meth:`run` is called)."""
        if self._telemetry_df is None:
            raise RuntimeError("No data available. Call .run() first.")
        return self._telemetry_df

    def summary_statistics(self) -> pd.DataFrame:
        """Per-regime descriptive statistics for all telemetry features.

        Returns
        -------
        stats : pd.DataFrame
            MultiIndex (regime, statistic) × feature.
        """
        df = self.data
        stats = (
            df.groupby("hidden_state")[list(TELEMETRY_FEATURES)]
            .agg(["mean", "std", "min", "max"])
        )
        return stats

    def state_frequencies(self) -> pd.Series:
        """Global fraction of timesteps spent in each behavioral regime."""
        counts = self.data["hidden_state"].value_counts(normalize=True)
        # Ensure all regimes appear even if count is zero
        return counts.reindex(PROFILE_NAMES, fill_value=0.0)

    def per_agent_statistics(self) -> pd.DataFrame:
        """Per-agent summary: dominant state, mean reward, mean error rate."""
        df = self.data
        rows = []
        for aid, grp in df.groupby("agent_id"):
            rows.append(
                {
                    "agent_id":       aid,
                    "dominant_state": grp["hidden_state"].mode()[0],
                    "mean_reward":    grp["reward"].mean(),
                    "mean_error":     grp["error_rate"].mean(),
                    "mean_latency":   grp["latency"].mean(),
                    # .shift() leaves the first row NaN, and NaN != anything is
                    # True, so counting from row 0 overcounts every agent's
                    # transitions by exactly one (a phantom transition into the
                    # first timestep). Compare consecutive pairs directly instead.
                    "n_transitions": int((grp["hidden_state"].values[:-1] != grp["hidden_state"].values[1:]).sum()),
                }
            )
        return pd.DataFrame(rows).set_index("agent_id")

    def feature_matrix(self, z_scored: bool = True) -> np.ndarray:
        """Return the raw feature matrix for manifold / PCA input.

        Parameters
        ----------
        z_scored : if True use pre-computed z-score columns; else raw values.

        Returns
        -------
        X : np.ndarray  shape (n_agents × n_timesteps, N_FEATURES)
        """
        df = self.data
        if z_scored:
            cols = [f"{f}_z" for f in TELEMETRY_FEATURES]
        else:
            cols = list(TELEMETRY_FEATURES)
        return df[cols].to_numpy(dtype=np.float32)

    def labels(self) -> np.ndarray:
        """Integer ground-truth regime labels.  shape (n_agents × n_timesteps,)"""
        return self.data["state_label"].to_numpy(dtype=np.int8)

    # ------------------------------------------------------------------ #
    # Dunder
    # ------------------------------------------------------------------ #
    def __repr__(self) -> str:  # pragma: no cover
        status = "ready" if self._telemetry_df is not None else "not run"
        return (
            f"TelemetryGenerator("
            f"n_agents={self.n_agents}, "
            f"n_timesteps={self.n_timesteps}, "
            f"status={status!r})"
        )


# ---------------------------------------------------------------------------
# Convenience entry-point
# ---------------------------------------------------------------------------
def generate_and_save(
    output_path: str | Path = "data/telemetry.csv",
    n_agents: int = 10,
    n_timesteps: int = 1_000,
    seed: int = 42,
) -> pd.DataFrame:
    """One-shot helper: simulate, save, and return the telemetry DataFrame.

    Parameters
    ----------
    output_path  : path for the CSV file
    n_agents     : number of parallel agents
    n_timesteps  : steps per agent
    seed         : master RNG seed

    Returns
    -------
    df : pd.DataFrame
    """
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(levelname)-7s  %(message)s",
        datefmt="%H:%M:%S",
    )

    gen = TelemetryGenerator(
        n_agents=n_agents,
        n_timesteps=n_timesteps,
        seed=seed,
        verbose=True,
    )
    df = gen.run()
    gen.save(output_path)
    return df


if __name__ == "__main__":
    generate_and_save()