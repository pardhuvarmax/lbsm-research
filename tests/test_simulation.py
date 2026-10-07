# tests/test_simulation.py

import numpy as np
import pytest

from src.simulation.agent import AdaptiveAgent, make_agent_pool
from src.simulation.behavior_profiles import (
    BEHAVIOR_PROFILES,
    PROFILE_NAMES,
    TELEMETRY_FEATURES,
)


# ---------------------------------------------------------------------------
# Agent initialization
# ---------------------------------------------------------------------------

def test_agent_initialization():

    agent = AdaptiveAgent(
        agent_id="test_agent",
        initial_state="stable",
    )

    print("\n=== AGENT INITIALIZATION ===")

    print("Agent ID:", agent.agent_id)
    print("Initial state:", agent.current_state)
    print("History length:", len(agent.history))

    assert agent.current_state == "stable"
    assert len(agent.history) == 0


# ---------------------------------------------------------------------------
# Single simulation step
# ---------------------------------------------------------------------------

def test_agent_step():

    agent = AdaptiveAgent(rng_seed=42)

    record = agent.step(timestep=0)

    print("\n=== AGENT STEP OUTPUT ===")

    for k, v in record.items():
        print(f"{k:20s}: {v}")

    print("\nTelemetry features present:")

    for feat in TELEMETRY_FEATURES:
        print(f"  {feat}: {'YES' if feat in record else 'NO'}")

    assert record["timestep"] == 0
    assert record["hidden_state"] in PROFILE_NAMES
    assert all(feat in record for feat in TELEMETRY_FEATURES)
    assert len(agent.history) == 1


# ---------------------------------------------------------------------------
# Transition matrix validation
# ---------------------------------------------------------------------------

def test_transition_matrix_validation():

    invalid_T = np.array([
        [0.5, 0.5],
        [0.5, 0.5],
    ])

    print("\n=== INVALID TRANSITION MATRIX ===")
    print(invalid_T)

    with pytest.raises(ValueError):
        AdaptiveAgent._validate_transition_matrix(invalid_T)

    print("Validation correctly raised ValueError")


# ---------------------------------------------------------------------------
# Stationary distribution
# ---------------------------------------------------------------------------

def test_stationary_distribution():

    agent = AdaptiveAgent()

    pi = agent.stationary_distribution()

    print("\n=== STATIONARY DISTRIBUTION ===")

    for state, prob in zip(PROFILE_NAMES, pi):
        print(f"{state:15s}: {prob:.6f}")

    print("\nDistribution sum:", pi.sum())

    assert np.allclose(pi.sum(), 1.0)
    assert (pi >= 0).all()


# ---------------------------------------------------------------------------
# Distributional correctness: emitted telemetry vs. the profile it was
# emitted under. Guards against AR(1) state leaking across regime
# transitions (each regime's realized mean/std should track its configured
# BehaviorProfile, not be inflated by the previous regime's carryover).
# ---------------------------------------------------------------------------

def test_emission_matches_profile_per_regime():

    agents = make_agent_pool(20, base_seed=42)
    for agent in agents:
        agent.simulate(2000)

    import pandas as pd

    df = pd.concat([agent.history for agent in agents], ignore_index=True)

    print("\n=== REALIZED VS CONFIGURED PER REGIME ===")

    for state in PROFILE_NAMES:
        sub = df[df["hidden_state"] == state]
        profile = BEHAVIOR_PROFILES[state]

        assert len(sub) > 100, f"too few samples in regime {state!r} to check"

        for i, feat in enumerate(TELEMETRY_FEATURES):
            realized_mean = sub[feat].mean()
            realized_std = sub[feat].std()
            cfg_mean = profile.means[i]
            cfg_std = profile.stds[i]

            print(
                f"{state:12s} {feat:13s} "
                f"mean cfg={cfg_mean:8.3f} realized={realized_mean:8.3f}  "
                f"std cfg={cfg_std:8.3f} realized={realized_std:8.3f}"
            )

            # Realized mean within a few configured std's of the true mean,
            # and realized std within 2x the configured std (loose bounds —
            # this is a leak detector, not a precise distributional test).
            assert abs(realized_mean - cfg_mean) < 3 * cfg_std, (
                f"{state}/{feat}: realized mean {realized_mean:.3f} too far "
                f"from configured {cfg_mean:.3f} (possible AR(1) carryover leak)"
            )
            assert realized_std < 2 * cfg_std, (
                f"{state}/{feat}: realized std {realized_std:.3f} more than "
                f"2x configured {cfg_std:.3f} (possible AR(1) carryover leak)"
            )