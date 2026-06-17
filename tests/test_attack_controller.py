"""Unit tests for the Cyber-Physical Attack Orchestrator.

These tests are SUMO-free: they build a synthetic VANET observation vector and a
fake TrafficSignal so that the attack injection logic can be validated in
isolation. They prove the two scientific guarantees of the redesigned
``corrupt_observation``:

1. Each attack perturbs ONLY the features it claims to target (exact indexing).
2. The returned observation can NEVER leave the Box[0, 1] observation space.
"""

import os
import tempfile

import numpy as np
import pytest

from sumo_rl.environment.attack_controller import (
    AttackType,
    CyberPhysicalAttackOrchestrator,
    compute_obs_layout,
)


class FakeTrafficSignal:
    """Minimal stand-in for TrafficSignal (no SUMO/TraCI required)."""

    def __init__(self, ts_id="A0", num_green_phases=4, n_lanes=3):
        self.id = ts_id
        self.num_green_phases = num_green_phases
        self.lanes = [f"lane_{i}" for i in range(n_lanes)]
        self.comm_ok = True


def build_observation(ts, density=0.5, queue=0.5, latency=0.1, comm=1.0):
    """Build a synthetic VANET observation matching VANETObservationFunction layout.

    Layout: [phase_one_hot | min_green | density | queue | latency | comm_flag]
    """
    n_lanes = len(ts.lanes)
    phase_one_hot = [0.0] * ts.num_green_phases
    phase_one_hot[0] = 1.0
    min_green = [1.0]
    density_block = [density] * n_lanes
    queue_block = [queue] * n_lanes
    obs = phase_one_hot + min_green + density_block + queue_block + [latency, comm]
    return np.array(obs, dtype=np.float32)


@pytest.fixture
def orchestrator():
    """Orchestrator writing its telemetry to a throwaway temp file."""
    tmp = tempfile.NamedTemporaryFile(suffix=".csv", delete=False)
    tmp.close()
    orch = CyberPhysicalAttackOrchestrator(log_path=tmp.name)
    yield orch
    os.unlink(tmp.name)


def _slices(ts):
    layout = compute_obs_layout(ts)
    od, oq, n = layout["offset_density"], layout["offset_queue"], layout["n_lanes"]
    return slice(od, od + n), slice(oq, oq + n)


def test_layout_matches_observation_structure():
    ts = FakeTrafficSignal(num_green_phases=4, n_lanes=3)
    layout = compute_obs_layout(ts)
    # phase(4) + min_green(1) = 5 -> density starts at index 5
    assert layout["offset_density"] == 5
    # density(3) -> queue starts at index 8
    assert layout["offset_queue"] == 8
    assert layout["n_lanes"] == 3


def test_no_attack_returns_observation_untouched(orchestrator):
    ts = FakeTrafficSignal()
    obs = build_observation(ts)
    out = orchestrator.corrupt_observation(ts, obs)
    np.testing.assert_array_equal(out, obs)
    assert ts.comm_ok is True


def test_data_poisoning_touches_only_queue(orchestrator):
    ts = FakeTrafficSignal()
    obs = build_observation(ts, density=0.5, queue=0.8)
    density_slice, queue_slice = _slices(ts)
    orchestrator.trigger_manual_attack(ts.id, AttackType.DATA_POISONING, 0.5, duration_steps=2)
    out = orchestrator.corrupt_observation(ts, obs)
    # Queue must be reduced (embouteillages cachés)
    assert np.all(out[queue_slice] < obs[queue_slice])
    # Density must be untouched
    np.testing.assert_array_almost_equal(out[density_slice], obs[density_slice])
    assert np.all((out >= 0.0) & (out <= 1.0))


def test_ghost_vehicles_touches_only_density(orchestrator):
    ts = FakeTrafficSignal()
    obs = build_observation(ts, density=0.3, queue=0.4)
    density_slice, queue_slice = _slices(ts)
    orchestrator.trigger_manual_attack(ts.id, AttackType.GHOST_VEHICLES, 0.4, duration_steps=2)
    out = orchestrator.corrupt_observation(ts, obs)
    # Density inflated by the Sybil attack
    assert np.all(out[density_slice] > obs[density_slice])
    # Queue untouched
    np.testing.assert_array_almost_equal(out[queue_slice], obs[queue_slice])
    assert np.all((out >= 0.0) & (out <= 1.0))


def test_position_jitter_touches_only_density(orchestrator):
    np.random.seed(0)
    ts = FakeTrafficSignal()
    obs = build_observation(ts, density=0.5, queue=0.5)
    density_slice, queue_slice = _slices(ts)
    orchestrator.trigger_manual_attack(ts.id, AttackType.POSITION_JITTER, 0.5, duration_steps=2)
    out = orchestrator.corrupt_observation(ts, obs)
    np.testing.assert_array_almost_equal(out[queue_slice], obs[queue_slice])
    assert np.all((out >= 0.0) & (out <= 1.0))


@pytest.mark.parametrize("atk", [AttackType.TEMPORAL_DOS, AttackType.FLOODING_DDOS, AttackType.SLOWLORIS_DDOS])
def test_latency_attacks_touch_only_latency(orchestrator, atk):
    np.random.seed(1)
    ts = FakeTrafficSignal()
    obs = build_observation(ts, density=0.5, queue=0.5, latency=0.1)
    density_slice, queue_slice = _slices(ts)
    orchestrator.trigger_manual_attack(ts.id, atk, 0.8, duration_steps=2)
    out = orchestrator.corrupt_observation(ts, obs)
    # density and queue blocks must be untouched
    np.testing.assert_array_almost_equal(out[density_slice], obs[density_slice])
    np.testing.assert_array_almost_equal(out[queue_slice], obs[queue_slice])
    # bounds respected
    assert np.all((out >= 0.0) & (out <= 1.0))


def test_jammer_cuts_comm_without_out_of_range_values(orchestrator):
    ts = FakeTrafficSignal()
    obs = build_observation(ts, latency=0.1, comm=1.0)
    orchestrator.trigger_manual_attack(ts.id, AttackType.JAMMER, 1.0, duration_steps=2)
    out = orchestrator.corrupt_observation(ts, obs)
    assert out[-1] == 0.0          # comm_flag cut
    assert out[-2] == 1.0          # latency maxed (NOT 99.0)
    assert ts.comm_ok is False
    assert np.all((out >= 0.0) & (out <= 1.0))


def test_all_attacks_respect_box_bounds_under_max_intensity(orchestrator):
    """Stress test: even at intensity 1.0, no attack may violate Box[0,1]."""
    np.random.seed(42)
    for atk in AttackType:
        if atk == AttackType.NONE:
            continue
        ts = FakeTrafficSignal(ts_id=f"ts_{atk.name}")
        obs = build_observation(ts, density=0.95, queue=0.95, latency=0.95, comm=1.0)
        orchestrator.trigger_manual_attack(ts.id, atk, 1.0, duration_steps=2)
        out = orchestrator.corrupt_observation(ts, obs)
        assert out.shape == obs.shape
        assert np.all((out >= 0.0) & (out <= 1.0)), f"{atk.name} violated Box[0,1]"


def test_telemetry_is_written(orchestrator):
    ts = FakeTrafficSignal()
    obs = build_observation(ts)
    orchestrator.trigger_manual_attack(ts.id, AttackType.GHOST_VEHICLES, 0.5, duration_steps=2)
    orchestrator.corrupt_observation(ts, obs)
    with open(orchestrator.log_path) as f:
        lines = f.readlines()
    # header + at least one telemetry row
    assert len(lines) >= 2
    assert "GHOST_VEHICLES" in lines[-1]


def test_attack_expires_and_restores_comm(orchestrator):
    ts = FakeTrafficSignal()
    obs = build_observation(ts)
    orchestrator.trigger_manual_attack(ts.id, AttackType.JAMMER, 1.0, duration_steps=1)
    orchestrator.corrupt_observation(ts, obs)  # consumes the single step
    # next call: remaining_steps hit 0 -> attack removed, comm restored
    out = orchestrator.corrupt_observation(ts, obs)
    assert ts.comm_ok is True
    assert ts.id not in orchestrator.active_attacks
    np.testing.assert_array_equal(out, obs)
