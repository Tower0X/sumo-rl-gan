"""Unit tests for the two-phase Fail-Safe VANET reward (thesis Eq. 3.4 / 3.5).

These tests are SUMO-free: the TraCI calls used by the reward
(``get_pressure``, ``get_total_queued``, ``get_collisions``, vehicle
acceleration) are mocked so the reward logic can be validated in isolation.

The central claim of the thesis is that the reward SWITCHES on ``comm_ok``.
We prove exactly that here.
"""

import types

import pytest

from sumo_rl.environment.traffic_signal import TrafficSignal


def make_reward_signal(
    *,
    comm_ok=True,
    pressure=0.0,
    total_queue=0,
    collisions=0,
    emergency_vehicles=0,
    n_lanes=4,
    params=None,
):
    """Build a minimal object exposing exactly what _vanet_reward needs.

    We do NOT instantiate a full TrafficSignal (that requires SUMO). Instead we
    create a bare object and bind the real ``_vanet_reward`` method to it, then
    stub every dependency it calls.
    """
    ts = types.SimpleNamespace()
    ts.lanes = [f"lane_{i}" for i in range(n_lanes)]
    ts.comm_ok = comm_ok
    ts.vanet_reward_params = params or {
        "alpha": 0.4,
        "beta": 0.6,
        "kappa_coll": 10.0,
        "kappa_quad": 0.25,
        "q_safe": 5.0,
        "emergency_penalty": 5.0,
        "emergency_decel_threshold": -4.5,
    }

    ts.get_pressure = lambda: pressure
    ts.get_total_queued = lambda: total_queue
    ts.get_collisions = lambda: collisions

    veh_ids = [f"veh_{i}" for i in range(emergency_vehicles)]
    ts._get_veh_list = lambda: veh_ids

    # All listed vehicles are hard-braking (accel below threshold).
    sumo = types.SimpleNamespace()
    sumo.vehicle = types.SimpleNamespace(getAcceleration=lambda v: -9.0)
    ts.sumo = sumo

    # Bind the real reward method.
    ts._vanet_reward = types.MethodType(TrafficSignal._vanet_reward, ts)
    return ts


def test_normal_mode_has_no_quadratic_term():
    # comm_ok=1, large queue: in normal mode the queue must NOT be penalised
    # quadratically, only linearly via beta.
    ts = make_reward_signal(comm_ok=True, total_queue=20, n_lanes=4)
    r = ts._vanet_reward()
    # Expected: -(0.4*0 + 0.6*(20/4)) = -3.0  (no collisions, no emergency)
    assert r == pytest.approx(-(0.6 * (20 / 4)))


def test_degraded_mode_is_strictly_worse_than_normal():
    """Same physical state, only comm_ok differs -> degraded must be lower."""
    normal = make_reward_signal(comm_ok=True, total_queue=20, n_lanes=4)._vanet_reward()
    degraded = make_reward_signal(comm_ok=False, total_queue=20, n_lanes=4)._vanet_reward()
    assert degraded < normal
    # The exact gap is the quadratic penalty -kappa_quad * S^2, S = 20 - 5 = 15
    expected_gap = 0.25 * (15 ** 2)
    assert (normal - degraded) == pytest.approx(expected_gap)


def test_quadratic_term_is_zero_below_q_safe():
    """Continuity at q_safe: queue <= q_safe -> no quadratic penalty even degraded."""
    normal = make_reward_signal(comm_ok=True, total_queue=4, n_lanes=4)._vanet_reward()
    degraded = make_reward_signal(comm_ok=False, total_queue=4, n_lanes=4)._vanet_reward()
    assert degraded == pytest.approx(normal)


def test_collisions_lower_the_reward():
    no_coll = make_reward_signal(comm_ok=True, total_queue=4, collisions=0)._vanet_reward()
    with_coll = make_reward_signal(comm_ok=True, total_queue=4, collisions=2)._vanet_reward()
    assert (no_coll - with_coll) == pytest.approx(10.0 * 2)


def test_emergency_braking_lowers_the_reward():
    calm = make_reward_signal(comm_ok=True, total_queue=4, emergency_vehicles=0)._vanet_reward()
    panic = make_reward_signal(comm_ok=True, total_queue=4, emergency_vehicles=3)._vanet_reward()
    assert (calm - panic) == pytest.approx(5.0 * 3)


def test_pressure_is_absolute_and_normalised():
    """Pressure uses |value|/n_lanes so the sign of the imbalance never helps."""
    pos = make_reward_signal(comm_ok=True, pressure=8.0, n_lanes=4)._vanet_reward()
    neg = make_reward_signal(comm_ok=True, pressure=-8.0, n_lanes=4)._vanet_reward()
    assert pos == pytest.approx(neg)
    # -(0.4 * 8/4) = -0.8
    assert pos == pytest.approx(-(0.4 * 8 / 4))


def test_coefficients_are_overridable():
    custom = {
        "alpha": 1.0, "beta": 1.0, "kappa_coll": 1.0, "kappa_quad": 1.0,
        "q_safe": 0.0, "emergency_penalty": 1.0, "emergency_decel_threshold": -4.5,
    }
    ts = make_reward_signal(comm_ok=False, total_queue=3, n_lanes=1, params=custom)
    r = ts._vanet_reward()
    # base = -(1*0 + 1*3) = -3 ; S = 3 ; quad = -1*9 ; total = -12
    assert r == pytest.approx(-12.0)
