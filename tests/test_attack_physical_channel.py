"""Unit tests for the PHYSICAL attack channel (Lot E).

SUMO-free: a FakeSumo stand-in records TraCI calls so we can prove that the
orchestrator produces REAL, visible side effects (red ghost vehicles, phase
freeze) and not just perceptual observation poisoning.
"""

import os
import tempfile

import pytest

from sumo_rl.environment.attack_controller import (
    AttackType,
    CyberPhysicalAttackOrchestrator,
    GHOST_VEHICLE_COLOR,
    GHOST_VEHICLE_TYPE_ID,
)


class _Domain:
    """Generic recording domain (vehicle / vehicletype / route / trafficlight)."""

    def __init__(self, ids=None):
        self._ids = list(ids or [])
        self.calls = []

    def getIDList(self):
        return list(self._ids)

    def __getattr__(self, name):
        def _recorder(*args, **kwargs):
            self.calls.append((name, args, kwargs))
            # Simulate state mutations for a couple of methods.
            if name in ("add", "copy"):
                new_id = args[1] if name == "copy" else args[0]
                if new_id not in self._ids:
                    self._ids.append(new_id)
            if name == "getRedYellowGreenState":
                return "GrGr"
            return None
        return _recorder


class FakeSumo:
    """Minimal stand-in for a TraCI connection."""

    def __init__(self):
        self.label = "test_conn"
        self.vehicle = _Domain()
        self.vehicletype = _Domain(ids=["DEFAULT_VEHTYPE"])
        self.route = _Domain()
        self.trafficlight = _Domain(ids=["A0"])


class FakeTrafficSignal:
    def __init__(self, ts_id="A0", n_lanes=3):
        self.id = ts_id
        self.lanes = [f"edge{ts_id}{i}_0" for i in range(n_lanes)]
        self.comm_ok = True


@pytest.fixture
def orchestrator():
    tmp = tempfile.NamedTemporaryFile(suffix=".csv", delete=False)
    tmp.close()
    orch = CyberPhysicalAttackOrchestrator(log_path=tmp.name)
    yield orch
    os.unlink(tmp.name)


def _call_names(domain):
    return [c[0] for c in domain.calls]


def test_ghost_vehicles_spawn_real_red_vehicles(orchestrator):
    ts = FakeTrafficSignal()
    sumo = FakeSumo()
    orchestrator.trigger_manual_attack(ts.id, AttackType.GHOST_VEHICLES, 1.0)
    effects = orchestrator.apply_physical_attack(ts, sumo)

    assert effects["ghosts_spawned"] > 0
    # Real vehicles were added to SUMO.
    assert "add" in _call_names(sumo.vehicle)
    # They were painted RED.
    color_calls = [c for c in sumo.vehicle.calls if c[0] == "setColor"]
    assert color_calls, "ghost vehicles must be colored"
    assert all(c[1][1] == GHOST_VEHICLE_COLOR for c in color_calls)
    # A dedicated red vehicle type was registered.
    assert GHOST_VEHICLE_TYPE_ID in sumo.vehicletype.getIDList()


def test_jammer_freezes_phase(orchestrator):
    ts = FakeTrafficSignal()
    sumo = FakeSumo()
    orchestrator.trigger_manual_attack(ts.id, AttackType.JAMMER, 1.0)
    effects = orchestrator.apply_physical_attack(ts, sumo)

    assert effects["phase_frozen"] is True
    assert "setRedYellowGreenState" in _call_names(sumo.trafficlight)
    # No ghost vehicles for a jammer.
    assert effects["ghosts_spawned"] == 0


def test_data_poisoning_has_no_physical_effect(orchestrator):
    ts = FakeTrafficSignal()
    sumo = FakeSumo()
    orchestrator.trigger_manual_attack(ts.id, AttackType.DATA_POISONING, 1.0)
    effects = orchestrator.apply_physical_attack(ts, sumo)

    assert effects["ghosts_spawned"] == 0
    assert effects["phase_frozen"] is False
    assert "add" not in _call_names(sumo.vehicle)


def test_no_physical_effect_without_active_attack(orchestrator):
    ts = FakeTrafficSignal()
    sumo = FakeSumo()
    effects = orchestrator.apply_physical_attack(ts, sumo)
    assert effects == {"ghosts_spawned": 0, "phase_frozen": False}


def test_rearming_extends_duration_without_duplicate_log(orchestrator):
    ts = FakeTrafficSignal()
    orchestrator.trigger_manual_attack(ts.id, AttackType.JAMMER, 0.5, duration_steps=3)
    first = orchestrator.active_attacks[ts.id]["remaining_steps"]
    # Re-arm same family: should refresh, not create a second entry.
    orchestrator.trigger_manual_attack(ts.id, AttackType.JAMMER, 0.9, duration_steps=5)
    atk = orchestrator.active_attacks[ts.id]
    assert atk["intensity"] == pytest.approx(0.9)
    assert atk["remaining_steps"] >= first
    assert len(orchestrator.active_attacks) == 1


def test_ghost_intensity_scales_spawn_count(orchestrator):
    ts = FakeTrafficSignal()
    sumo_low = FakeSumo()
    sumo_high = FakeSumo()
    orchestrator.trigger_manual_attack(ts.id, AttackType.GHOST_VEHICLES, 0.2)
    low = orchestrator.apply_physical_attack(ts, sumo_low)["ghosts_spawned"]
    orchestrator.active_attacks.clear()
    orchestrator.trigger_manual_attack(ts.id, AttackType.GHOST_VEHICLES, 1.0)
    high = orchestrator.apply_physical_attack(ts, sumo_high)["ghosts_spawned"]
    assert high > low
