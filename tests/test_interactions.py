from unittest.mock import MagicMock

import pytest

from bonfire_palm import BonfirePalmController


class FakeHand:
    def __init__(self, detected=True, is_open_palm=True, x=0.25, y=-0.15):
        self.detected = detected
        self.is_open_palm = is_open_palm
        self.palm_ndc_x = x
        self.palm_ndc_y = y


class TestBonfirePalmController:
    def test_open_palm_charges_state(self):
        ctrl = BonfirePalmController()
        state = ctrl.update(FakeHand(), 0.5)

        assert state.detected is True
        assert state.is_open is True
        assert state.charge > 0.0
        assert state.intensity > 0.0
        assert state.kindle_radius > ctrl.MIN_RADIUS
        assert state.phase == "Kindling"

    def test_open_palm_applies_particle_effects(self):
        ctrl = BonfirePalmController()
        particles = MagicMock()

        state = ctrl.update(FakeHand(x=0.1, y=0.2), 0.5, particles=particles)

        particles.kindle_nearby.assert_called_once()
        particles.spawn_palm_sparks.assert_called_once()
        _, _, kwargs = particles.spawn_palm_sparks.mock_calls[0]
        assert kwargs["count"] > 10
        assert kwargs["spread"] > 0.0
        assert kwargs["intensity"] == pytest.approx(0.65 + state.intensity * 0.9)

    def test_just_ignited_only_on_open_edge(self):
        ctrl = BonfirePalmController()

        first_ignited = ctrl.update(FakeHand(is_open_palm=True), 0.1).just_ignited
        second_ignited = ctrl.update(FakeHand(is_open_palm=True), 0.1).just_ignited

        assert first_ignited is True
        assert second_ignited is False

    def test_closed_hand_decays_charge(self):
        ctrl = BonfirePalmController()
        ctrl.update(FakeHand(is_open_palm=True), 0.5)
        charged = ctrl.state.charge

        state = ctrl.update(FakeHand(is_open_palm=False), 0.25)

        assert state.detected is True
        assert state.is_open is False
        assert state.charge < charged
        assert state.phase == "Open Palm Needed"

    def test_disabled_decays_and_skips_particles(self):
        ctrl = BonfirePalmController()
        particles = MagicMock()
        ctrl.update(FakeHand(is_open_palm=True), 0.5)
        ctrl.enabled = False

        state = ctrl.update(FakeHand(is_open_palm=True), 0.2, particles=particles)

        assert state.is_open is False
        assert state.phase == "Disabled"
        particles.kindle_nearby.assert_not_called()
        particles.spawn_palm_sparks.assert_not_called()
