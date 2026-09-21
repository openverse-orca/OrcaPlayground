"""The example controller maps names and time; it does not own physics."""

import math

import numpy as np
import pytest

from examples.euler.sensor_provider.dexhand_control import (
    APPROACH_TARGETS,
    FINGER_TARGETS,
    REQUIRED_ACTUATORS,
    DexHandGraspController,
)


class FakeModel:
    def __init__(self, *, reverse=False, extra=False):
        names = list(REQUIRED_ACTUATORS)
        if extra:
            names.append("unrelated_actuator")
        if reverse:
            names.reverse()
        self.actuators = {
            name: {"ActuatorId": index, "CtrlLimited": True, "CtrlRange": [-7.0, 7.0]}
            for index, name in enumerate(names)
        }
        self.nu = len(names)

    def get_actuator_dict(self):
        return self.actuators


def by_name(model, action):
    return {name: action[metadata["ActuatorId"]]
            for name, metadata in model.get_actuator_dict().items()}


def finger_target(name, scale=1.004):
    return FINGER_TARGETS[name] * (scale if name.startswith("act_joint4_") else 1.0)


def test_named_targets_and_timed_phases():
    model = FakeModel()
    controller = DexHandGraspController(model)
    assert controller.phase_at(0) == "settle"
    assert controller.phase_at(0.009) == "settle"
    np.testing.assert_array_equal(controller.action_at(0.009), np.zeros(model.nu))
    assert controller.phase_at(0.01) == "approach"
    approaching = by_name(model, controller.action_at(0.01))
    for name in REQUIRED_ACTUATORS:
        assert approaching[name] == pytest.approx(APPROACH_TARGETS.get(name, 0.0) / 990)
    assert controller.phase_at(1.0) == "close"
    closing = by_name(model, controller.action_at(1.0))
    assert closing["mot_joint1"] == 0.0
    assert closing["mot_joint2"] == -1.8508
    for name in FINGER_TARGETS:
        assert closing[name] == pytest.approx(finger_target(name) * 0.01)
    assert controller.phase_at(2.0) == "hold"
    held = by_name(model, controller.action_at(2.0))
    for name in FINGER_TARGETS:
        assert held[name] == finger_target(name)


@pytest.mark.parametrize("elapsed,alpha", [(0.01, 1 / 990), (0.504, 0.5),
                                          (0.998, 989 / 990), (0.999, 1),
                                          (0.9995, 1)])
def test_ramp_approach_interpolates_and_reaches_full_target_before_closing(elapsed, alpha):
    model = FakeModel()
    controls = by_name(model, DexHandGraspController(model).action_at(elapsed))
    for name in REQUIRED_ACTUATORS:
        assert controls[name] == pytest.approx(APPROACH_TARGETS.get(name, 0.0) * alpha)


def test_step_mode_preserves_explicit_full_approach_target():
    model = FakeModel()
    controller = DexHandGraspController(model, pregrasp_mode="step")
    np.testing.assert_array_equal(controller.action_at(0.009), np.zeros(model.nu))
    for elapsed in (0.01, 0.5, 0.999):
        controls = by_name(model, controller.action_at(elapsed))
        for name in REQUIRED_ACTUATORS:
            assert controls[name] == APPROACH_TARGETS.get(name, 0.0)
    ramp = DexHandGraspController(model)
    for elapsed in (1.0, 1.1, 2.0):
        np.testing.assert_array_equal(controller.action_at(elapsed), ramp.action_at(elapsed))


def test_smoothing_matches_one_percent_per_millisecond():
    model = FakeModel()
    controller = DexHandGraspController(model)
    for tick in (1, 20, 100, 300):
        targets = by_name(model, controller.action_at(1.0 + (tick - 1) * 0.001))
        for name in FINGER_TARGETS:
            assert targets[name] == pytest.approx(finger_target(name) * (1.0 - 0.99 ** tick))


@pytest.mark.parametrize("pregrasp_mode", ["ramp", "step"])
def test_time_based_controls_match_discrete_one_millisecond_sequence(pregrasp_mode):
    model = FakeModel(reverse=True, extra=True)
    controller = DexHandGraspController(model, pregrasp_mode=pregrasp_mode)
    expected = dict.fromkeys(model.get_actuator_dict(), 0.0)
    for frame in range(2200):
        if 10 <= frame < 1000:
            alpha = (frame - 10 + 1) / 990 if pregrasp_mode == "ramp" else 1.0
            for name, target in APPROACH_TARGETS.items():
                expected[name] = alpha * target
        if frame >= 1000:
            expected["mot_joint1"] = 0.0
            for name in FINGER_TARGETS:
                target = finger_target(name)
                remaining = target - expected[name]
                expected[name] = (target if abs(remaining) <= 0.001 else
                                  expected[name] + 0.01 * remaining)
        actual = by_name(model, controller.action_at(frame * 0.001))
        assert actual == pytest.approx(expected, abs=1e-12)
        if frame >= 1000:
            all_reached = all(expected[name] == finger_target(name)
                              for name in FINGER_TARGETS)
            assert controller.phase_at(frame * 0.001) == ("hold" if all_reached else "close")


@pytest.mark.parametrize("scale", [0.95, 1.0, 1.004, 1.05])
def test_ring_scale_only_changes_three_ring_finger_closing_targets(scale):
    model = FakeModel()
    baseline = DexHandGraspController(model, ring_target_scale=1.0)
    controller = DexHandGraspController(model, ring_target_scale=scale)
    for elapsed in (0.0, 0.01, 0.504, 0.999):
        np.testing.assert_array_equal(controller.action_at(elapsed), baseline.action_at(elapsed))
    for elapsed, factor in ((1.0, 0.01), (2.0, 1.0)):
        actual = by_name(model, controller.action_at(elapsed))
        expected = by_name(model, baseline.action_at(elapsed))
        for name in ("act_joint4_2", "act_joint4_3", "act_joint4_4"):
            expected[name] = finger_target(name, scale) * factor
        assert actual == pytest.approx(expected)


@pytest.mark.parametrize("mode", ["official", "", None, True, 1])
def test_invalid_pregrasp_mode_is_rejected(mode):
    with pytest.raises(ValueError, match="pregrasp_mode"):
        DexHandGraspController(FakeModel(), pregrasp_mode=mode)


@pytest.mark.parametrize("scale", [0.949, 1.051, float("nan"), float("inf"), True])
def test_invalid_ring_target_scale_is_rejected(scale):
    with pytest.raises(ValueError, match="ring_target_scale"):
        DexHandGraspController(FakeModel(), ring_target_scale=scale)


def test_scaled_ring_target_is_checked_against_control_range():
    model = FakeModel()
    model.actuators["act_joint4_3"]["CtrlRange"] = [0.0, FINGER_TARGETS["act_joint4_3"]]
    DexHandGraspController(model, ring_target_scale=1.0)
    with pytest.raises(ValueError, match="outside CtrlRange.*act_joint4_3"):
        DexHandGraspController(model)


def test_mapping_is_independent_of_xml_actuator_order_and_extra_actuators():
    first = FakeModel()
    second = FakeModel(reverse=True, extra=True)
    controllers = DexHandGraspController(first), DexHandGraspController(second)
    for elapsed in (0.0, 0.02, 1.2, 2.0):
        first_values = by_name(first, controllers[0].action_at(elapsed))
        second_values = by_name(second, controllers[1].action_at(elapsed))
        assert second_values.pop("unrelated_actuator") == 0.0
        assert first_values == second_values


def test_calls_are_time_based_repeatable_and_return_owned_arrays():
    controller = DexHandGraspController(FakeModel())
    expected = controller.action_at(1.12)
    assert expected.dtype == np.float64
    assert expected.flags.owndata
    for elapsed in (1.9, 0.0, 1.04, 100.0):
        controller.action_at(elapsed)[:] = math.nan
    np.testing.assert_array_equal(controller.action_at(1.12), expected)
    dense = {round(index * 0.001, 3): controller.action_at(index * 0.001)
             for index in range(1000, 1251)}
    for index in range(1000, 1251, 5):
        np.testing.assert_array_equal(controller.action_at(index * 0.001),
                                      dense[round(index * 0.001, 3)])


@pytest.mark.parametrize("elapsed", [-1.0, float("nan"), float("inf"), True])
def test_invalid_times(elapsed):
    controller = DexHandGraspController(FakeModel())
    with pytest.raises(ValueError, match="elapsed_seconds"):
        controller.action_at(elapsed)
    with pytest.raises(ValueError, match="elapsed_seconds"):
        controller.phase_at(elapsed)


def test_missing_required_name_is_not_silently_ignored():
    model = FakeModel()
    del model.actuators["act_joint5_4"]
    with pytest.raises(ValueError, match="missing actuators: act_joint5_4"):
        DexHandGraspController(model)


@pytest.mark.parametrize("index", [-1, 26, 1.5, True, 1])
def test_invalid_and_duplicate_indices_are_rejected(index):
    model = FakeModel()
    model.actuators["mot_joint1"]["ActuatorId"] = index
    with pytest.raises(ValueError, match="ActuatorId"):
        DexHandGraspController(model)


@pytest.mark.parametrize("limits", [[0.0, 1.0], [0.1, 2.0], [2.0, 1.0],
                                    [float("nan"), 2.0], [1.0]])
def test_bad_or_incompatible_ranges_are_rejected(limits):
    model = FakeModel()
    model.actuators["act_joint1_1"]["CtrlRange"] = limits
    with pytest.raises(ValueError, match="CtrlRange"):
        DexHandGraspController(model)


def test_unlimited_actuator_does_not_treat_zero_range_as_a_limit():
    model = FakeModel()
    model.actuators["act_joint1_1"].update(CtrlLimited=False, CtrlRange=[0.0, 0.0])
    assert np.isfinite(DexHandGraspController(model).action_at(2.0)).all()


def test_nonfinite_targets_are_rejected(monkeypatch):
    monkeypatch.setitem(APPROACH_TARGETS, "mot_joint1", float("nan"))
    with pytest.raises(ValueError, match="Non-finite grasp target"):
        DexHandGraspController(FakeModel())


@pytest.mark.parametrize("nu", [0, -1, 2.5, True, 27])
def test_invalid_or_incomplete_model_control_size(nu):
    model = FakeModel()
    model.nu = nu
    with pytest.raises(ValueError, match="model.nu|every model control index"):
        DexHandGraspController(model)
