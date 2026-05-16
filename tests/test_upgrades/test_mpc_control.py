"""Tests for MPCController."""

import pytest
import numpy as np

casadi = pytest.importorskip("casadi")

from tt_sim.interfaces import HighLevelController
from tt_sim.registry import load
from tt_sim.upgrades.control import MPCController

# Reuse Dummy implementations from test_control.py
from tests.test_upgrades.test_control import (
    DummyPerceiver, DummyPredictor, DummySpinEstimator,
    DummyAimer, DummySwingPlanner,
)


def test_import():
    assert MPCController is not None


def test_subclass():
    assert issubclass(MPCController, HighLevelController)


def test_registry():
    cls = load("control", "mpc")
    assert cls is MPCController


def test_holds_without_fk():
    """Without FK dict, controller holds position."""
    ctrl = MPCController(
        DummyPerceiver(), DummyPredictor(), DummySpinEstimator(),
        DummyAimer(), DummySwingPlanner(), fk_dict=None,
    )
    q = np.array([0.5, 0.5])
    for _ in range(10):
        result = ctrl.step({}, q)
        np.testing.assert_array_equal(result, q)


def test_reset_clears_state():
    ctrl = MPCController(
        DummyPerceiver(), DummyPredictor(), DummySpinEstimator(),
        DummyAimer(), DummySwingPlanner(), fk_dict=None,
    )
    q = np.array([0.5, 0.5])
    for _ in range(10):
        ctrl.step({}, q)
    ctrl.reset()
    assert ctrl._observations == []
    assert ctrl._frame_count == 0
    assert ctrl._prev_solution is None
