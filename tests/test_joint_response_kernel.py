import importlib
import math
from pathlib import Path

import numpy as np
import pytest


@pytest.fixture
def k(monkeypatch):
    monkeypatch.syspath_prepend(
        str(Path(__file__).parents[1] / "examples/isolated_glu")
    )
    return importlib.import_module("joint_response_kernel")


def literal(k):
    return k.Parameters(
        np.array([[0.3, -0.2], [0.1, 0.4]]),
        np.array([[0.7, 0.2], [-0.5, 0.6]]),
        np.array([[0.8, -0.3], [0.2, 0.9]]),
    )


def scalar(p, x):
    return np.array(
        [
            [
                sum(
                    (
                        sum(row[j] * p.g[h, j] for j in range(2))
                        / (1 + math.exp(-sum(row[j] * p.g[h, j] for j in range(2))))
                    )
                    * sum(row[j] * p.u[h, j] for j in range(2))
                    * p.d[o, h]
                    for h in range(2)
                )
                for o in range(2)
            ]
            for row in x
        ]
    )


def test_forward_loss_and_all_analytic_derivatives(k):
    p = literal(k)
    x = np.array([[0.1, -0.7], [0.5, 0.3], [-0.2, 0.4]])
    y = np.array([[0.2, 0.1], [-0.1, 0.3], [0.4, -0.2]])
    np.testing.assert_allclose(k.predict(p, x), scalar(p, x), rtol=1e-14, atol=1e-16)
    loss, grad = k.loss_gradient(p, x, y)
    assert loss == pytest.approx(0.5 * np.mean((scalar(p, x) - y) ** 2))
    for name in ("g", "u", "d"):
        for index in np.ndindex(getattr(p, name).shape):
            values = {n: getattr(p, n).copy() for n in ("g", "u", "d")}
            values[name][index] += 1e-6
            plus = 0.5 * np.mean((scalar(k.Parameters(**values), x) - y) ** 2)
            values[name][index] -= 2e-6
            minus = 0.5 * np.mean((scalar(k.Parameters(**values), x) - y) ** 2)
            assert getattr(grad, name)[index] == pytest.approx(
                (plus - minus) / 2e-6, abs=1e-10
            )


def test_parameters_are_owned_immutable_and_strict(k):
    p = literal(k)
    for a in (p.g, p.u, p.d):
        with pytest.raises(ValueError):
            a.setflags(write=True)
    g = np.array([[1.0]])
    q = k.Parameters(g, g, g)
    g[0, 0] = 7
    assert q.g[0, 0] == 1
    for bad in (
        np.ones((1, 1), dtype=np.float32),
        np.array([[np.nan]]),
        np.ones((97, 1)),
    ):
        with pytest.raises(ValueError):
            k.Parameters(bad, bad, bad)
    with pytest.raises(ValueError):
        k.Parameters(np.ones((2, 1)), np.ones((1, 1)), np.ones((1, 2)))


def test_train_only_normalization_and_physical_folding(k):
    x = np.array([[2.0, 4.0], [-1.0, 3.0]])
    y = np.array([[0.1, -0.4], [1.0, 2.0]])
    sx, sy = k.training_scales(x, y)
    assert sx == pytest.approx(np.sqrt(np.mean(x * x)))
    assert sy == pytest.approx(np.sqrt(np.mean(y * y)))
    p = literal(k)
    np.testing.assert_allclose(
        k.predict(k.physical_parameters(p, sx, sy), x),
        sy * k.predict(p, x / sx),
        rtol=1e-14,
        atol=1e-14,
    )
    for bad in (np.zeros((2, 2)), np.full((2, 2), 1e-14)):
        with pytest.raises(ValueError):
            k.training_scales(bad, y)
    huge = np.full((2, 2), 1e200)
    assert k.training_scales(huge, y)[0] == pytest.approx(1e200)
    with pytest.raises(ValueError):
        k.physical_parameters(p, 0.0, sy)


def test_stable_silu_and_fail_closed_numeric_schema(k):
    p = k.Parameters(np.ones((1, 1)), np.ones((1, 1)), np.ones((1, 1)))
    x = np.array([[-1000.0], [1000.0]])
    np.testing.assert_allclose(k.predict(p, x), [[0.0], [1000000.0]])
    with pytest.raises(ValueError):
        k.predict(p, np.zeros((0, 1)))
    with pytest.raises(ValueError):
        k.loss_gradient(p, x, np.zeros((1, 1)))
    huge = k.Parameters(np.full((1, 1), 1e200), np.ones((1, 1)), np.ones((1, 1)))
    with pytest.raises(FloatingPointError):
        k.predict(huge, np.full((1, 1), 1e200))


def test_fixed_adam_first_update_and_callback_failure(k):
    p = literal(k)
    x = np.array([[0.2, -0.3]])
    y = np.array([[0.4, -0.1]])
    _, grad = k.loss_gradient(p, x, y)
    seen = []

    class Stop(Exception):
        pass

    def observer(event):
        seen.append(event)
        if isinstance(event, k.Step) and event.step == 1:
            raise Stop

    with pytest.raises(Stop):
        k.fit(p, x, y, x, y, np.zeros((4000, 256), dtype=np.int64), observer)
    assert seen[0].step == 0 and seen[-1].step == 1
    # A step event binds post-update parameters but pre-update sampled loss.
    for name in ("g", "u", "d"):
        g = getattr(grad, name)
        expected = getattr(p, name) - 0.01 * g / (np.abs(g) + 1e-8)
        np.testing.assert_allclose(
            getattr(seen[-1].parameters, name), expected, rtol=1e-14, atol=1e-16
        )
    assert k.learning_rate(0) == pytest.approx(0.01)
    assert k.learning_rate(3999) == pytest.approx(0.0001)
    for invalid in (-1, 4000, True):
        with pytest.raises(ValueError):
            k.learning_rate(invalid)


def test_complete_schedule_checkpoint_ties_zero_reference_and_immutability(k):
    zero = np.zeros((1, 1))
    p = k.Parameters(zero, zero, zero)
    x = np.ones((1, 1))
    schedule = np.zeros((4000, 256), dtype=np.int64)
    events = []
    result = k.fit(p, x, zero, x, zero, schedule, events.append)
    assert result.updates_completed == 4000 and result.selected_step == 0
    assert [c.step for c in result.checkpoints] == [0, *range(256, 4000, 256), 4000]
    assert len(result.checkpoints) == 17
    assert all(
        c.guard_relative_error is None and c.guard_mse == 0 for c in result.checkpoints
    )
    assert len([e for e in events if isinstance(e, k.Step)]) == 4000
    assert events[-1].step == 4000
    assert not np.any(schedule) and not np.any(p.g)
    for c in result.checkpoints:
        for a in (c.parameters.g, c.first_moment.u, c.second_moment.d):
            with pytest.raises(ValueError):
                a.setflags(write=True)


def test_guard_does_not_change_training_and_invalid_schedule_fails_before_observer(k):
    p = literal(k)
    x = np.array([[0.2, 0.1], [-0.1, 0.4]])
    y = np.array([[0.3, -0.2], [0.4, 0.1]])
    schedule = np.tile(np.array([0, 1] * 128, dtype=np.int64), (4000, 1))

    class Stop(Exception):
        pass

    def first(guard):
        seen = []

        def observer(e):
            if isinstance(e, k.Step):
                seen.append(e)
                if e.step == 2:
                    raise Stop

        with pytest.raises(Stop):
            k.fit(p, x, y, x, guard, schedule, observer)
        return seen

    a, b = first(y), first(y * 100)
    for left, right in zip(a, b):
        for name in ("g", "u", "d"):
            np.testing.assert_array_equal(
                getattr(left.parameters, name), getattr(right.parameters, name)
            )
    for bad in (schedule[:2], schedule.astype(np.int32), schedule + 2):
        events = []
        with pytest.raises(ValueError):
            k.fit(p, x, y, x, y, bad, events.append)
        assert not events


def test_numeric_failure_is_not_successful_prefix(k):
    p = k.Parameters(np.ones((1, 1)), np.ones((1, 1)), np.ones((1, 1)))
    x = np.full((1, 1), 1e200)
    with pytest.raises(k.NumericalFailure) as error:
        k.fit(
            p,
            x,
            np.zeros((1, 1)),
            x,
            np.zeros((1, 1)),
            np.zeros((4000, 256), dtype=np.int64),
            lambda e: None,
        )
    assert error.value.updates_completed == 0


def test_nonconstant_schedule_rows_are_consumed_and_guard_selects_finite_minimum(k):
    p = k.Parameters(np.array([[0.4]]), np.array([[0.3]]), np.array([[0.7]]))
    x = np.array([[0.2], [0.8]])
    y = np.array([[0.3], [-0.1]])
    indices = np.tile(np.arange(4000, dtype=np.int64)[:, None] % 2, (1, 256))
    events = []
    result = k.fit(p, x, y, x, y, indices, events.append)
    steps = [e for e in events if isinstance(e, k.Step)]
    previous = p
    for e in steps:
        row = (e.step - 1) % 2
        expected = k.loss_gradient(previous, x[row : row + 1], y[row : row + 1])[0]
        assert e.sampled_loss_before_update == pytest.approx(expected, abs=1e-15)
        previous = e.parameters
    minimum = min(result.checkpoints, key=lambda c: c.guard_mse)
    assert result.selected_step == minimum.step
    assert result.selected is minimum.parameters
    assert result.final is steps[-1].parameters
    for c in result.checkpoints:
        residual = k.predict(c.parameters, x) - y
        assert c.guard_mse == pytest.approx(np.mean(residual * residual))
        assert c.guard_relative_error == pytest.approx(
            np.linalg.norm(residual) / np.linalg.norm(y)
        )


def test_callback_cannot_mutate_later_training_inputs_or_schedule(k):
    p = k.Parameters(np.array([[0.4]]), np.array([[0.3]]), np.array([[0.7]]))
    x = np.array([[0.2]])
    y = np.array([[0.3]])
    schedule = np.zeros((4000, 256), dtype=np.int64)

    class Stop(Exception):
        pass

    def observer(e):
        if e.step == 0:
            x[:] = 1e200
            y[:] = 1e200
            schedule[:] = 10
        elif isinstance(e, k.Step):
            assert math.isfinite(e.sampled_loss_before_update)
            raise Stop

    with pytest.raises(Stop):
        k.fit(p, x, y, x, y, schedule, observer)


def test_guard_normalized_mse_uses_actual_zero_reference_only(k):
    p = k.Parameters(np.ones((1, 1)), np.ones((1, 1)), np.ones((1, 1)))

    class Stop(Exception):
        pass

    def observer(e):
        if e.step == 0:
            assert e.guard_relative_error == pytest.approx(
                (1 / (1 + math.exp(-1))) * 1e200
            )
            raise Stop

    with pytest.raises(Stop):
        k.fit(
            p,
            np.ones((1, 1)),
            np.zeros((1, 1)),
            np.ones((1, 1)),
            np.full((1, 1), 1e-200),
            np.zeros((4000, 256), dtype=np.int64),
            observer,
        )
    zero = k.Parameters(np.zeros((1, 1)), np.zeros((1, 1)), np.zeros((1, 1)))
    with pytest.raises(k.NumericalFailure):
        k.fit(
            zero,
            np.ones((1, 1)),
            np.zeros((1, 1)),
            np.ones((1, 1)),
            np.full((1, 1), 1e-200),
            np.zeros((4000, 256), dtype=np.int64),
            lambda e: None,
        )


def test_nonzero_loss_cannot_silently_underflow_to_zero(k):
    z = np.zeros((1, 1))
    p = k.Parameters(z, z, z)
    with pytest.raises(FloatingPointError):
        k.loss_gradient(p, z, np.full((1, 1), 1e-200))
    target = np.zeros((8, 1))
    target[0, 0] = np.nextafter(0.0, 1.0)
    with pytest.raises(k.NumericalFailure):
        k.fit(
            p,
            np.zeros((8, 1)),
            target,
            np.zeros((8, 1)),
            target,
            np.zeros((4000, 256), dtype=np.int64),
            lambda e: None,
        )


def test_subnormal_mean_and_relative_ratio_underflow_fail_closed(k):
    z = np.zeros((1, 1))
    p = k.Parameters(z, z, z)
    target = np.zeros((8, 1))
    target[0, 0] = np.nextafter(0.0, 1.0)
    with pytest.raises(k.NumericalFailure):
        k.fit(
            p,
            np.zeros((8, 1)),
            target,
            np.zeros((8, 1)),
            target,
            np.zeros((4000, 256), dtype=np.int64),
            lambda e: None,
        )
    p = k.Parameters(np.ones((1, 1)), np.ones((1, 1)), np.ones((1, 1)))
    gx = np.array([[1e154], [0.0]])
    gy = k.predict(p, gx)
    gy[1, 0] = 1e-100
    with pytest.raises(k.NumericalFailure):
        k.fit(p, z, z, gx, gy, np.zeros((4000, 256), dtype=np.int64), lambda e: None)
