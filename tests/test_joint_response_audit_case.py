import importlib
import math
from pathlib import Path

import numpy as np
import pytest


@pytest.fixture
def m(monkeypatch):
    monkeypatch.syspath_prepend(
        str(Path(__file__).parents[1] / "examples/isolated_glu")
    )
    return importlib.import_module("joint_response_audit_case")


def fixture(m):
    from joint_response_join import JointJob
    from variable_response_view import ResponseBatch

    batches = []
    for step in range(2):
        for probe in range(4):
            x = np.array(
                [[0.2 + 0.1 * probe + 0.01 * step], [0.25 + 0.1 * probe + 0.01 * step]]
            )
            y = np.array(
                [
                    [1.2 * (0.5 * v / (1 + math.exp(-0.5 * v))) * (0.8 * v)]
                    for v in x[:, 0]
                ]
            )
            batches.append(ResponseBatch(x, y))
    job = JointJob("job00", 1, 1, 1, 8, iter(batches))
    data = m._collect_fixture(job, probes=4, rows=2)
    meta = data.metadata()
    params = {
        "g": np.array([[0.5 * data.sx]]),
        "u": np.array([[0.8 * data.sx]]),
        "d": np.array([[1.2 / data.sy]]),
    }
    checkpoints = {
        s: {k: v * (1 + index * 0.05) if k == "d" else v for k, v in params.items()}
        for index, s in enumerate(m.CHECKPOINTS)
    }
    records = []
    for step, p in checkpoints.items():

        def metric(indices):
            residual = []
            target = []
            for i in indices:
                z = float(data.x[i, 0] / data.sx)
                g = float(p["g"][0, 0])
                u = float(p["u"][0, 0])
                d = float(p["d"][0, 0])
                t = float(data.y[i, 0] / data.sy)
                residual.append(
                    (d * (g * z / (1 + math.exp(-g * z))) * (u * z) - t) ** 2
                )
                target.append(t * t)
            return sum(residual) / len(residual), math.sqrt(sum(residual) / sum(target))

        train, _ = metric(data.train)
        guard, rel = metric(data.guard)
        records.append(
            {
                "step": step,
                "train_half_mse": train / 2,
                "guard_mse": guard,
                "guard_relative_error": rel,
                "selected_at_checkpoint": step == 0,
                "selected_step_so_far": 0,
            }
        )
    physical = {"g": np.array([[0.5]]), "u": np.array([[0.8]]), "d": np.array([[1.2]])}
    predictions = [data.y[:8], data.y[8:]]
    return data, meta, records, checkpoints, physical, predictions


def run(m, f, **changes):
    data, meta, records, checkpoints, physical, predictions = f
    args = dict(
        data=data,
        metadata=meta,
        records=records,
        selected_step=0,
        checkpoint=lambda s: checkpoints[s],
        physical=physical,
        predictions=iter(predictions),
        prediction_rows=8,
        saved_split={"train": data.train, "guard": data.guard},
    )
    args.update(changes)
    return m.audit_case(**args)


def test_literal_all17_metrics_selection_physical_and_all_predictions(m):
    f = fixture(m)
    a = run(m, f)
    assert a["status"] == "agreed" and len(a["checkpoints"]) == 17
    assert a["selected_step_agreed"] and a["selection_certified"]
    assert a["physical_parameters_agreed"] and a["prediction_rows_checked"] == 16
    assert a["all_predictions_agreed"] and a["metrics_agreed"]
    assert a["certificate"] is False and a["private_gate"] == "closed"


def test_disagreement_preserved_and_remaining_checkpoints_checked(m):
    f = fixture(m)
    f[2][5]["guard_mse"] += 1
    rows = []
    a = run(m, f, emit=rows.append)
    assert (
        a["status"] == "disagreed" and len(a["checkpoints"]) == 17 and len(rows) == 17
    )
    assert not a["checkpoints"][5]["metrics_agreed"]


def test_schema_data_hash_split_rejected_before_checkpoint_read(m):
    f = fixture(m)
    meta = dict(f[1])
    meta["input_array_sha256"] = "0" * 64

    def forbidden(s):
        raise AssertionError("checkpoint before data binding")

    with pytest.raises(ValueError):
        run(m, f, metadata=meta, checkpoint=forbidden)
    split = {"train": f[0].guard, "guard": f[0].train}
    with pytest.raises(ValueError):
        run(m, f, saved_split=split, checkpoint=forbidden)
    records = list(f[2])
    records[2] = dict(records[2], step=True)
    with pytest.raises(ValueError):
        run(m, f, records=records, checkpoint=forbidden)


def test_scaling_and_physical_and_prediction_disagreement(m):
    f = fixture(m)
    f[1]["sx"] *= 2
    a = run(m, f)
    assert not a["normalization_agreed"] and a["status"] == "disagreed"
    f = fixture(m)
    f[4]["d"] *= 2
    assert not run(m, f)["physical_parameters_agreed"]
    f = fixture(m)
    f[5][1] = f[5][1] + 1
    a = run(m, f)
    assert not a["all_predictions_agreed"] and a["prediction_rows_checked"] == 16


def test_near_tie_is_ambiguous_and_does_not_replace_saved_model(m):
    f = fixture(m)
    f[3][256] = f[3][0]
    f[2][1] = dict(f[2][0], step=256, selected_at_checkpoint=False)
    a = run(m, f)
    assert a["selection_ambiguous"] and not a["selection_certified"]
    assert a["saved_selected_step"] == 0 and a["status"] == "ambiguous"


def test_numeric_failure_preserved_no_success_prefix(m):
    f = fixture(m)
    f[3][256] = dict(f[3][256], g=np.array([[1e308]]))
    a = run(m, f)
    assert a["status"] == "numerical_failure" and len(a["checkpoints"]) == 17
    assert a["checkpoints"][1]["status"] == "numerical_failure"
    assert not a["selection_certified"]


def test_prediction_count_shape_and_indices_native(m):
    f = fixture(m)
    with pytest.raises(ValueError):
        run(m, f, predictions=iter(f[5][:1]))
    with pytest.raises(ValueError):
        run(m, f, predictions=iter([*f[5], f[5][0]]))
    with pytest.raises(ValueError):
        run(m, f, saved_split={"train": f[0].train.astype(float), "guard": f[0].guard})


def test_assembly_exact_count_and_immutable_without_original_assembler(m):
    data, *_ = fixture(m)
    assert data.x.shape == (16, 1) and len(data.train) == 12 and len(data.guard) == 4
    with pytest.raises(ValueError):
        data.x.setflags(write=True)
    from joint_response_join import JointJob

    with pytest.raises(ValueError):
        m._collect_fixture(JointJob("job00", 1, 1, 1, 8, iter([])), probes=4, rows=2)


def test_near_tie_with_different_saved_winner_remains_uncertified(m):
    f = fixture(m)
    f[3][256] = f[3][0]
    f[2][0]["guard_mse"] = 1e-20
    f[2][1] = dict(
        f[2][0],
        step=256,
        guard_mse=0.0,
        selected_at_checkpoint=True,
        selected_step_so_far=256,
    )
    for row in f[2][2:]:
        row["selected_step_so_far"] = 256
    a = run(m, f, selected_step=256)
    assert a["status"] == "ambiguous" and not a["selected_step_agreed"]
    assert not a["selection_certified"] and a["saved_selected_step"] == 256


def test_failed_prediction_rows_are_visited_not_falsely_checked(m):
    f = fixture(m)
    f[4]["u"] = np.array([[1e308]])
    f[4]["d"] = np.array([[1e308]])
    a = run(m, f)
    assert a["status"] == "numerical_failure"
    assert a["prediction_rows_visited"] == 16 and a["prediction_rows_checked"] == 0
    assert a["prediction_numeric_failure"] and not a["selection_certified"]


def test_saved_physical_tolerance_cannot_hide_wrong_functional_export(m):
    from joint_response_join import JointJob
    from variable_response_view import ResponseBatch

    batches = [ResponseBatch(np.array([[1e20]]), np.ones((1, 1))) for _ in range(8)]
    data = m._collect_fixture(
        JointJob("job00", 1, 1, 1, 8, iter(batches)), probes=4, rows=1
    )
    checkpoint = {
        s: {
            "g": np.ones((1, 1)),
            "u": np.ones((1, 1)),
            "d": np.array([[1.0 if s == 0 else -1.0]]),
        }
        for s in m.CHECKPOINTS
    }
    rows = []
    for s in m.CHECKPOINTS:
        value = (1.0 if s == 0 else -1.0) / (1 + math.exp(-1.0))
        error = (value - 1) ** 2
        rows.append(
            dict(
                step=s,
                train_half_mse=error / 2,
                guard_mse=error,
                guard_relative_error=abs(value - 1),
                selected_at_checkpoint=s == 0,
                selected_step_so_far=0,
            )
        )
    a = m.audit_case(
        data=data,
        metadata=data.metadata(),
        records=rows,
        selected_step=0,
        checkpoint=lambda s: checkpoint[s],
        physical={"g": np.zeros((1, 1)), "u": np.zeros((1, 1)), "d": np.ones((1, 1))},
        predictions=iter([np.zeros((8, 1))]),
        saved_split={"train": data.train, "guard": data.guard},
        prediction_rows=8,
    )
    assert a[
        "physical_parameters_agreed"
    ]  # coefficient absolute tolerance alone is insufficient
    assert not a["all_predictions_agreed"] and a["status"] == "disagreed"
    assert not a["selection_certified"]


@pytest.mark.parametrize("invalid", [True, None, 1e-15])
def test_both_scale_fields_validated_before_any_checkpoint_even_if_first_disagrees(
    m, invalid
):
    f = fixture(m)
    f[1]["sx"] *= 2
    f[1]["sy"] = invalid

    def forbidden(s):
        raise AssertionError("checkpoint before both scales validated")

    with pytest.raises(ValueError):
        run(m, f, checkpoint=forbidden)
