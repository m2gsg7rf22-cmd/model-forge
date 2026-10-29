import importlib
import io
from pathlib import Path
import zipfile

import numpy as np
import pytest


@pytest.fixture
def m(monkeypatch):
    monkeypatch.syspath_prepend(
        str(Path(__file__).parents[1] / "examples/isolated_glu")
    )
    return importlib.import_module("joint_response_audit_projection")


def blob(**changes):
    values = {
        prefix + key: np.array([[1.0]])
        for prefix in ("", "m_", "v_")
        for key in ("g", "u", "d")
    }
    values.update(changes)
    out = io.BytesIO()
    np.savez(out, **values)
    return out.getvalue()


def test_projection_only_three_views_and_immutable(m, monkeypatch):
    payload = blob(v_d=np.array([[float("nan")]]))
    original = np.frombuffer
    calls = []

    def tracked(*args, **kwargs):
        calls.append(True)
        return original(*args, **kwargs)

    monkeypatch.setattr(np, "frombuffer", tracked)
    result = m.decode_checkpoint(payload, 1, n=1, outputs=1)
    assert set(result) == {"g", "u", "d"} and len(calls) == 3
    for a in result.values():
        assert a[0, 0] == 1
        with pytest.raises(ValueError):
            a.setflags(write=True)


@pytest.mark.parametrize("kind", ["shape", "dtype", "short", "extra", "compressed"])
def test_all_unused_headers_before_any_view(m, monkeypatch, kind):
    value = (
        np.ones((2, 1))
        if kind == "shape"
        else np.array([[1.0]], dtype=object)
        if kind == "dtype"
        else np.ones((1, 1))
    )
    payload = blob(v_d=value)
    if kind in ("short", "extra", "compressed"):
        out = io.BytesIO()
        with (
            zipfile.ZipFile(io.BytesIO(payload)) as old,
            zipfile.ZipFile(
                out,
                "w",
                compression=zipfile.ZIP_DEFLATED
                if kind == "compressed"
                else zipfile.ZIP_STORED,
            ) as new,
        ):
            for name in old.namelist():
                data = old.read(name)
                if name == "v_d.npy":
                    data = (
                        data[:-1]
                        if kind == "short"
                        else data + b"X"
                        if kind == "extra"
                        else data
                    )
                new.writestr(name, data)
        payload = out.getvalue()

    def forbidden(*args, **kwargs):
        raise AssertionError("array before all header validation")

    monkeypatch.setattr(np, "frombuffer", forbidden)
    with pytest.raises(ValueError):
        m.decode_checkpoint(payload, 1, n=1, outputs=1)


def test_nonfinite_required_parameter_and_bad_profile(m):
    with pytest.raises(ValueError):
        m.decode_checkpoint(blob(g=np.array([[float("inf")]])), 1, n=1, outputs=1)
    with pytest.raises(ValueError):
        m.decode_checkpoint(blob(), True, n=1, outputs=1)
    with pytest.raises(ValueError):
        m.decode_checkpoint(blob(), 1, n=0, outputs=1)
