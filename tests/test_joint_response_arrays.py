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
    return importlib.import_module("joint_response_arrays")


def spec():
    from joint_response_inventory import Field, Shard

    return Shard("fixture", (Field("a", "float64", (1, 2)),))


def archive(header, payload=b"", *, name="a.npy", compression=zipfile.ZIP_STORED):
    member = io.BytesIO()
    np.lib.format.write_array_header_1_0(member, header)
    out = io.BytesIO()
    with zipfile.ZipFile(out, "w", compression=compression) as z:
        z.writestr(name, member.getvalue() + payload)
    return out.getvalue()


def test_literal_float_and_integer_transport_immutable(m):
    from joint_response_inventory import Field, Shard

    expected = Shard(
        "fixture", (Field("a", "float64", (1, 2)), Field("indices", "int64", (2,)))
    )
    output = io.BytesIO()
    np.savez(output, a=np.array([[1.0, 2.0]]), indices=np.array([0, 1], dtype=np.int64))
    values = m.decode_typed(output.getvalue(), expected)
    np.testing.assert_array_equal(values["a"], [[1.0, 2.0]])
    assert values["indices"].dtype == np.int64
    with pytest.raises(ValueError):
        values["a"].setflags(write=True)


@pytest.mark.parametrize(
    "kind",
    [
        "shape",
        "dtype",
        "fortran",
        "short",
        "extra",
        "boolshape",
        "member",
        "compressed",
    ],
)
def test_invalid_schema_is_rejected_before_any_array_view(m, monkeypatch, kind):
    h = {"descr": "<f8", "fortran_order": False, "shape": (1, 2)}
    payload = bytes(16)
    kwargs = {}
    if kind == "shape":
        h["shape"] = (2, 2)
    if kind == "dtype":
        h["descr"] = "|O"
    if kind == "fortran":
        h["fortran_order"] = True
    if kind == "short":
        payload = bytes(8)
    if kind == "extra":
        payload = bytes(24)
    if kind == "boolshape":
        h["shape"] = (True, 2)
    if kind == "member":
        kwargs["name"] = "context.npy"
    if kind == "compressed":
        kwargs["compression"] = zipfile.ZIP_DEFLATED
    blob = archive(h, payload, **kwargs)

    def forbidden(*a, **k):
        raise AssertionError("array view created before schema validation")

    monkeypatch.setattr(m.np, "frombuffer", forbidden)
    with pytest.raises(ValueError):
        m.decode_typed(blob, spec())


def test_header_cap_and_nonfinite_payload_fail_closed(m):
    h = {"descr": "<f8", "fortran_order": False, "shape": (1, 2)}
    with pytest.raises(ValueError):
        m.decode_typed(archive(h, np.array([np.nan, 0.0]).tobytes()), spec())
    member = b"\x93NUMPY\x01\x00" + (4097).to_bytes(2, "little") + b"{}"
    out = io.BytesIO()
    with zipfile.ZipFile(out, "w") as z:
        z.writestr("a.npy", member)
    with pytest.raises(ValueError):
        m.decode_typed(out.getvalue(), spec())
