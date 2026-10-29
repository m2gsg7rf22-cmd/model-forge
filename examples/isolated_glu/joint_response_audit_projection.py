"""Project saved candidate G/U/D only after all nine exact headers validate.

Moment payload bytes are bounded/CRC checked but never made into numeric views;
their values are outside this forward audit. Reuses frozen header grammar only,
not the fitted forward/loss/selection. No IO, RNG or experiment invocation.
"""

from __future__ import annotations

import io
from math import prod
import zipfile

import numpy as np
from numpy.typing import NDArray
from joint_response_arrays import _header, _schema
from joint_response_inventory import Field, Shard


def decode_checkpoint(
    blob: bytes, width: int, *, n: int = 32, outputs: int = 32
) -> dict[str, NDArray[np.float64]]:
    if (
        any(type(v) is not int for v in (width, n, outputs))
        or not 1 <= width <= 96
        or not 1 <= n <= 32
        or not 1 <= outputs <= 32
    ):
        raise ValueError("bounded native candidate dimensions required")
    params = (
        Field("g", "float64", (width, n)),
        Field("u", "float64", (width, n)),
        Field("d", "float64", (outputs, width)),
    )
    fields = tuple(
        Field(prefix + f.name, f.dtype, f.shape)
        for prefix in ("", "m_", "v_")
        for f in params
    )
    spec = Shard("checkpoint", fields)
    _schema(spec)
    if not isinstance(blob, bytes) or not 0 < len(blob) <= spec.encoded_bound:
        raise ValueError("bounded encoded checkpoint required")
    selected = []
    with zipfile.ZipFile(io.BytesIO(blob)) as archive:
        infos = archive.infolist()
        if len(infos) != 9 or {i.filename for i in infos} != {
            f.name + ".npy" for f in fields
        }:
            raise ValueError("exact nine unique checkpoint fields required")
        for field in fields:
            info = archive.getinfo(field.name + ".npy")
            if (
                info.compress_type != zipfile.ZIP_STORED
                or info.flag_bits & 1
                or info.compress_size != info.file_size
                or not 0 < info.file_size <= prod(field.shape) * 8 + 4096
            ):
                raise ValueError(
                    "stored unencrypted bounded checkpoint member required"
                )
            member = archive.read(info)
            offset, count = _header(member, field)
            if field.name in ("g", "u", "d"):
                selected.append((field, member, offset, count))
    # All nine header/payload boundaries validated before the first numeric view.
    result = {}
    for field, member, offset, count in selected:
        array = np.frombuffer(
            member, dtype=np.float64, count=count, offset=offset
        ).reshape(field.shape)
        if not np.isfinite(array).all():
            raise ValueError("nonfinite candidate parameter")
        result[field.name] = array
    return result
