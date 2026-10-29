"""Bounded typed ZIP_STORED/NPY decoder: validate headers BEFORE array views.

Only native finite float64/int64, exact caller schemas and C-order are accepted.
No np.load/read_array or object/structured dtype interpretation is used.
"""

from __future__ import annotations
import ast
import io
from math import prod
from pathlib import Path
import re
from typing import Any
import zipfile

import numpy as np
from numpy.typing import NDArray
from joint_response_inventory import Field, Shard
from variable_response_manifest import _read


def _schema(spec: Shard) -> None:
    if (
        not isinstance(spec, Shard)
        or not 1 <= len(spec.fields) <= 16
        or len({f.name for f in spec.fields}) != len(spec.fields)
    ):
        raise ValueError("bounded unique field schema required")
    for field in spec.fields:
        if (
            not isinstance(field, Field)
            or re.fullmatch("[A-Za-z][A-Za-z0-9_]{0,63}", field.name) is None
            or field.dtype not in ("float64", "int64")
            or not isinstance(field.shape, tuple)
            or not 1 <= len(field.shape) <= 2
            or any(type(d) is not int or not 1 <= d <= 71808 for d in field.shape)
        ):
            raise ValueError("exact bounded native numeric schema required")
    if spec.encoded_bound > 8 * 1024**2:
        raise ValueError("typed archive exceeds8MiBcap")


def _header(member: bytes, field: Field) -> tuple[int, int]:
    if len(member) < 10 or member[:6] != b"\x93NUMPY":
        raise ValueError("NPY magic required")
    version = member[6:8]
    if version == b"\x01\x00":
        size_bytes = 2
    elif version == b"\x02\x00":
        size_bytes = 4
    else:
        raise ValueError("NPY1/2 headers only")
    start = 8 + size_bytes
    length = int.from_bytes(member[8:start], "little")
    end = start + length
    if not 0 < length <= 4096 or end > len(member):
        raise ValueError("NPY header cap/truncation")
    try:
        text = member[start:end].decode("ascii").strip()
        tree = ast.parse(text, mode="eval")
        node = tree.body
        if (
            not isinstance(node, ast.Dict)
            or len(node.keys) != 3
            or any(
                not isinstance(k, ast.Constant) or not isinstance(k.value, str)
                for k in node.keys
            )
        ):
            raise ValueError("exact NPY header dictionary required")
        keys = [k.value for k in node.keys if isinstance(k, ast.Constant)]
        if set(keys) != {"descr", "fortran_order", "shape"} or len(set(keys)) != 3:
            raise ValueError("duplicate/unknown NPY header key")
        value = ast.literal_eval(node)
    except (SyntaxError, UnicodeError, TypeError, RecursionError) as exc:
        raise ValueError("invalid bounded literal NPY header") from exc
    if (
        not isinstance(value, dict)
        or value["descr"] != np.dtype(field.dtype).str
        or value["fortran_order"] is not False
        or not isinstance(value["shape"], tuple)
        or any(type(d) is not int for d in value["shape"])
        or value["shape"] != field.shape
    ):
        raise ValueError("NPY declared dtype/shape/order differs from exact schema")
    count = prod(field.shape)
    if len(member) != end + count * 8:
        raise ValueError("NPY exact data payload length required")
    return end, count


def decode_typed(blob: bytes, spec: Shard) -> dict[str, NDArray[Any]]:
    _schema(spec)
    if not isinstance(blob, bytes) or not 0 < len(blob) <= spec.encoded_bound:
        raise ValueError("bounded encoded typed archive required")
    validated = []
    with zipfile.ZipFile(io.BytesIO(blob)) as archive:
        infos = archive.infolist()
        if len(infos) != len(spec.fields) or {i.filename for i in infos} != {
            f.name + ".npy" for f in spec.fields
        }:
            raise ValueError("exact unique typed archive members required")
        for field in spec.fields:
            info = archive.getinfo(field.name + ".npy")
            if (
                info.compress_type != zipfile.ZIP_STORED
                or info.flag_bits & 1
                or info.compress_size != info.file_size
                or not 0 < info.file_size <= prod(field.shape) * 8 + 4096
            ):
                raise ValueError("stored native member size/encryption cap exceeded")
            member = archive.read(
                info
            )  # ZIP sizes bounded before this byte allocation.
            offset, count = _header(member, field)
            validated.append((field, member, offset, count))
    # Every header and payload has been checked before any NumPy array exists.
    result = {}
    for field, member, offset, count in validated:
        array = np.frombuffer(
            member, dtype=np.dtype(field.dtype), count=count, offset=offset
        ).reshape(field.shape)
        if not np.isfinite(array).all():
            raise ValueError("nonfinite typed array")
        result[field.name] = array
    return result


def read_typed(path: Path, spec: Shard) -> dict[str, NDArray[Any]]:
    _schema(spec)
    return decode_typed(_read(path, spec.encoded_bound), spec)
