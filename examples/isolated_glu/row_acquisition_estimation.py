"""Public-only paired-prefix estimator, fixed numerical selection unchanged."""

from __future__ import annotations
import json
from pathlib import Path
import sys
from typing import Any, cast
from fresh_rank_estimation import estimate as base_estimate
from fresh_rank_common import create_store, save_json, sha
from row_acquisition_view import RowView, registered_view


def estimate(
    view: RowView, store: Any, output: Path, *, registered: bool = True
) -> dict[str, Any]:
    if registered:
        registered_view(view)
    # Frozen estimator uses only the FD view interface; no concrete-type checks.
    return base_estimate(cast(Any, view), store, output, registered=False)


def build(request: dict[str, Any]) -> dict[str, Any]:
    output = Path(request["output"])
    output.mkdir(exist_ok=False)
    path = output / "estimator-report.json"
    report: dict[str, Any] = dict(
        status="started",
        certificate=False,
        private_gate="closed",
        new_queries=0,
        new_fits=0,
    )
    save_json(path, report)
    try:
        view = RowView(
            Path(request["public_root"]),
            request["cases_sha256"],
            request["manifest_sha256"],
        )
        report.update(
            estimate(
                view, create_store(output / "arrays", request["store_limits"]), output
            ),
            source_commit=request["source_commit"],
        )
        save_json(path, report)
        return report
    except Exception as exc:
        report.update(
            status="failed", error_type=type(exc).__name__, error=str(exc)[:500]
        )
        save_json(path, report)
        raise


if __name__ == "__main__":
    request = json.load(sys.stdin)
    try:
        r = build(request)
        print(
            json.dumps(
                dict(
                    status=r["status"],
                    report_sha256=sha(
                        Path(request["output"]) / "estimator-report.json"
                    ),
                )
            )
        )
    except Exception as exc:
        print(
            json.dumps(
                dict(
                    status="failed", error_type=type(exc).__name__, error=str(exc)[:500]
                )
            )
        )
