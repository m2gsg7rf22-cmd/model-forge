"""Descriptive evaluation of existing authorized text probe reports; no network or training."""

from __future__ import annotations

import json
import math
import statistics
from collections import defaultdict
from dataclasses import dataclass
from typing import Any, Mapping


@dataclass(frozen=True)
class _Sample:
    prompt: str
    output: str
    latency: float | None


def _samples(report: Mapping[str, Any]) -> list[_Sample]:
    rows = report.get("results")
    if not isinstance(rows, list) or len(rows) > 100000:
        raise ValueError("Report results must be a list of at most 100000 samples")
    result = []
    for row in rows:
        if (
            not isinstance(row, dict)
            or not isinstance(row.get("prompt"), str)
            or not isinstance(row.get("output"), str)
        ):
            raise ValueError("Every sample must have string prompt and output fields")
        latency = row.get("latency_seconds")
        if latency is not None and (
            isinstance(latency, bool)
            or not isinstance(latency, (int, float))
            or not math.isfinite(latency)
            or latency < 0
        ):
            raise ValueError("Latency must be a finite nonnegative number or null")
        result.append(
            _Sample(
                row["prompt"],
                row["output"],
                float(latency) if latency is not None else None,
            )
        )
    return result


def _json_valid(output: str) -> bool:
    # RFC JSON excludes NaN and Infinity even though Python accepts them by default.
    def reject_constant(value: str) -> None:
        raise ValueError(f"Not JSON: {value}")

    try:
        json.loads(output, parse_constant=reject_constant)
        return True
    except (ValueError, RecursionError):
        return False


def _rate(successes: int, total: int) -> dict[str, Any]:
    if not total:
        return {
            "successes": 0,
            "sample_count": 0,
            "rate": None,
            "wilson_95_reference_interval": None,
            "status": "insufficient_observations",
        }
    rate = successes / total
    z = 1.959963984540054
    denom = 1 + z * z / total
    center = (rate + z * z / (2 * total)) / denom
    margin = (
        z * math.sqrt(rate * (1 - rate) / total + z * z / (4 * total * total)) / denom
    )
    return {
        "successes": successes,
        "sample_count": total,
        "rate": rate,
        "wilson_95_reference_interval": [
            max(0.0, center - margin),
            min(1.0, center + margin),
        ],
        "status": (
            "observed_all"
            if successes == total
            else "observed_none" if successes == 0 else "mixed_observations"
        ),
    }


def _groups(samples: list[_Sample]) -> dict[str, list[_Sample]]:
    groups: dict[str, list[_Sample]] = defaultdict(list)
    for sample in samples:
        groups[sample.prompt].append(sample)
    return dict(groups)


def _median(samples: list[_Sample]) -> float | None:
    values = [sample.latency for sample in samples if sample.latency is not None]
    return float(statistics.median(values)) if values else None


INTERVAL_LIMITATION = (
    "Wilson intervals are binomial reference intervals, not calibrated confidence for correlated or selected probes; "
    "no independence, representativeness, architectural prior or posterior probability is established."
)
IDENTIFIABILITY = {
    "architecture": "not_identifiable_from_these_observations",
    "weights": "not_identifiable_from_these_observations",
    "behavioral_reconstruction": "not_established",
}


def rank_behavioral_hypotheses(report: Mapping[str, Any]) -> dict[str, Any]:
    """Rank two observable empirical properties by Wilson reference lower bound.

    Consistency compares each repeat to its immediately previous occurrence of the
    identical prompt. These dependent comparisons are descriptive, not IID evidence.
    """
    samples = _samples(report)
    groups = _groups(samples)
    comparisons = [
        (a.output == b.output)
        for group in groups.values()
        for a, b in zip(group, group[1:])
    ]
    hypotheses = [
        {
            "claim": "Observed outputs parse as RFC JSON values",
            "property": "json_validity",
            **_rate(sum(_json_valid(s.output) for s in samples), len(samples)),
        },
        {
            "claim": "Repeated identical prompts produce identical adjacent sampled text",
            "property": "repeat_consistency",
            **_rate(sum(comparisons), len(comparisons)),
        },
    ]

    def lower_bound(hypothesis: dict[str, Any]) -> float:
        interval = hypothesis["wilson_95_reference_interval"]
        return float(interval[0]) if interval is not None else -1.0

    hypotheses.sort(key=lower_bound, reverse=True)
    for index, hypothesis in enumerate(hypotheses, 1):
        hypothesis["rank"] = index
    return {
        "kind": "descriptive_behavioral_hypotheses",
        "sample_count": len(samples),
        "unique_prompt_count": len(groups),
        "ranking_basis": "descending Wilson reference lower bound; ties retain property order",
        "hypotheses": hypotheses,
        "identifiability": dict(IDENTIFIABILITY),
        "limitations": [
            INTERVAL_LIMITATION,
            "Text equality and JSON syntax do not establish semantic correctness, schema compliance, determinism or a context limit.",
        ],
    }


def compare_reports(
    left: Mapping[str, Any], right: Mapping[str, Any]
) -> dict[str, Any]:
    """Match exact prompt strings and pair repeated samples by occurrence order."""
    left_samples, right_samples = _samples(left), _samples(right)
    left_groups, right_groups = _groups(left_samples), _groups(right_samples)
    pairs = [
        (a, b)
        for prompt, group in left_groups.items()
        for a, b in zip(group, right_groups.get(prompt, []))
    ]
    left_matched = [a for a, _ in pairs]
    right_matched = [b for _, b in pairs]
    changed = sum(a.output != b.output for a, b in pairs)
    left_json = _rate(sum(_json_valid(s.output) for s in left_matched), len(pairs))
    right_json = _rate(sum(_json_valid(s.output) for s in right_matched), len(pairs))
    per_prompt = [
        {
            "prompt": prompt,
            "left_samples": len(group),
            "right_samples": len(right_groups.get(prompt, [])),
            "paired_samples": min(len(group), len(right_groups.get(prompt, []))),
        }
        for prompt, group in left_groups.items()
    ]
    per_prompt.extend(
        {
            "prompt": prompt,
            "left_samples": 0,
            "right_samples": len(group),
            "paired_samples": 0,
        }
        for prompt, group in right_groups.items()
        if prompt not in left_groups
    )
    return {
        "kind": "matched_probe_comparison",
        "pairing": "exact prompt text, occurrence order, minimum sample count per prompt",
        "matched_sample_count": len(pairs),
        "left_unmatched_sample_count": len(left_samples) - len(pairs),
        "right_unmatched_sample_count": len(right_samples) - len(pairs),
        "per_prompt": per_prompt,
        "output_change": _rate(changed, len(pairs)),
        "left_json": left_json,
        "right_json": right_json,
        "json_rate_delta_right_minus_left": (
            right_json["rate"] - left_json["rate"] if pairs else None
        ),
        "left_median_latency_seconds": _median(left_matched),
        "right_median_latency_seconds": _median(right_matched),
        "left_latency_sample_count": sum(s.latency is not None for s in left_matched),
        "right_latency_sample_count": sum(s.latency is not None for s in right_matched),
        "identifiability": dict(IDENTIFIABILITY),
        "limitations": [
            "All aggregate comparisons use matched samples only; unmatched repetitions are reported, not silently overwritten.",
            "Occurrence pairing is a bookkeeping convention, not experimental randomization. Model settings, stochastic sampling, service routing and timing may differ.",
            "Observed differences establish neither statistical significance nor a causal model-version effect. Latency is end-to-end, not inference time or model size.",
            INTERVAL_LIMITATION,
        ],
    }


def benchmark_report(
    report: Mapping[str, Any], expected: Mapping[str, str]
) -> dict[str, Any]:
    """Score exact text equality only for explicitly supplied prompt labels."""
    if not all(
        isinstance(key, str) and isinstance(value, str)
        for key, value in expected.items()
    ):
        raise ValueError("Expected labels must map prompt strings to output strings")
    samples = _samples(report)
    labeled = [sample for sample in samples if sample.prompt in expected]
    groups = _groups(labeled)
    return {
        "kind": "explicit_label_exact_match",
        "metric": "exact Unicode text equality, no normalization",
        "score": _rate(
            sum(sample.output == expected[sample.prompt] for sample in labeled),
            len(labeled),
        ),
        "total_sample_count": len(samples),
        "unlabeled_sample_count": len(samples) - len(labeled),
        "labeled_prompt_count": len(groups),
        "expected_prompt_count": len(expected),
        "missing_labeled_prompts": [
            prompt for prompt in expected if prompt not in groups
        ],
        "per_prompt": [
            {
                "prompt": prompt,
                "score": _rate(
                    sum(s.output == expected[prompt] for s in group), len(group)
                ),
            }
            for prompt, group in groups.items()
        ],
        "limitations": [
            "Labels are user-provided ground truth and are not independently verified. Exact match does not assess semantic correctness or broad capabilities.",
            "Repeated samples count individually and may be correlated; this is not an established standardized benchmark.",
            INTERVAL_LIMITATION,
        ],
    }
