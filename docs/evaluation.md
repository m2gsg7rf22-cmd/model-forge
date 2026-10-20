# Offline behavioral evaluation

Engine C evaluation operates on saved probe observations. It makes no network requests and never trains on report outputs.

```python
from modelforge.evaluation import rank_behavioral_hypotheses, compare_reports, benchmark_report

summary = rank_behavioral_hypotheses(report)
comparison = compare_reports(before_report, after_report)
score = benchmark_report(report, {"What is 2+2?": "4"})
```

Each report uses the existing probe format: `{"results": [{"prompt": "...", "output": "...", "latency_seconds": 0.25}]}`. Repeated prompts remain separate rows; evaluation groups them into sample lists without overwriting observations. Latency may be absent or null; supplied numbers must be finite and nonnegative. Invalid reports raise `ValueError`.

`rank_behavioral_hypotheses` ranks two observable empirical properties: RFC JSON syntax validity and identical output text between successive occurrences of an identical prompt. It recalculates JSON validity from actual outputs rather than trusting a stored flag. JSON values include strings, numbers, arrays and null; validity does not imply an object or schema compliance. NaN and Infinity are rejected. Prompts observed once provide no repeat-consistency evidence.

For each property the report contains successes, sample count, empirical rate, status and a Wilson 95% binomial reference interval. Ranking uses the lower endpoint of that interval, with unobserved properties last. The Wilson formula uses z=1.959963984540054, center `(p+z²/(2n))/(1+z²/n)` and radius `z sqrt(p(1-p)/n+z²/(4n²))/(1+z²/n)`. These are reference intervals, not calibrated probabilities: selected probes and repeated samples need not be independent or representative. The ranks do not use an architectural prior, identify weights, prove determinism or establish behavioral reconstruction.

`compare_reports` matches exact prompt strings and pairs repeated samples by occurrence order up to the smaller sample count for each prompt. It reports leftovers on each side, exact output change frequency, matched-only JSON rates and their right-minus-left difference, and matched-only median available latencies. Per-prompt counts make missing data visible. This pairing is a bookkeeping convention. It does not establish statistical significance or causality; sampling settings, routing, hidden prompts, network conditions and model implementations may differ. Empty comparisons return null rates and latencies rather than zero evidence masquerading as agreement.

`benchmark_report` scores exact Unicode text equality only where the caller supplies an explicit prompt-to-answer label. It does not normalize whitespace, verify labels, guess expected answers or measure semantic equivalence. Repeated labeled samples count individually; it reports unlabeled observations, missing expected prompts and per-prompt counts. This is an explicit-label experimental metric, not a claim of standardized benchmark performance.
