import math

import pytest

from modelforge.evaluation import benchmark_report, compare_reports, rank_behavioral_hypotheses


def report(*samples):
    return {"results":[{"prompt":prompt,"output":output,"latency_seconds":latency,"valid_json":False} for prompt,output,latency in samples]}


def properties(result):
    return {h["property"]:h for h in result["hypotheses"]}


def test_rank_observations_and_wilson_reference_values():
    fixture = report(("same",'{}',1),("same",'{}',2),("same",'no',3),("other",'[]',4))
    result = rank_behavioral_hypotheses(fixture)
    props = properties(result)
    assert props["json_validity"]["successes"] == 3
    assert props["json_validity"]["rate"] == .75
    assert props["json_validity"]["sample_count"] == 4
    assert props["json_validity"]["wilson_95_reference_interval"] == pytest.approx([0.30064184,0.95441274])
    assert props["repeat_consistency"]["rate"] == .5
    assert props["repeat_consistency"]["sample_count"] == 2
    assert props["json_validity"]["rank"] == 1
    assert result["identifiability"]["weights"].startswith("not_identifiable")


def test_repeat_grouping_preserves_nonadjacent_occurrences():
    fixture = report(("a","same",None),("b","x",None),("a","same",None),("b","y",None))
    props = properties(rank_behavioral_hypotheses(fixture))
    assert props["repeat_consistency"]["successes"] == 1
    assert props["repeat_consistency"]["sample_count"] == 2


def test_empty_data_and_singletons_never_manufacture_consistency():
    result = rank_behavioral_hypotheses(report())
    assert all(h["rate"] is None and h["status"] == "insufficient_observations" for h in result["hypotheses"])
    result = rank_behavioral_hypotheses(report(("a","x",None),("b","y",None)))
    assert properties(result)["repeat_consistency"]["sample_count"] == 0


@pytest.mark.parametrize("text,valid",[("null",True),("true",True),("42",True),('"text"',True),("[]",True),("{}",True),("NaN",False),("Infinity",False),("{bad",False),("```json\n{}\n```",False),("",False)])
def test_json_is_recomputed_from_output_and_rfc_constants_rejected(text,valid):
    result = properties(rank_behavioral_hypotheses(report(("p",text,None))))["json_validity"]
    assert result["rate"] == int(valid)
    assert result["status"] == ("observed_all" if valid else "observed_none")


def test_compare_repeated_lists_pair_by_order_without_overwriting():
    left = report(("a",'{}',1),("a",'bad',3),("a",'unmatched',999),("b",'hi',None),("left",'x',1000))
    right = report(("a",'{}',2),("a",'[]',4),("b",'bye',6),("right",'x',2000))
    result = compare_reports(left,right)
    assert result["matched_sample_count"] == 3
    assert result["left_unmatched_sample_count"] == 2
    assert result["right_unmatched_sample_count"] == 1
    assert result["output_change"]["successes"] == 2
    assert result["output_change"]["rate"] == pytest.approx(2/3)
    assert result["json_rate_delta_right_minus_left"] == pytest.approx(1/3)
    assert result["left_median_latency_seconds"] == 2
    assert result["right_median_latency_seconds"] == 4
    assert result["left_latency_sample_count"] == 2
    assert next(p for p in result["per_prompt"] if p["prompt"] == "a")["paired_samples"] == 2


def test_comparison_no_shared_prompts_and_missing_latencies():
    result = compare_reports(report(("a","x",None)),report(("b","x",None)))
    assert result["matched_sample_count"] == 0
    assert result["output_change"]["rate"] is None
    assert result["json_rate_delta_right_minus_left"] is None
    assert result["left_median_latency_seconds"] is None
    same = compare_reports(report(("a","x",None)),report(("a","x",None)))
    assert same["output_change"]["rate"] == 0
    assert same["right_median_latency_seconds"] is None


def test_benchmark_only_explicit_labels_exact_unmodified_text():
    fixture = report(("p","yes",1),("p","yes ",1),("other","yes",1),("q","no",1))
    result = benchmark_report(fixture,{"p":"yes","q":"no","missing":"x"})
    assert result["score"]["successes"] == 2
    assert result["score"]["sample_count"] == 3
    assert result["unlabeled_sample_count"] == 1
    assert result["labeled_prompt_count"] == 2
    assert result["missing_labeled_prompts"] == ["missing"]
    assert benchmark_report(fixture,{})["score"]["rate"] is None


@pytest.mark.parametrize("fixture", [{},{"results":None},{"results":{}},{"results":[None]},{"results":[{"prompt":"x","output":42}]},{"results":[{"prompt":42,"output":"x"}]}])
def test_malformed_reports_rejected(fixture):
    for evaluate in (rank_behavioral_hypotheses,lambda r:compare_reports(r,report()),lambda r:benchmark_report(r,{})):
        with pytest.raises(ValueError):
            evaluate(fixture)


@pytest.mark.parametrize("latency",[math.nan,math.inf,-1,True,"1"])
def test_invalid_latency_rejected(latency):
    with pytest.raises(ValueError,match="Latency"):
        rank_behavioral_hypotheses(report(("p","x",latency)))


def test_nonstring_labels_rejected():
    with pytest.raises(ValueError,match="labels"):
        benchmark_report(report(),{"p":123})


def test_rank_can_put_consistency_first_without_claiming_architecture():
    result = rank_behavioral_hypotheses(report(("p","bad",None),("p","bad",None),("p","bad",None)))
    assert result["hypotheses"][0]["property"] == "repeat_consistency"
    assert result["identifiability"]["architecture"].startswith("not_identifiable")
    assert any("no independence" in limitation for limitation in result["limitations"])
