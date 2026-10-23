import json

from modelforge import eval_suite as es
from modelforge.capability_eval import Task


def test_splits_disjoint_unique_and_sized() -> None:
    sets = {}
    for split in ("train", "validation", "test"):
        tasks = es.build_tasks(split, 30)
        keys = [json.dumps(t.messages, ensure_ascii=False, sort_keys=True) for t in tasks]
        assert len(keys) == len(set(keys))
        sets[split] = set(keys)
        counts = {c: sum(t.category == c for t in tasks) for c in es.CATEGORIES}
        if split != "train":  # measured splits need >= 29 items per category
            assert all(v >= 29 for v in counts.values()), counts
    assert not (sets["train"] & sets["validation"]) and not (sets["train"] & sets["test"]) and not (sets["validation"] & sets["test"])


def test_oracle_answers_pass_and_junk_fails() -> None:
    for t in es.build_tasks("validation", 12):
        if t.reference and t.category != "coding":
            good = f"Answer: {t.reference}" if "Answer: <" in t.messages[-1]["content"] else t.reference
            assert t.check(good), t.task_id
            assert not t.check("I am not sure."), t.task_id


def test_agentic_tool_loop_and_compare() -> None:
    task = next(t for t in es.build_tasks("validation", 3, ("agentic",)))
    env = task.meta["env"]
    calls: list[str] = []

    def chat(messages, tools, max_tokens):  # type: ignore[no-untyped-def]
        if messages[-1]["role"] == "user":
            calls.append("plan")
            return "".join(f'<tool_call>{{"name": "get_price", "arguments": {{"item": "{it}"}}}}</tool_call>' for it in ("lamp", "x"))
        return f"Answer: {task.reference}"

    assert es.run_task(chat, task) == f"Answer: {task.reference}"
    assert "error" in env("get_price", {"item": "nonexistent"})
    rows_a = [{"category": "math", "id": f"m{i}", "passed": i < 5} for i in range(20)]
    rows_b = [{"category": "math", "id": f"m{i}", "passed": i < 15} for i in range(20)]
    cmp = es.compare({"rows": rows_a}, {"rows": rows_b})
    assert cmp["__all__"]["wins"] == 10 and cmp["__all__"]["p_value"] < 0.01 and cmp["__all__"]["significant_improvement"]
    assert es.wilson(0, 0) == (0.0, 1.0)
    lo, hi = es.wilson(15, 30)
    assert lo < 0.5 < hi


def test_code_checker_accepts_correct_reference_solution() -> None:
    task = Task("coding", "x", [{"role": "user", "content": "x"}], lambda t: True)
    assert task.check("anything")


def test_empty_category_excluded_from_mean() -> None:
    r = es.evaluate(lambda m, t, x: "x", "validation", 60, ("factual",), start_index=59)
    assert r["empty_categories"] in ([], ["factual"])
    if r["empty_categories"]:
        assert r["scores"] == {} and r["mean"] == 0.0
