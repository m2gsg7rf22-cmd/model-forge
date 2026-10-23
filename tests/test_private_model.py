import json
import threading
import urllib.request
from pathlib import Path
from typing import Any

import pytest

from modelforge import capability_eval as ce
from modelforge import private_build as pb
from modelforge import private_runtime as pr


def test_splits_are_disjoint_and_deterministic() -> None:
    val = ce.build_tasks("validation", 4)
    again = ce.build_tasks("validation", 4)
    test = ce.build_tasks("test", 4)
    assert [t.messages for t in val] == [t.messages for t in again]
    assert len(val) == 4 * len(ce.CATEGORIES)
    assert {json.dumps(t.messages) for t in val}.isdisjoint({json.dumps(t.messages) for t in test})
    with pytest.raises(ValueError):
        ce.build_tasks("bogus", 1)


def test_checkers_accept_reference_and_reject_wrong() -> None:
    for task in ce.build_tasks("validation", 6, ("math", "english", "long_conversation", "hebrew")):
        if task.reference:
            good = task.reference if task.category != "math" and "Answer" not in task.messages[-1]["content"] else f"Answer: {task.reference}"
            assert task.check(good), task.task_id
            assert not task.check("I do not know"), task.task_id


def test_number_and_tool_parsing() -> None:
    assert ce._last_number("steps 12 then 7\nAnswer: 1,234") == 1234
    calls = ce.parse_tool_calls('<tool_call>\n{"name": "multiply", "arguments": {"a": 2, "b": 3}}')
    assert calls == [{"name": "multiply", "arguments": {"a": 2, "b": 3}}]
    assert ce.parse_tool_calls("<tool_call>not json</tool_call>") == []


def test_code_runner_executes_in_isolation() -> None:
    code = "def add(a, b):\n    return a + b\n"
    assert ce.run_python_tests(code, "add", [(1, 2), (3, 4)], [3, 7])
    assert not ce.run_python_tests(code, "add", [(1, 2)], [4])
    assert not ce.run_python_tests("def add(:", "add", [(1, 2)], [3])
    assert not ce.run_python_tests("def add(a, b):\n    while True: pass\n", "add", [(1, 2)], [3], timeout=1)


def test_coding_tasks_have_valid_references() -> None:
    for task in ce.build_tasks("validation", 8, ("coding",)):
        assert not task.check("no code here")


def test_evaluate_scores_a_perfect_oracle() -> None:
    tasks = {t.task_id: t for t in ce.build_tasks("validation", 2, ("english", "factual"))}

    def oracle(task: ce.Task) -> str:
        return task.reference

    report = ce.evaluate(oracle, "validation", 2, ("english", "factual"))
    assert report["scores"] == {"english": 1.0, "factual": 1.0}
    assert len(report["rows"]) == len(tasks)
    assert "not a measured GPT-5.6 Luna" in report["target_profile"]


def test_wikidata_examples_exclude_eval_countries(tmp_path: Path) -> None:
    rows = [
        {"cHe": {"value": "צרפת"}, "capHe": {"value": "פריז"}},
        {"cHe": {"value": "פרו"}, "capHe": {"value": "לימה"}, "contHe": {"value": "דרום אמריקה"}},
        {"cHe": {"value": "בוליביה"}, "capHe": {"value": "סוקרה"}},
        {"cHe": {"value": "בוליביה"}, "capHe": {"value": "לה פס"}},
    ]
    path = tmp_path / "w.json"
    path.write_text(json.dumps({"results": {"bindings": rows}}), encoding="utf-8")
    examples = pb.wikidata_capital_examples(path)
    text = json.dumps(examples, ensure_ascii=False)
    assert "פריז" not in text and "בוליביה" not in text
    assert "לימה" in text and "דרום אמריקה" in text
    assert len(examples) == 3


def test_rejection_sampling_keeps_only_verified(tmp_path: Path) -> None:
    def gen(task: ce.Task) -> str:
        return task.reference if task.category == "english" else "wrong"

    kept, stats = pb.rejection_sample(gen, 3, ("english", "math"))
    assert stats["english"] == {"kept": 3, "sampled": 3}
    assert stats["math"]["kept"] == 0
    assert all(ex["messages"][-1]["role"] == "assistant" for ex in kept)
    counts = pb.write_split(kept * 4, tmp_path / "d")
    assert counts["train"] + counts["valid"] == 12
    assert (tmp_path / "d" / "train.jsonl").is_file()


def test_bundle_validation_refuses_unsafe(tmp_path: Path) -> None:
    root = tmp_path / "m"
    root.mkdir()
    with pytest.raises(ValueError):
        pr.validate_bundle(root)
    (root / "config.json").write_text(json.dumps({"model_type": "qwen3"}))
    with pytest.raises(ValueError, match="SafeTensors"):
        pr.validate_bundle(root)
    (root / "model.safetensors").write_bytes(b"x")
    assert pr.validate_bundle(root) == root.resolve()
    (root / "modeling.py").write_text("print(1)")
    with pytest.raises(ValueError, match="custom code"):
        pr.validate_bundle(root)
    (root / "modeling.py").unlink()
    (root / "config.json").write_text(json.dumps({"model_type": "evil"}))
    with pytest.raises(ValueError, match="Unsupported"):
        pr.validate_bundle(root)


def test_list_models_and_memory_limits(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    bundle = tmp_path / "models" / "tiny"
    bundle.mkdir(parents=True)
    (bundle / "config.json").write_text(json.dumps({"model_type": "llama"}))
    (bundle / "model.safetensors").write_bytes(b"1234")
    (bundle / "modelforge.json").write_text(json.dumps({"name": "tiny"}))
    monkeypatch.setenv("MODELFORGE_PRIVATE_HOME", str(tmp_path))
    models = pr.list_private_models()
    assert models[0]["name"] == "tiny" and models[0]["size_bytes"] == 4 and models[0]["card"] == {"name": "tiny"}
    with pytest.raises(ValueError):
        pr.PrivateChatModel(bundle, memory_limit_mb=999_999)
    assert pr.MANAGER.resolve("tiny") == bundle.resolve()
    with pytest.raises(RuntimeError, match="No private model"):
        pr.PrivateModelManager().chat([{"role": "user", "content": "hi"}])


class _FakeModel:
    def __init__(self) -> None:
        self.path = Path("/x/fake")
        self.loaded = True

    def chat(self, messages: list[dict[str, Any]], **settings: Any) -> dict[str, Any]:
        return {"reply": f"echo {messages[-1]['content']}", "tool_calls": [], "seconds": 0.01,
                "generation_tokens": 2, "prompt_tokens": 3, "finish_reason": "stop"}

    def status(self) -> dict[str, Any]:
        return {"loaded": True}


def test_local_api_serves_chat(monkeypatch: pytest.MonkeyPatch) -> None:
    import socket

    monkeypatch.setattr(pr.MANAGER, "current", _FakeModel())
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        port = s.getsockname()[1]
    threading.Thread(target=pr.serve_api, kwargs={"port": port}, daemon=True).start()
    import time

    body = json.dumps({"messages": [{"role": "user", "content": "שלום"}]}).encode()
    for _ in range(50):
        try:
            req = urllib.request.Request(f"http://127.0.0.1:{port}/v1/chat/completions", body, {"Content-Type": "application/json"})
            data = json.loads(urllib.request.urlopen(req, timeout=5).read())
            break
        except OSError:
            time.sleep(0.05)
    assert data["choices"][0]["message"]["content"] == "echo שלום"
    with pytest.raises(ValueError):
        pr.serve_api(host="0.0.0.0")


def test_mcp_exposes_private_tools(tmp_path: Path) -> None:
    import asyncio

    from modelforge.mcp_server import create_server

    server = create_server(str(tmp_path))
    names = {tool.name for tool in asyncio.run(server.list_tools())}
    assert {"private_model_status", "private_model_load", "private_model_unload", "private_chat", "private_leaderboard"} <= names


def test_mlx_work_runs_on_one_dedicated_thread(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Regression: MLX GPU streams are thread-local; callers on other threads must work."""
    bundle = tmp_path / "m"
    bundle.mkdir()
    (bundle / "config.json").write_text(json.dumps({"model_type": "llama"}))
    (bundle / "model.safetensors").write_bytes(b"x")
    model = pr.PrivateChatModel(bundle, memory_limit_mb=2048)
    seen: set[int] = set()

    def record() -> int:
        seen.add(threading.get_ident())
        return 1

    for _ in range(3):
        threading.Thread(target=lambda: model._executor.submit(record).result()).start()
    import time

    time.sleep(0.3)
    assert len(seen) == 1 and threading.get_ident() not in seen
    other = pr.PrivateChatModel(bundle, memory_limit_mb=2048)
    assert other._executor is model._executor  # model switching keeps the same MLX thread


def test_split_reasoning() -> None:
    assert pr.split_reasoning("plan a</think>\n\nAnswer: 4") == ("plan a", "Answer: 4")
    assert pr.split_reasoning("<think>\nx\n</think>ok") == ("x", "ok")
    assert pr.split_reasoning("<think>unfinished") == ("unfinished", "")
    assert pr.split_reasoning("plain") == ("", "plain")


def test_default_model_marker(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("MODELFORGE_PRIVATE_HOME", str(tmp_path))
    for name in ("a-model", "b-model"):
        b = tmp_path / "models" / name
        b.mkdir(parents=True)
        (b / "config.json").write_text(json.dumps({"model_type": "llama"}))
        (b / "model.safetensors").write_bytes(b"x")
    assert pr.default_model() is None and pr.list_private_models()[0]["name"] == "a-model"
    pr.set_default_model("b-model")
    assert pr.default_model() == "b-model" and pr.list_private_models()[0]["name"] == "b-model"
    with pytest.raises(ValueError):
        pr.set_default_model("missing")
