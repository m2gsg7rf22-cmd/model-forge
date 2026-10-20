import json
import hashlib
import math
import struct

import numpy as np
import pytest

from modelforge.builder import (
    create_model,
    load_model,
    train_model,
    compare_models,
    quantize_model,
    reconstruct_model,
)


def _rewrite_header(path, edit, payload_edit=None):
    data = path.read_bytes()
    size = struct.unpack("<Q", data[:8])[0]
    header = json.loads(data[8 : 8 + size])
    payload = bytearray(data[8 + size :])
    edit(header)
    if payload_edit:
        payload_edit(header, payload)
    encoded = json.dumps(header).encode()
    encoded += b" " * (-len(encoded) % 8)
    path.write_bytes(struct.pack("<Q", len(encoded)) + encoded + payload)


def test_exact_reload_and_standalone_export(tmp_path):
    a, b = tmp_path / "a", tmp_path / "b"
    result = create_model(a, d_model=16)
    original = load_model(a)
    original.save(b)
    reloaded = load_model(b)
    assert result["parameters"] > 256
    assert np.array_equal(original.logits([1, 2, 3]), reloaded.logits([1, 2, 3]))
    assert original.generate("hello", 8) == reloaded.generate("hello", 8)
    assert compare_models(a, b)["exact_on_all_tested_prompts"]
    assert (b / "runtime.py").exists()
    assert "not a reconstruction" in (b / "README.md").read_text()
    # The export executes independently of the installed modelforge package.
    import importlib.util

    spec = importlib.util.spec_from_file_location("exported_runtime", b / "runtime.py")
    runtime = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(runtime)
    assert np.array_equal(original.logits([3, 4]), runtime.load_model(b).logits([3, 4]))


def test_causal_attention_does_not_read_future(tmp_path):
    create_model(tmp_path)
    model = load_model(tmp_path)
    assert np.array_equal(model.logits([1, 2, 3])[:2], model.logits([1, 2, 255])[:2])


def test_training_reduces_loss_and_persists(tmp_path):
    create_model(tmp_path, seed=12)
    before = load_model(tmp_path)
    result = train_model(
        tmp_path, "abc abc abc abc abc abc abc", steps=150, learning_rate=0.5
    )
    report = result["training"]
    assert report["final_loss"] < report["initial_loss"] * 0.5
    after = load_model(tmp_path)
    assert np.array_equal(before.weights["query"], after.weights["query"])
    assert not np.array_equal(before.weights["output"], after.weights["output"])
    assert after.provenance["training"][-1] == report


def test_quantized_export_is_lossy_bounded_and_smaller(tmp_path):
    a, b = tmp_path / "float", tmp_path / "int8"
    create_model(a)
    quantize_model(a, b)
    x, y = load_model(a).logits([65, 66]), load_model(b).logits([65, 66])
    assert not np.array_equal(x, y)
    assert np.max(np.abs(x - y)) < 0.2
    assert (b / "model.safetensors").stat().st_size < (
        a / "model.safetensors"
    ).stat().st_size
    assert "lossy" in json.loads((b / "provenance.json").read_text())["fidelity"]
    with pytest.raises(ValueError):
        quantize_model(a, a)


def test_preflight_and_input_guards(tmp_path):
    with pytest.raises(MemoryError):
        create_model(tmp_path, memory_limit_mb=0.01)
    assert not (tmp_path / "model.safetensors").exists()
    with pytest.raises(ValueError):
        create_model(tmp_path, d_model=100000)
    create_model(tmp_path)
    with pytest.raises(MemoryError):
        load_model(tmp_path, memory_limit_mb=0.01)
    model = load_model(tmp_path)
    for ids in ([], [256], [-1], [0.5], [1] * 513):
        with pytest.raises(ValueError):
            model.logits(ids)
    with pytest.raises(ValueError):
        model.train("a")
    with pytest.raises(ValueError):
        model.generate("a", 100000)


def test_reject_corrupt_config_and_weights(tmp_path):
    create_model(tmp_path)
    config_path = tmp_path / "config.json"
    config = json.loads(config_path.read_text())
    config["num_hidden_layers"] = 12
    config_path.write_text(json.dumps(config))
    with pytest.raises(ValueError, match="contradicts"):
        load_model(tmp_path)
    create_model(tmp_path)
    weights_path = tmp_path / "model.safetensors"
    weights_path.write_bytes(b"invalid")
    with pytest.raises(ValueError, match="Truncated"):
        load_model(tmp_path)


def test_runtime_matches_independent_scalar_attention_reference(tmp_path):
    """Independent scalar loops catch graph mistakes that save/reload cannot."""
    create_model(tmp_path, d_model=4, context_length=8)
    model = load_model(tmp_path)
    w = model.weights
    tokens = [9, 32, 64]
    d = 4

    def multiply(vector, matrix):
        return [
            sum(float(vector[i]) * float(matrix[i, j]) for i in range(len(vector)))
            for j in range(matrix.shape[1])
        ]

    x = [
        [float(w["embedding"][token, j]) + float(w["position"][t, j]) for j in range(d)]
        for t, token in enumerate(tokens)
    ]
    queries = [multiply(row, w["query"]) for row in x]
    keys = [multiply(row, w["key"]) for row in x]
    values = [multiply(row, w["value"]) for row in x]
    reference = []
    for t in range(len(tokens)):
        scores = [
            sum(a * b for a, b in zip(queries[t], keys[s])) / math.sqrt(d)
            for s in range(t + 1)
        ]
        exp_scores = [math.exp(score - max(scores)) for score in scores]
        probs = [score / sum(exp_scores) for score in exp_scores]
        attended = [
            sum(probs[s] * values[s][j] for s in range(t + 1)) for j in range(d)
        ]
        projected = multiply(attended, w["attention_out"])
        hidden = [math.tanh(x[t][j] + projected[j]) for j in range(d)]
        reference.append(
            [
                v + float(w["bias"][j])
                for j, v in enumerate(multiply(hidden, w["output"]))
            ]
        )
    np.testing.assert_allclose(model.logits(tokens), reference, atol=2e-7, rtol=2e-5)


def test_reconstruct_accessible_fixture_preserves_tokens_and_source_hashes(tmp_path):
    source, rebuilt = tmp_path / "source", tmp_path / "rebuilt"
    create_model(source)
    train_model(source, "authorized project corpus corpus corpus", steps=5)
    result = reconstruct_model(source, rebuilt)
    assert result["parity"]["exact_on_all_tested_prompts"]
    provenance = load_model(rebuilt).provenance
    assert (
        provenance["reconstruction"]["source_weights_sha256"]
        == hashlib.sha256((source / "model.safetensors").read_bytes()).hexdigest()
    )
    assert provenance["training"][0]["steps"] == 5
    assert load_model(source).generate("hello", 10) == load_model(rebuilt).generate(
        "hello", 10
    )
    with pytest.raises(ValueError, match="separate"):
        reconstruct_model(source, source)


@pytest.mark.parametrize(
    "mutation",
    [
        lambda h: h.pop("query"),
        lambda h: h.update(unexpected=h["query"]),
        lambda h: h["query"].update(dtype="F64"),
        lambda h: h["query"].update(shape=[1000000000]),
        lambda h: h["query"].update(data_offsets=[0.0, 4.0]),
        lambda h: h["query"].update(data_offsets=[-1, 3]),
        lambda h: h["query"].update(data_offsets=[0, 4]),
        lambda h: h["query"].update(data_offsets=h["key"]["data_offsets"]),
        lambda h: h.update({"query.scale": h["bias"]}),
    ],
)
def test_malformed_tensor_metadata_rejected(tmp_path, mutation):
    create_model(tmp_path)
    _rewrite_header(tmp_path / "model.safetensors", mutation)
    with pytest.raises(ValueError):
        load_model(tmp_path)


def test_nonfinite_tensor_rejected(tmp_path):
    create_model(tmp_path)

    def inject_nan(header, payload):
        start = header["query"]["data_offsets"][0]
        payload[start : start + 4] = struct.pack("<f", float("nan"))

    _rewrite_header(tmp_path / "model.safetensors", lambda h: None, inject_nan)
    with pytest.raises(ValueError, match="Nonfinite"):
        load_model(tmp_path)


@pytest.mark.parametrize(
    "mutation",
    [
        lambda h: h.pop("query.scale"),
        lambda h: h["query.scale"].update(shape=[2]),
        lambda h: h["query.scale"].update(dtype="I8"),
        lambda h: h["query.scale"].update(data_offsets=[0, 4]),
    ],
)
def test_malformed_quantization_scales_rejected(tmp_path, mutation):
    source, quantized = tmp_path / "a", tmp_path / "b"
    create_model(source)
    quantize_model(source, quantized)
    _rewrite_header(quantized / "model.safetensors", mutation)
    with pytest.raises(ValueError):
        load_model(quantized)


@pytest.mark.parametrize("scale", [0, -1, float("inf"), float("nan")])
def test_invalid_quantization_scale_values_rejected(tmp_path, scale):
    source, quantized = tmp_path / "a", tmp_path / "b"
    create_model(source)
    quantize_model(source, quantized)

    def inject_scale(header, payload):
        start = header["query.scale"]["data_offsets"][0]
        payload[start : start + 4] = struct.pack("<f", scale)

    _rewrite_header(quantized / "model.safetensors", lambda h: None, inject_scale)
    with pytest.raises(ValueError, match="scale value"):
        load_model(quantized)


@pytest.mark.parametrize("steps", [0, -1, 10001, True, 1.5])
def test_training_step_guards(tmp_path, steps):
    create_model(tmp_path)
    with pytest.raises(ValueError, match="steps"):
        load_model(tmp_path).train("abc", steps=steps)


@pytest.mark.parametrize("rate", [0, -1, 11, float("nan"), float("inf")])
def test_training_learning_rate_guards(tmp_path, rate):
    create_model(tmp_path)
    with pytest.raises(ValueError, match="learning_rate"):
        load_model(tmp_path).train("abc", learning_rate=rate)


@pytest.mark.parametrize("budget", [0, -1, float("nan"), float("inf")])
def test_invalid_memory_budgets_rejected(tmp_path, budget):
    with pytest.raises(ValueError, match="memory_limit_mb"):
        create_model(tmp_path, memory_limit_mb=budget)


def test_artifact_hashes_detect_finite_weight_and_source_changes(tmp_path):
    create_model(tmp_path)
    data = bytearray((tmp_path / "model.safetensors").read_bytes())
    data[-1] ^= 1
    (tmp_path / "model.safetensors").write_bytes(data)
    with pytest.raises(ValueError, match="hash mismatch"):
        load_model(tmp_path)
    create_model(tmp_path)
    with (tmp_path / "runtime.py").open("a") as stream:
        stream.write("\n# changed\n")
    with pytest.raises(ValueError, match="hash mismatch"):
        load_model(tmp_path)


def test_bounded_inputs_and_context_training_report(tmp_path):
    create_model(tmp_path, context_length=4)
    model = load_model(tmp_path)
    report = model.train("abc abc abc", steps=1)
    assert report["corpus_truncated"]
    assert report["training_bytes"] == 5
    assert model.decode(model.encode("שלום")) == "שלום"
    assert model.generate("", 0) == ""
    assert len(model.generate("abc" * 100, 2)) <= 2
    with pytest.raises(ValueError, match="corpus"):
        model.train("a" * (1024**2 + 1))
    with pytest.raises(ValueError, match="Prompt"):
        model.generate("a" * (1024**2 + 1))


def test_disk_preflight_failure_injection(tmp_path, monkeypatch):
    """Injected low disk result; no claim that the real disk was exhausted."""
    import shutil

    usage = shutil.disk_usage(tmp_path)
    monkeypatch.setattr(
        "modelforge.builder.shutil.disk_usage",
        lambda _: type(usage)(usage.total, usage.used, 0),
    )
    with pytest.raises(OSError, match="disk space"):
        create_model(tmp_path)
    assert not (tmp_path / "model.safetensors").exists()


def test_bounded_json_and_duplicate_header_rejection(tmp_path):
    create_model(tmp_path)
    config_path = tmp_path / "config.json"
    config_path.write_text(" " * 65537)
    with pytest.raises(ValueError, match="size limit"):
        load_model(tmp_path)
    create_model(tmp_path)
    config_path.write_text("[]")
    with pytest.raises(ValueError, match="object"):
        load_model(tmp_path)
    create_model(tmp_path)
    path = tmp_path / "model.safetensors"
    header = b'{"query":{},"query":{}}'
    path.write_bytes(struct.pack("<Q", len(header)) + header)
    with pytest.raises(ValueError, match="Duplicate"):
        load_model(tmp_path)


def test_tokenizer_mismatch_rejected(tmp_path):
    create_model(tmp_path)
    (tmp_path / "tokenizer.json").write_text('{"type":"bpe"}')
    with pytest.raises(ValueError, match="tokenizer"):
        load_model(tmp_path)
