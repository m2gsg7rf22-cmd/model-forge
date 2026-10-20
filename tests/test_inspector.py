"""Real generated-artifact inspector tests; no network or model runtime."""

import json
import struct
import tempfile
import unittest
from pathlib import Path

from modelforge.inspector import MAX_HEADER, inspect_model


def safetensor(path: Path, header=None, data=b"\0" * 24) -> Path:
    if header is None:
        header = {
            "model.embed_tokens.weight": {
                "dtype": "F32",
                "shape": [2, 3],
                "data_offsets": [0, 24],
            }
        }
    raw = json.dumps(header).encode()
    path.write_bytes(struct.pack("<Q", len(raw)) + raw + data)
    return path


def string(value: str) -> bytes:
    raw = value.encode()
    return struct.pack("<Q", len(raw)) + raw


def gguf(
    path: Path, *, shape=(2, 3), kind=0, data=b"\0" * 24, offset=0, alignment=32
) -> Path:
    raw = b"GGUF" + struct.pack("<IQQ", 3, 1, 2)
    raw += string("general.architecture") + struct.pack("<I", 8) + string("llama")
    raw += string("general.alignment") + struct.pack("<II", 4, alignment)
    raw += string("blk.0.attn_q.weight") + struct.pack("<I", len(shape))
    raw += struct.pack("<" + "Q" * len(shape), *shape) + struct.pack(
        "<IQ", kind, offset
    )
    raw += b"\0" * ((32 - len(raw) % 32) % 32)
    path.write_bytes(raw + data)
    return path


class InspectorTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)

    def tearDown(self):
        self.temp.cleanup()

    def test_safetensors_real_bytes_and_evidence(self):
        report = inspect_model(str(safetensor(self.root / "weights.safetensors")))
        self.assertEqual(report["parameters"]["total"], 6)
        self.assertEqual(report["tensors"][0]["shape"], [2, 3])
        self.assertEqual(report["architecture"]["nodes"][1]["type"], "embedding")
        self.assertEqual(report["architecture"]["nodes"][1]["certainty"], "inferred")
        self.assertFalse(report["reconstruction"]["exact"])
        self.assertTrue(report["evidence"])
        json.dumps(report, allow_nan=False)

    def test_empty_and_scalar_tensors(self):
        p = safetensor(
            self.root / "empty.safetensors",
            {
                "empty": {"dtype": "F32", "shape": [0], "data_offsets": [0, 0]},
                "scalar": {"dtype": "F32", "shape": [], "data_offsets": [0, 4]},
            },
            b"\0" * 4,
        )
        self.assertEqual(inspect_model(str(p))["parameters"]["total"], 1)

    def test_invalid_safetensors_boundaries(self):
        entries = [
            ({"dtype": "F32", "shape": [2, 3], "data_offsets": [0, 25]}, b"\0" * 24),
            ({"dtype": "F32", "shape": [-1], "data_offsets": [0, 4]}, b"\0" * 4),
            ({"dtype": "F32", "shape": [True], "data_offsets": [0, 4]}, b"\0" * 4),
            ({"dtype": "F32", "shape": [2**65], "data_offsets": [0, 4]}, b"\0" * 4),
            ({"dtype": "F32", "shape": [1], "data_offsets": [1, 5]}, b"\0" * 5),
            ({"dtype": "F32", "shape": [1], "data_offsets": [0, 4]}, b"\0" * 5),
            ({"dtype": "BOGUS", "shape": [1], "data_offsets": [0, 4]}, b"\0" * 4),
            ({"dtype": "F32", "shape": [1], "data_offsets": [0]}, b"\0" * 4),
        ]
        for entry, data in entries:
            with self.subTest(entry=entry):
                p = safetensor(self.root / "invalid.safetensors", {"x": entry}, data)
                with self.assertRaises(ValueError):
                    inspect_model(str(p))

    def test_safetensors_duplicate_keys_overlap_and_header_limit(self):
        p = self.root / "invalid.safetensors"
        raw = b'{"a":{},"a":{}}'
        p.write_bytes(struct.pack("<Q", len(raw)) + raw)
        with self.assertRaisesRegex(ValueError, "Duplicate"):
            inspect_model(str(p))
        p.write_bytes(struct.pack("<Q", MAX_HEADER + 1))
        with self.assertRaises(ValueError):
            inspect_model(str(p))
        safetensor(
            p,
            {
                x: {"dtype": "F32", "shape": [1], "data_offsets": [0, 4]}
                for x in ["a", "b"]
            },
            b"\0" * 4,
        )
        with self.assertRaisesRegex(ValueError, "overlapping"):
            inspect_model(str(p))
        p.write_bytes(b"abc")
        with self.assertRaises(ValueError):
            inspect_model(str(p))

    def test_config_is_declaration_not_proof(self):
        p = self.root / "config.json"
        p.write_text(
            json.dumps(
                {
                    "model_type": "llama",
                    "num_hidden_layers": 2,
                    "auto_map": {"AutoModel": "evil.py"},
                }
            )
        )
        report = inspect_model(str(p))
        self.assertIsNone(report["parameters"]["total"])
        self.assertEqual(report["architecture"]["nodes"][0]["label"], "llama")
        self.assertEqual(report["metadata"]["num_hidden_layers"], 2)

    def test_hf_directory_shards_and_evidence_ids(self):
        (self.root / "config.json").write_text('{"model_type":"llama"}')
        safetensor(self.root / "a.safetensors")
        safetensor(
            self.root / "b.safetensors",
            {
                "lm_head.weight": {
                    "dtype": "F32",
                    "shape": [2, 3],
                    "data_offsets": [0, 24],
                }
            },
        )
        report = inspect_model(str(self.root))
        self.assertEqual(report["parameters"]["total"], 12)
        identifiers = [e["id"] for e in report["evidence"]]
        self.assertEqual(len(identifiers), len(set(identifiers)))
        self.assertTrue(
            all(i in identifiers for t in report["tensors"] for i in t["evidence_ids"])
        )
        (self.root / "model.safetensors.index.json").write_text(
            json.dumps(
                {
                    "weight_map": {
                        "model.embed_tokens.weight": "a.safetensors",
                        "lm_head.weight": "b.safetensors",
                    }
                }
            )
        )
        self.assertEqual(inspect_model(str(self.root))["parameters"]["total"], 12)

    def test_hf_index_missing_or_wrong_shard(self):
        safetensor(self.root / "a.safetensors")
        index = self.root / "model.safetensors.index.json"
        for mapping in [{"x": "../secret.safetensors"}, {"x": "a.safetensors"}]:
            index.write_text(json.dumps({"weight_map": mapping}))
            with self.assertRaises(ValueError):
                inspect_model(str(self.root))

    def test_duplicate_shard_tensors(self):
        safetensor(self.root / "a.safetensors")
        safetensor(self.root / "b.safetensors")
        with self.assertRaisesRegex(ValueError, "Duplicate tensor"):
            inspect_model(str(self.root))

    def test_gguf_real_metadata_and_bounds(self):
        p = gguf(self.root / "model.gguf")
        report = inspect_model(str(p))
        self.assertEqual(report["metadata"]["general.architecture"], "llama")
        self.assertEqual(report["parameters"]["total"], 6)
        self.assertEqual(report["tensors"][0]["dtype"], "F32")
        self.assertEqual(report["architecture"]["nodes"][1]["type"], "attention")
        for kwargs in [
            {"data": b"\0"},
            {"offset": 1},
            {"alignment": 3},
            {"shape": (33,), "kind": 2},
        ]:
            gguf(p, **kwargs)
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                inspect_model(str(p))

    def test_gguf_quantized_and_unknown_dtype(self):
        p = gguf(self.root / "q.gguf", shape=(32,), kind=2, data=b"\0" * 18)
        self.assertEqual(inspect_model(str(p))["tensors"][0]["nbytes"], 18)
        gguf(p, shape=(2,), kind=999)
        report = inspect_model(str(p))
        self.assertEqual(report["tensors"][0]["dtype"], "GGML_TYPE_999")
        self.assertTrue(any("not validated" in x for x in report["limitations"]))

    def test_gguf_malformed_metadata(self):
        p = self.root / "bad.gguf"
        for raw in [
            b"GGUF",
            b"bad!" + struct.pack("<IQQ", 3, 0, 0),
            b"GGUF" + struct.pack("<IQQ", 99, 0, 0),
            b"GGUF"
            + struct.pack("<IQQ", 3, 0, 1)
            + string("a")
            + struct.pack("<I", 999),
            b"GGUF" + struct.pack("<IQQ", 3, 0, 1) + struct.pack("<Q", 2**63),
        ]:
            p.write_bytes(raw)
            with self.subTest(raw=raw), self.assertRaises(ValueError):
                inspect_model(str(p))

    def test_unsafe_checkpoint_never_deserializes(self):
        p = self.root / "model.pt"
        p.write_bytes(b"pickle payload deliberately not parsed")
        with self.assertRaisesRegex(ValueError, "refused"):
            inspect_model(str(p))

    def test_unknown_and_large_optional_format(self):
        p = self.root / "model.unknown"
        p.write_bytes(b"anything")
        self.assertEqual(inspect_model(str(p))["status"], "unsupported")
        p = self.root / "large.onnx"
        with p.open("wb") as handle:
            handle.truncate(MAX_HEADER + 1)
        self.assertEqual(inspect_model(str(p))["status"], "unsupported")
        package = self.root / "x.mlpackage"
        package.mkdir()
        self.assertEqual(inspect_model(str(package))["status"], "unsupported")

    def test_missing_source_and_empty_directory(self):
        with self.assertRaises(FileNotFoundError):
            inspect_model(str(self.root / "missing"))
        with self.assertRaises(ValueError):
            inspect_model(str(self.root))


class ExtendedInspectorTests(unittest.TestCase):
    setUp = InspectorTests.setUp
    tearDown = InspectorTests.tearDown

    def test_json_metadata_validation(self):
        p = self.root / "config.json"
        for raw in [b"[]", b"{", b'{"x":NaN}', b"\xff"]:
            p.write_bytes(raw)
            with self.subTest(raw=raw), self.assertRaises(ValueError):
                inspect_model(str(p))

    def test_safetensors_invalid_header_entries(self):
        p = self.root / "bad.safetensors"
        for header in [
            {"x": []},
            {"__metadata__": {"x": 12}},
            {"x": {"dtype": "F32", "shape": [1]}},
        ]:
            safetensor(p, header, b"")
            with self.subTest(header=header), self.assertRaises(ValueError):
                inspect_model(str(p))
        raw = b' {"x":{}}'
        p.write_bytes(struct.pack("<Q", len(raw)) + raw)
        with self.assertRaises(ValueError):
            inspect_model(str(p))

    def test_gguf_array_preview_and_numeric_types(self):
        p = self.root / "array.gguf"
        raw = b"GGUF" + struct.pack("<IQQ", 3, 0, 4)
        raw += string("tokens") + struct.pack("<IIQ", 9, 8, 65)
        raw += b"".join(string(str(i)) for i in range(65))
        raw += string("flag") + struct.pack("<I?", 7, True)
        raw += string("temperature") + struct.pack("<If", 6, 0.5)
        raw += string("count") + struct.pack("<IQ", 10, 42)
        p.write_bytes(raw)
        report = inspect_model(str(p))
        self.assertEqual(report["metadata"]["tokens"]["count"], 65)
        self.assertEqual(len(report["metadata"]["tokens"]["preview"]), 64)
        self.assertEqual(report["metadata"]["count"], 42)

    def test_gguf_refuses_bad_values(self):
        p = self.root / "invalid.gguf"
        payloads = [
            struct.pack("<IIQ", 9, 8, 1000001),
            struct.pack("<IIQIQ", 9, 9, 1, 8, 0),
            struct.pack("<Id", 12, float("inf")),
            struct.pack("<IQ", 8, 1) + b"\xff",
        ]
        for payload in payloads:
            p.write_bytes(
                b"GGUF" + struct.pack("<IQQ", 3, 0, 1) + string("x") + payload
            )
            with self.subTest(payload=payload), self.assertRaises(ValueError):
                inspect_model(str(p))

    def test_architecture_declared_properties(self):
        p = self.root / "config.json"
        p.write_text(
            '{"model_type":"llama","num_attention_heads":8,"hidden_act":"silu"}'
        )
        report = inspect_model(str(p))
        properties = report["architecture"]["properties"]
        self.assertEqual(properties["attention_heads"]["value"], 8)
        self.assertEqual(properties["activation"]["certainty"], "declared")

    def test_tokenizer_directory_metadata(self):
        (self.root / "config.json").write_text('{"model_type":"llama"}')
        (self.root / "tokenizer_config.json").write_text(
            '{"tokenizer_class":"PreTrainedTokenizerFast","eos_token":"</s>"}'
        )
        (self.root / "tokenizer.json").write_text(
            '{"model":{"type":"BPE","vocab":{"a":0,"b":1}}}'
        )
        report = inspect_model(str(self.root))
        self.assertEqual(report["metadata"]["tokenizer"]["algorithm"], "BPE")
        self.assertEqual(report["metadata"]["tokenizer"]["vocabulary_entries"], 2)
        self.assertEqual(report["metadata"]["tokenizer"]["eos_token"], "</s>")

    def test_q8_1_current_ggml_block_size(self):
        p = gguf(self.root / "q8.gguf", shape=(32,), kind=9, data=b"\0" * 36)
        self.assertEqual(inspect_model(str(p))["tensors"][0]["nbytes"], 36)

    def test_onnx_real_graph_and_unopened_external_weights(self):
        try:
            import onnx
        except ImportError:
            self.skipTest("optional onnx package unavailable")
        helper = onnx.helper
        tensor = helper.make_tensor("weight", onnx.TensorProto.FLOAT, [2], [1.0, 2.0])
        graph = helper.make_graph(
            [
                helper.make_node("Add", ["x", "weight"], ["z"]),
                helper.make_node("Relu", ["z"], ["y"]),
            ],
            "tiny",
            [helper.make_tensor_value_info("x", onnx.TensorProto.FLOAT, [2])],
            [helper.make_tensor_value_info("y", onnx.TensorProto.FLOAT, [2])],
            [tensor],
        )
        model = helper.make_model(graph)
        path = self.root / "tiny.onnx"
        onnx.save(model, str(path))
        report = inspect_model(str(path))
        self.assertEqual(report["parameters"]["total"], 2)
        self.assertEqual(len(report["architecture"]["edges"]), 1)
        tensor = model.graph.initializer[0]
        tensor.ClearField("float_data")
        tensor.data_location = onnx.TensorProto.EXTERNAL
        loc = tensor.external_data.add()
        loc.key, loc.value = "location", "missing-weights.bin"
        path.write_bytes(model.SerializeToString())
        report = inspect_model(str(path))
        self.assertEqual(report["parameters"]["total"], 2)
        self.assertFalse((self.root / "missing-weights.bin").exists())

    def test_coreml_real_spec_without_runtime(self):
        try:
            from coremltools.proto import Model_pb2, FeatureTypes_pb2
        except ImportError:
            self.skipTest("optional coremltools package unavailable")
        spec = Model_pb2.Model()
        spec.specificationVersion = 1
        spec.glmRegressor.offset.append(0.0)
        inp = spec.description.input.add()
        inp.name = "x"
        inp.type.doubleType.MergeFrom(FeatureTypes_pb2.DoubleFeatureType())
        out = spec.description.output.add()
        out.name = "y"
        out.type.doubleType.MergeFrom(FeatureTypes_pb2.DoubleFeatureType())
        path = self.root / "regression.mlmodel"
        path.write_bytes(spec.SerializeToString())
        report = inspect_model(str(path))
        self.assertEqual(report["metadata"]["model_type"], "glmRegressor")
        self.assertEqual(report["metadata"]["inputs"], ["x"])
        self.assertIsNone(report["parameters"]["total"])


if __name__ == "__main__":
    unittest.main()
