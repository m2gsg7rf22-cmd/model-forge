# Complete local research project

After installation, run from a writable folder:

```sh
modelforge --home ./example-data example --destination ./example-export.zip
```

This is a real CPU workflow: SQLite project creation; project-owned causal attention
weights and byte tokenizer; config/tensor inspection; greedy inference; output-head
training; reconstruction into a second supported bundle; finite exact-logit parity;
ZIP export including runtime source, model card, provenance and findings.

Open the same data in the desktop with `modelforge --home ./example-data gui`.
Browse the project model directory in Explorer, inspect the architecture graph,
and review reconstruction parity in the command output. Build/inspection/training
findings persist. The secondary parity result is emitted by the CLI example; it
is not a claim about arbitrary imported architectures.

A second project can reconstruct the first through Reconstruction Studio's
supported-bundle field. Quantize the current fixture to store lossy INT8 weights.
No downloaded weights or provider outputs are needed. The model is a correctness
fixture with random initial weights, not a useful pretrained general LLM.
