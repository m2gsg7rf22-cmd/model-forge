# Upstream design review

No upstream application code is vendored. Links were inspected on 2026-10-20.

[REA](https://github.com/morluto/rea) provides artifact investigation workflows and agent tools. ModelForge borrows the interaction idea of shared investigations, evidence and runnable tools. It independently implements model-specific metadata readers and a research service; it does not wrap REA or claim its reverse-engineering capabilities. Check the upstream license for any future code reuse.

[Anthropic official plugins](https://github.com/anthropics/claude-plugins-official) supplies layout examples. ModelForge follows the documented manifest/component directory convention with its own content. Repository-level license is Apache-2.0; individual plugin licensing may differ.

[MCP Python SDK](https://github.com/modelcontextprotocol/python-sdk) is the actual runtime dependency (MIT). ModelForge pins its compatible 1.x API (`FastMCP`); SDK v2 introduces different APIs and should not be upgraded without transport tests. The SDK is imported, not copied into source.

[Karpathy autoresearch](https://github.com/karpathy/autoresearch) demonstrates bounded experiment loops around a defined training task. ModelForge uses persistent jobs and measured results, but does not include autoresearch's GPU training loop, datasets or code. No requirement for its hardware is introduced.

See [ecosystem.md](ecosystem.md) for the requested model-library evaluation and license links. The fixture's NumPy implementation and original model assets are MIT; installed dependencies retain their own licenses. Frozen app builds collect dependency LICENSE/COPYING notices into THIRD_PARTY_NOTICES, including Qt/PySide license terms. Qt stays dynamically linked; users may replace compatible libraries. Qt upstream source and license information: https://code.qt.io/ and https://doc.qt.io/qtforpython-6/licenses.html .
