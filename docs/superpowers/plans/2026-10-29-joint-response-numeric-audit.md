# Independent joint-response numeric audit implementation plan

Design: docs/glu-joint-response-numeric-audit-design.md, SHA79a6ded70cddd137111d45444643b239b1b7722cf467a55e60172b3da836b1e3. Independent design review feb1728295d1c2e4eee01446383711e3551d2cc57c0d88c247c49c8a71d6de6b permits implementation only. Actual numeric execution remains prohibited until full-source approval and actual gates. Parent owns source; independent agent reviews without concurrent repository edits.

## 1. Pure independent arithmetic and bounded checkpoint projection

- [ ] Add a pure independent forward/metric module and literal fixtures. Do not import the fitted kernel's predict/loss/selection functions or any RNG/teacher/data loader.
- [ ] Implement stable SiLU from a separately expressed formula, finite chunked scalar RMS/loss/Frobenius reductions with explicit rescaling, overflow/underflow detection and zero-reference policy. Relative error is null only for genuinely zero target; nonzero residual cannot silently become zero.
- [ ] Freeze scalar tolerance 1e-12+1e-10*abs(reference); array max-absolute and Frobenius criteria exactly as the design. Exact metadata/schema/ordering is never approximate.
- [ ] Freeze near-tie policy: tau_i=1e-12+1e-10*abs(saved guard MSE_i). Compare independently derived minimum against each competitor; a gap <= tau_i+tau_j is ambiguity. Report literal earliest minimum and ambiguity without replacing the saved checkpoint or certifying selection under ambiguity.
- [ ] Add a new projection reader without changing the old experiment's frozen parser/source. Validate every checkpoint ZIP/NPY header and exact payload length before any numeric view; materialize only G/U/D. No moment or schedule decoding. Native finite FP64, exact bounded shapes/C-order, no pickle or unrestricted np.load.
- [ ] TDD using tiny literal arrays/header fixtures only: independent scalar forward, scaling/physical conversion, streaming aggregation, zero/subnormal/overflow, selection/tie, malformed or extra members, projection before-view rejection and immutable ownership. No large malicious-header allocation probe. Run only new-path tests, strict mypy and Ruff; independent component review before integration.

## 2. Auditable all-case iterator and protocol

- [ ] Bind engineering audit8724353ed385a28349fc862127c78352f2e5f87f718e3befb9f90de5acb52714, actual585eb97, source review/prereg/marker/terminal outputs and original frozen response whitelist. Do not broaden any old loader or overwrite the consumed experiment.
- [ ] Assemble one case, independently derive fixed whole-center indices and training scales, and forward all17checkpoints on all calibration rows in chunks<=4096. Compare all saved metrics/journals/selection. Check selected physical parameters and nine stored prediction shards. Release arrays before next case. No original teacher, hidden truth, private reference, context, heldout, RNG or optimizer calls.
- [ ] Persist all6 outcomes including disagreements/ambiguities and partial failures. No retuning, new winner/model promotion or generalization claim.
- [ ] Exact byte/shard/memory inventory: prior base857668616bytes includes original1209byteglobalmarker, plus every audit marker/metadata/transport/temp/partial reserve. No raw/candidate duplication; combined<1GiB.
- [ ] New prereg/controller/defaultCLI and exclusive audit marker; full source/runtime/input hashes; actual sampled process+children preflight/typed transport, shared180s512MiB, BLAS1,2KiBaggregateoutput, actualdisk>=2GiB. Fail closed before data on missing review/gates. No retry.
- [ ] Fixture integration checks and independent full-source review; commit clean freeze. Implementation checkpoints never constitute permission for actual data/RNG/fit.

## 3. One actual audit after gates, terminal evidence and review

- [ ] Verify source and use only approved default CLI once after actual gates. Do not invoke internal stages manually. No experiment optimizer replay.
- [ ] Terminal checkpoint with full6x17 agreement/disagreement/ambiguity, scope, source digests, counters, resource limitations and final bytes before independent saved-outcome review.
- [ ] If metrics disagree, isolate defects with literal fixtures while preserving original outcome. If they agree, document genuine consumed-split discrepancy without attributing its cause; design a distinct fresh-owned study with sealed broad-domain validation before any new teachers/data. Universal/full-decoder/original-weight claims remain unproved; private final-model gate CLOSED.
