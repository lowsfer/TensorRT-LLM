<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# KVCacheManagerV2 C++ Migration Plan for Main and PR #15633

## Scope and comparison points

This plan covers only behavior implemented under
`tensorrt_llm/runtime/kv_cache_manager_v2/`. PyExecutor, model-specific cache
managers, disaggregated-serving orchestration, and other consumers are not
migration targets, although selected consumer tests are included in the final
end-to-end validation.

The source comparisons are:

- PR #14047 base: `1aa232a0fc58a615bb16a9c3a2f36e79cd5b42d0`
- Main: `790c8660c03cd04eace398b568ec9f82b074bf0d`
- Main plus PR #15633: `db2972bed2ad97a194fe86992be030049992b933`

Each migration change below should be independently reviewable and should
leave both the Python and C++ backends usable. Unless a test is explicitly
Python-only, run it once with
`TLLM_KV_CACHE_MANAGER_V2_BACKEND=python` as the reference and once with
`TLLM_KV_CACHE_MANAGER_V2_BACKEND=cpp` as the implementation under test.

## Change 1: Explicit initial pool ratio

Status: implemented and verified on 2026-07-01. `TestInitRatioConfig` passes
all 19 cases with both the Python reference backend and the C++ backend.

### Implementation

- Add `initialPoolRatio` to the C++ `KVCacheManagerConfig` and nanobind API.
- Validate that its length equals the number of pool groups, every value is
  positive, and the sum is within `1e-6` of `1.0`.
- When present, use it instead of `typicalStep`, constraints, or fallback
  sizing when creating every cache level.
- Do not derive initial minimum slots from constraints when an explicit ratio
  is present. Constraints remain relevant to later runtime operations but must
  not change the explicit initial partition.
- Expose the resulting ratio through the existing backend-neutral
  introspection helper.

### Targeted tests

- Existing `TestInitRatioConfig` tests in
  `tests/unittest/kv_cache_manager_v2_tests/test_kv_cache_manager_v2.py`.
- Convert `test_initial_pool_ratio_overrides_typical_step_and_constraints` to
  use `_introspection.current_gpu_ratio()` instead of a Python-private member.
- Add invalid-value coverage for an empty ratio, non-positive values, a sum
  outside tolerance, and a length mismatch.
- Verify identical slot counts and ratios for Python and C++ backends.

### Acceptance command

```bash
pytest tests/unittest/kv_cache_manager_v2_tests/test_kv_cache_manager_v2.py::TestInitRatioConfig -vs
```

## Change 2: `clamp_max_seq_len_for_mem` low-memory parity

Status: implemented and verified on 2026-07-01. The five targeted low-memory,
sliding-window, and multi-pool-group cases pass with both backends.

### Implementation

- Return `0` when reserving the minimum sequence for the other batch members
  makes any pool group's remaining slots negative.
- Return `0` when the current request cannot fit one block.
- Preserve the current binary-search result for all feasible inputs.
- Avoid relying on debug-only assertions for these runtime outcomes.

### Targeted tests

- Add a backend-neutral test class covering zero capacity, a single feasible
  block, multiple pool groups, sliding-window lifecycles, and a batch size that
  consumes the remaining capacity.
- Compare Python and C++ results for the same generated configurations.

### Acceptance command

```bash
pytest tests/unittest/kv_cache_manager_v2_tests/test_kv_cache_manager_v2.py -k clamp_max_seq_len_for_mem -vs
```

## Change 3: Combined scratch-slot readiness synchronization

Status: implemented and verified on 2026-07-01. All seven `TestScratchReuse`
cases, including the cross-stream ready-event test, pass with both backends.

### Implementation

- After newly allocated slots are combined with slots detached from excess
  scratch locks, wait for every combined slot's ready event before constructing
  new pages.
- Keep the existing earlier wait for newly allocated slots; the second wait
  covers detached scratch slots and must occur before they are consumed.
- Do not introduce a device-wide synchronization.

### Targeted tests

- Extend `TestScratchReuse` with a delayed ready-event case that forces a slot
  to move from the scratch set into a normal page.
- Run the existing scratch toggle, slot-count, shared-slot-ID, chunk-size, and
  rewind-length tests on both backends.
- Use the existing kernel-delay hooks to verify cross-stream ordering without
  relying only on final values.

### Acceptance command

```bash
pytest tests/unittest/kv_cache_manager_v2_tests/test_kv_cache_manager_v2.py::TestScratchReuse -vs
```

## Change 4: Host-memory THP selection and parallel prefault

Status: implemented and verified on 2026-07-01. The dedicated C++ HostMem
suite passes all nine page-mode, prefault fallback/error, resize, CUDA
registration, and GPU round-trip cases. The SM100 full wheel build and
editable install also complete successfully.

### Implementation

- Honor `TLLM_KV_CACHE_MANAGER_V2_THP`; use `MADV_HUGEPAGE` by default and
  `MADV_NOHUGEPAGE` when it is set to `0`.
- Honor `TLLM_KV_CACHE_MANAGER_V2_PREFAULT_THREADS`; `0` disables prefault.
- Prefault initial allocations in 512 MiB chunks with
  `MADV_POPULATE_WRITE` before CUDA host registration.
- Fall back to touching the chunk on `EINVAL` or `ENOSYS`.
- Convert `ENOMEM` into `HostOOMError`; propagate other errors with errno.
- Match Python behavior on resize: reapply the selected madvise mode and
  register again, but do not run the parallel prefault path again.

### Targeted tests

- Add host-memory tests for both THP modes and disabled prefault.
- Add injectable/system-call wrappers so `EINVAL`, `ENOSYS`, and `ENOMEM` can
  be tested deterministically without exhausting the node.
- Verify that CUDA registration succeeds and a GPU can read/write the pinned
  allocation after prefault.

### Acceptance command

```bash
pytest tests/unittest/kv_cache_manager_v2_tests -k "host_mem or prefault or hugepage" -vs
```

## Change 5: Statistics types, configuration, and manager-level API

Status: implemented and verified on 2026-07-01. The dedicated C++ stats suite
passes all four delta, manager commit/reset, request-ID tracking, and disabled
accounting tests. The backend-neutral API suite passes all three cases with
both backends, and the PyExecutor `stats_disabled` acceptance test passes with
both backends.

### Implementation

- Add `enableStats` to the C++ config and nanobind constructor.
- Add C++ equivalents of `KVCacheStatsDelta` and
  `KVCacheIterationStatsDelta`, including add, subtract, clear, copy, and empty
  operations.
- Add committed global stats and per-lifecycle iteration accumulators to the
  manager.
- Add dirty and excluded request-ID tracking.
- Add `expectedPromptLength` to cache creation and track the transition from
  context allocation to generation allocation.
- Bind all manager and cache APIs declared in `__init__.pyi`. Returning the
  existing Python dataclasses through an adapter is acceptable, but their
  observable fields and reset semantics must match.

### Targeted tests

- `test_stats_disabled_suppresses_v2_accounting`
- New direct tests for delta arithmetic, manager commit, iteration reset,
  dirty IDs, excluded IDs, and `expected_prompt_length` boundaries.
- Nanobind signature and return-type tests.

### Acceptance command

```bash
pytest tests/unittest/kv_cache_manager_v2_tests/test_kv_cache_stats_behavior.py -k stats_disabled -vs
```

## Change 6: Pending allocation and reuse statistics

Status: implemented and verified on 2026-07-01. All 13 allocation, rollback,
generation-boundary, reuse, SWA, and pool-group behavior tests pass with both
the Python reference backend and the C++ backend. The C++ stats suite also
passes pending-segment rollback coverage under debug assertions.

### Implementation

- Port pending allocation segments and reversible range subtraction.
- Record new/total/missed allocation at resize, excluding stale and scratch
  ranges.
- Record full and partial reuse per attention lifecycle.
- Count the private copy created for partial reuse as allocation, but count a
  miss only when no reusable source page exists.
- Commit pending request stats only when requested; discard them on close,
  cancellation, or explicit rollback.
- Keep SSM lifecycles out of attention reuse metrics.

### Targeted tests

- Context and generation allocation reporting.
- Reverted generation and unscheduled pending allocation behavior.
- Chunked context generation boundaries.
- Full reuse, partial leaf reuse, and block-reuse-disabled cases.
- SWA stale-prefix and scratch exclusion.

### Acceptance command

```bash
pytest tests/unittest/kv_cache_manager_v2_tests/test_kv_cache_stats_behavior.py -vs
```

## Change 7: Transfer and last-tier drop statistics

Status: implemented and verified on 2026-07-01. The C++ backend passes the
full 13-case stats behavior suite, including partial-reuse intra-device copy
counts and bytes. A dedicated two-tier C++ debug test verifies exact offload,
onboard, and last-tier drop recorder counts, and the full C++ stats suite
passes all seven cases.

### Implementation

- Propagate a migration recorder through new-slot allocation,
  `prepareFreeSlots`, batched lock-to-GPU, and batched migration.
- Record GPU-to-lower-tier offload, lower-tier-to-GPU onboard, and same-tier
  GPU copy blocks and bytes.
- Add `iterHostDroppedBlocks` and `iterHostDroppedBytes`.
- Invoke the drop recorder before last-level evicted pages lose their final
  strong references in both normal free-slot preparation and `forceEvict`.
- Attribute all counters to the correct attention lifecycle and pool-group
  slot size.

### Targeted tests

- Add direct two-tier tests for offload, onboard, intra-device copy, and
  last-tier drop counts and bytes.
- Verify that defragmentation does not count as a user-visible transfer.
- Verify that dropping an SSM page does not enter attention statistics.
- Re-run last-level eviction and multi-tier migration tests.

### Acceptance command

```bash
pytest tests/unittest/kv_cache_manager_v2_tests -k "onboard or offload or intra_device or host_dropped or last_level" -vs
```

## Change 8: Per-iteration peak block statistics

Status: implemented and verified on 2026-07-01. The native C++ manager tracks
available, unavailable, and evictable peaks independently per cache level and
pool group. Eight C++ stats tests pass under debug assertions, including
two-tier consecutive-reset interval coverage. The combined C++ backend stats
API and behavior suites pass all 16 cases.

### Implementation

- Add `PoolGroupPeakBlockStats` with available, unavailable, and evictable
  values.
- Track peaks independently for every cache level and pool group.
- Refresh peaks at the same commit points as Python.
- Implement `getAndResetIterationPeakBlockStats(cacheLevel)` so the returned
  interval is complete and the next interval starts from the current snapshot.
- Add nanobind exports and stub parity.

### Targeted tests

- Existing
  `tests/unittest/executor/test_stats_serializer.py::TestKVCacheV2PeakBlockStats`-style coverage.
- Add direct tests with multiple cache levels and pool groups.
- Verify two consecutive get-and-reset intervals and an interval with no
  activity.

### Acceptance command

```bash
pytest tests/unittest/executor/test_stats_serializer.py -k peak_block_stats -vs
```

## Change 9: Event model, queue, hash compatibility, and public API

Status: migrated to native C++ on 2026-07-02. The C++ manager owns the event
model, bounded queue, lifecycle registry, coalescing, batching, blocking reads,
and hash selection. V2 events reuse the radix tree's native SHA-256 block key
directly and identify it as `v2_sha256` (or `v2_sha256_64`); v1 events preserve
the legacy 64-bit block-key hashing algorithm. The Python event manager remains
available only with the explicit Python KV-cache-manager backend.

### Implementation

- Implement or adapt all event data types and the bounded event queue.
- Preserve stored-event coalescing, removed-event batching, event IDs,
  blocking reads, iteration flush, per-layer-group windows, and attention-DP
  gather callbacks.
- Support legacy v1 plus native `v2_sha256` and `v2_sha256_64` output modes,
  reporting the SHA-256 algorithm in emitted events (byte-identical to the
  Python backend's `hashlib.sha256` block keys).
- Add golden tests for SHA-256 root/chained block hashes and legacy v1 hashes.
- Keep Python interaction only at the optional attention-DP gather callback;
  event payloads are pickleable so MPI object gather remains supported.

### Targeted tests

- Pure event serialization, queue capacity, blocking, coalescing, and gather
  tests from `test_kv_cache_event_manager.py`.
- Cross-backend hash golden vectors for integer tokens, multimodal tokens,
  LoRA IDs, cache salts, and parent chains.

### Acceptance command

```bash
pytest tests/unittest/kv_cache_manager_v2_tests/test_kv_cache_event_manager.py -k "serialization or coalesces or hash or gathers" -vs
```

## Change 10: Event emission from cache, radix tree, and storage

Status: implemented and verified on 2026-07-01. C++ emits stored events for
new blocks and lifecycle refills, removed events for lifecycle loss and every
radix-tree detach path, and deduplicated cache-level updates for real
cross-tier migrations. The full C++ event suite passes all 35 cases, the
stats API/behavior regressions pass all 16 cases, the debug C++ stats suite
passes all eight cases, and the complete C++ manager regression passes 83
cases with 12 expected performance skips.

### Implementation

- Emit stored events when a block or an additional lifecycle becomes
  committed.
- Emit removed events for subtree removal, partial-block sibling replacement,
  lifecycle removal, tail pruning, clear, and last-tier drop.
- Emit cache-level updated events once per block/lifecycle transition during
  migration, excluding defragmentation and orphaned blocks.
- Preserve ordering: pending removals must be flushed before later stored or
  updated events.
- Ensure orphan blocks and detached parent links cannot dereference stale
  nodes while generating events.

### Targeted tests

- All stored/reused-prefix tests in `test_kv_cache_event_manager.py`.
- All partial-lifecycle and whole-block removal tests.
- Last-level drop and cache-level migration updated-event tests.
- Clear-tree and partial-sibling replacement regression tests on both
  backends.

### Acceptance command

```bash
pytest tests/unittest/kv_cache_manager_v2_tests/test_kv_cache_event_manager.py -vs
```

## Per-change quality gate

Every change must satisfy all of the following before the next change starts:

1. Its targeted Python reference tests pass.
2. The same targeted tests pass with the C++ backend.
3. Relevant C++ unit tests pass under debug assertions where practical.
4. The full `test_kv_cache_manager_v2.py` suite has no new failures.
5. Public signatures in `__init__.pyi`, Python, and nanobind agree.
6. No Python-private implementation object is exposed only to make a test
   pass; use stable introspection or public result types.

## Final validation

### Backend parity suite

Run the complete KVCacheManagerV2 unit suite with both backends:

```bash
TLLM_KV_CACHE_MANAGER_V2_BACKEND=python \
pytest tests/unittest/kv_cache_manager_v2_tests -vs

TLLM_KV_CACHE_MANAGER_V2_BACKEND=cpp \
pytest tests/unittest/kv_cache_manager_v2_tests -vs
```

Run stats serialization and public event API tests with the C++ backend:

```bash
TLLM_KV_CACHE_MANAGER_V2_BACKEND=cpp \
pytest tests/unittest/executor/test_stats_serializer.py \
       tests/unittest/llmapi/test_llm_kv_cache_events.py -vs
```

### C++ tests

Run the KVCacheManagerV2 C++ unit-test target or its matching CTest filter,
including a debug build for ownership and state assertions:

```bash
ctest --test-dir cpp/build -R "KVCacheManagerV2|kv_cache_manager_v2" --output-on-failure
```

### GPU end-to-end tests

Set `LLM_MODELS_ROOT` and force the C++ backend for all E2E runs.

1. Run the KV-cache V2 scheduler comparison tests:

   ```bash
   TLLM_KV_CACHE_MANAGER_V2_BACKEND=cpp \
   LLM_MODELS_ROOT=/scratch.trt_llm_data/llm-models \
   pytest tests/integration/defs/kv_cache/test_kv_cache_v2_scheduler.py -vs
   ```

2. Run one single-GPU PyTorch LLM request mix with prefix reuse, chunked
   context, cancellation, and generation. Compare generated token IDs and
   reuse statistics against the Python backend.

3. Run a two-tier GPU+host configuration that forces offload, onboard, and
   last-tier drops. Validate generated token IDs, transfer counters, peak
   counters, and the stored/updated/removed event sequence.

4. Run a four-GPU tensor-parallel request mix on GB200 with block reuse and
   event collection enabled. Compare output token IDs and aggregate metrics
   with the Python backend.

5. Run the KV pool rebalance accuracy test:

   ```bash
   TLLM_KV_CACHE_MANAGER_V2_BACKEND=cpp \
   LLM_MODELS_ROOT=/scratch.trt_llm_data/llm-models \
   pytest tests/integration/defs/accuracy/test_kv_pool_rebalance_accuracy.py -vs
   ```

### Validation status on 2026-07-01

- The complete KVCacheManagerV2 unit directory passes with both backends:
  138 passed and 12 expected performance skips for C++, and 138 passed and
  12 expected performance skips for Python.
- A single-GPU Llama-3.2-1B run with shared-prefix reuse, chunked prefill,
  GPU+host cache levels, and deterministic decoding produced identical token
  IDs with the C++ and Python backends.
- A four-GPU Llama-3.2-1B TP=4 run with shared-prefix reuse and event
  collection produced identical token IDs, event ordering, event types, and
  public block hashes with the C++ and Python backends.
- The DeepSeekV4 cache-manager suite passes all 53 cases with both backends.
  The four-GPU DeepSeek-V4-Flash `test_auto_dtype` integration smoke also
  passes with the C++ backend, including MMLU and GSM8K evaluation.
- The scheduler comparison's basic V1/V2 cases pass. Its token-budget case
  produces different V1 and V2 text, but the same failure reproduces with the
  Python V2 backend; this is therefore not a C++ migration delta.
- The pool-rebalance accuracy test passes in no-overlap mode. Its overlap case
  changes greedy output at the same prompt, token index, and token values with
  both C++ and Python V2 backends, so that failure is also a Python-reference
  baseline issue rather than a C++ migration delta.
- The model root used on the allocated GB200 node was
  `/scratch.trt_llm_data/llm-models`.

### Final acceptance criteria

- Python and C++ produce identical allocation/reuse/transfer counters for the
  deterministic parity scenarios.
- Event payloads, ordering, layer-group IDs, and hashes match.
- Generated token IDs match in single-GPU, GPU+host, and four-GPU runs.
- No host-registration regression is observed with THP enabled or disabled.
- No sanitizer/debug assertion, leaked live cache, stale page-index pointer,
  or CUDA stream-ordering error is observed.
