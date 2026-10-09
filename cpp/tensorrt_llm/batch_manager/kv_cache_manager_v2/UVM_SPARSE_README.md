# UVM-backed sparse KV cache for DeepSeek-V4 (experiment)

Status: **experimental, off by default, not upstream.** Lives on branch
`kvcm2-sparse-uvm` (fork `lowsfer/TensorRT-LLM`). Scoped to one model and one platform
family; hacks and env knobs are deliberate.

## What it does

DeepSeek-V4 decoding at `compress_ratio == 4` reads only a small, per-step subset of its
compressed history, but the whole history sits in HBM so a request's memory bill still
grows with context length.

This experiment backs the pool holding that history with `cudaMallocManaged` instead of
CUDA VMM, and lets the driver decide what is GPU-resident. On a hardware-coherent
(Grace C2C) platform, `cudaMemAdviseSetAccessedBy` switches the driver from
fault-and-migrate to **access-counter migration**: hot pages move into HBM on access
counts, cold pages are evicted back to Grace under memory pressure.

Nothing in KVCM2 moves data, tracks residency, or maintains a device-side cache. The
allocation stays at cache level 0; only its physical backing changes.

### Why this and not the explicit-copy design

The alternative — upstream PR #19580 plus follow-ups — pins history in host memory at
cache level 1, teaches the lock path about tiers, and adds a per-request device cache
with an LRU and a fetch kernel. This branch **reverts** that (first commit) and trades
all of it for the driver's migration policy.

The two diverge at the storage layer rather than composing, which is why the revert is
there. If this experiment loses, the revert is what gets dropped.

## Prerequisites

| requirement | why | how to check |
| --- | --- | --- |
| Grace C2C coherent platform (GH200/GB200) | access-counter migration needs the GPU to walk host page tables | enforced at construction; see below |
| access counters enabled in `nvidia_uvm` | without them migration falls back to fault-and-migrate | `modinfo nvidia_uvm`, and the startup log |
| HBM for the resident indexer at the chosen ratio | the indexer cannot be offloaded | see *Sizing* |

On a non-coherent device, construction **throws** rather than falling back. A silent
fallback would run at fault-and-migrate speed and produce a number that looks like a
measurement of C2C but is not.

## Enabling it

```bash
export TLLM_KVCM2_UVM_SPARSE=1        # off unless exactly "1"
export TLLM_KVCM2_UVM_OVERSIZE=4.0    # slot-count multiplier, default 4.0, must be >= 1.0
```

**No Python-side change is needed.** In particular this does *not* require
`enable_kv_cache_offload`, which belongs to the explicit-copy design and still raises
`NotImplementedError`. Run DeepSeek-V4 normally with the variables set.

## Confirming it is actually active

At warning level, so it appears without raising the log level:

```
KVCM2 UVM: experimental managed-memory path ACTIVE (TLLM_KVCM2_UVM_SPARSE=1)
KVCM2 UVM:   device=0 coherent(pageableMemoryAccessUsesHostPageTables)=1
KVCM2 UVM:   oversize ratio=4.00 (TLLM_KVCM2_UVM_OVERSIZE)
KVCM2 UVM:   uvm_perf_access_counter_mimc_migration_enable=-1
KVCM2 UVM:   uvm_perf_access_counter_momc_migration_enable=-1
KVCM2 UVM:   uvm_perf_access_counter_batch_count=256
KVCM2 UVM:   pool backed by cudaMallocManaged: slotSize=... numSlots=... bytes=... (N.NN GiB)
```

Read it as:

- The environment block appears **once per process**. If it is absent, the path is off.
- `-1` on the counter parameters means the driver's default policy, `0` means disabled,
  `1` or more means enabled. A literal `0` produces an extra warning, because the run
  still completes and merely looks slow.
- `<unreadable>` means the sysfs file is missing — usually a different driver layout
  rather than the feature being off. Worth investigating before trusting a result.
- The pool line repeats **per managed pool**. For DeepSeek-V4 expect it for the COMPRESS
  pool and, under nvfp4, the block-scale pool. More than that means the role match
  caught something unintended.

## Which pool is selected, and why

Selection is per pool, keyed on the roles the pool carries:

- **Managed**: any role matching the prefix `deepseek_v4_compress` — covers
  `deepseek_v4_compress`, `deepseek_v4_compress_block_scale`, and the nvfp4 padding roles
  of both, which are sized to coalesce into those two pools.
- **Stays in HBM**: `deepseek_v4_indexer_compress`. The lightning indexer scores every
  candidate block each step, so it has no working set to migrate and would only compete
  with COMPRESS for residency. This is the buffer HiSparse deliberately keeps resident.

Note the prefix does not match `deepseek_v4_indexer_compress`, which begins
`deepseek_v4_indexer`.

With the default (non-offload) configuration the sparse layer is
`[COMPRESS, INDEXER_COMPRESS]`, and `createStorageConfig` groups buffers by
`(lifecycle, expanded size)`. Those two sizes differ, so COMPRESS already gets its own
pool. **If a configuration ever makes them equal**, they coalesce into one pool and the
indexer would silently go managed — construction throws instead, naming both roles.

## Sizing

The oversize ratio multiplies the **pool group's** slot count, not one pool's byte
capacity. A slot id spans every pool in a group: one slot is a block's COMPRESS page
*and* its INDEXER_COMPRESS page, and selection cannot score history it has no index for.
So the indexer grows with the ratio by necessity.

Per token of history held, with nvfp4 compress and an fp8 indexer:

| buffer | bytes/token | at ratio R |
| --- | --- | --- |
| COMPRESS | 288 | managed; only the working set in HBM |
| COMPRESS_BLOCK_SCALE | 36 | managed; only the working set in HBM |
| INDEXER_COMPRESS | 132 | **R × 132 B/token of real HBM** |

Against 456 B/token of HBM for the same history without the experiment. The ceiling on R
is therefore the indexer: at R=4 it needs `4 × 132` B/token resident, plus whatever
working set the managed pools pull in. **If that exceeds the GPU tier quota, allocation
fails late rather than at config time** — check it before a long run.

Derivation of the constants: `head_dim = 512` (`DIM_NOPE 448 + DIM_ROPE 64`),
`NVFP4_COMPRESS_RESIDUAL_DIM = 64`, `NVFP4_VECTOR_SIZE = 16`, `index_head_dim = 128`,
`DEEPSEEK_V4_SPARSE_RATIO = 4`. Under `use_fp8_ds_mla` instead, COMPRESS is
`TOKEN_BYTES = 584` B/token with scales in the footer and there is no separate scale
buffer.

## Guards — exceptions you should expect

All of these are deliberate. Each protects an assumption whose violation would otherwise
produce a plausible-looking but meaningless result.

| condition | behaviour |
| --- | --- |
| device does not use host page tables | throws at pool construction |
| an offloadable and a resident role share a pool | throws at pool construction, naming both |
| `resize()` with a quota different from the current one | throws |
| `adjust()` while a managed pool exists | throws |
| pool asked to grow past its initial allocation | throws, naming `TLLM_KVCM2_UVM_OVERSIZE` |
| `TLLM_KVCM2_UVM_OVERSIZE < 1.0` | throws |

The `resize()` check sits **ahead of** the `try` block that converts failures into a
`false` return, so it propagates rather than being reported as "the quota did not fit".

The experiment assumes a **static quota and static pool ratios**. Both guards exist to
make that assumption fail loudly if the deployment violates it.

## Building and deploying

```bash
cd cpp/build
cmake --build . --target bindings -j$(nproc)
cp tensorrt_llm/libtensorrt_llm.so       ../../tensorrt_llm/libs/
cp tensorrt_llm/thop/libth_common.so     ../../tensorrt_llm/libs/
cp tensorrt_llm/nanobind/bindings.cpython-*.so ../../tensorrt_llm/
```

For a fast syntax check without producing anything loadable:

```bash
cmake --build . --target tensorrt_llm_batch_manager_static -j$(nproc)
```

## Where the code is

| file | what changed |
| --- | --- |
| `storage/core.h`, `storage/core.cpp` | `UvmSlotPool`; `poolUsesUvm` / `slotDescHasUvmPool`; `uvmOversizeRatio`; per-pool dispatch in `GpuPoolGroup`; group oversize in `GpuCacheLevelStorage`; the startup logging |
| `storageManager.h` | `hasUvmPool(level)` / `anyUvmPool()` |
| `kvCacheManager.cpp` | the `resize()` and `adjust()` guards |

Everything else on the branch is the revert of #19580.

## What to measure

The claim is **capacity**, not raw speed: more history per GB of HBM at acceptable
latency. Suggested axes:

- Max context / max concurrency before OOM, against the same build with
  `TLLM_KVCM2_UVM_SPARSE=0`.
- Per-token latency at matched concurrency — migration should cost something; the
  question is how much.
- Sweep `TLLM_KVCM2_UVM_OVERSIZE` (1.0 means no oversubscription, so nothing migrates
  and it should behave like the baseline apart from allocator differences; that is a
  useful control).
- Watch for thrashing at high R: the driver has no notion of the selection pattern, so a
  working set larger than HBM will cycle.

## Known limitations

- **No admission control.** The driver decides residency, so there is no reservation and
  no bound on thrashing. `resume()` cannot fail early the way it does on the VMM path.
- **Migration granularity is the driver's, not ours.** Access counters move large
  granules; a granule holds several consecutive block ordinals across all coalesced
  layers, so useful fraction depends on how clustered the selection is.
- **Device-to-host movement is pressure-driven only.** Per published Grace Hopper work
  the Grace CPU has no access counters, so pages return to host under memory pressure
  rather than by counting. The oversize ratio is what creates that pressure — at R=1
  nothing migrates.
- **Access-counter migration of *managed* memory is documented as an implementation
  detail** of current systems rather than a guarantee (CUDA Programming Guide, Unified
  Memory, footnote on `cudaMemAdviseSetAccessedBy`). It may change across driver or CUDA
  versions. That is why the knob exists and why the startup log prints the driver
  parameters.
- **`cudaMemAdviseSetPreferredLocation` is deliberately not set.** Pointing it at the
  host *prevents* counter migration. Do not "fix" its absence.

## If it looks slow

In order:

1. Is the environment block in the log at all? If not, the path is off.
2. Is `coherent=1`? If not, construction should have thrown — if it did not, the build is
   stale.
3. Are the counter parameters `0`? Then migration is disabled at the driver.
4. Is the oversize ratio `1.0`? Then there is no pressure and nothing migrates.
5. How many pool lines appeared? More than COMPRESS and block scale means the role match
   is too broad.
6. Is the working set simply larger than HBM? Lower R and compare.
