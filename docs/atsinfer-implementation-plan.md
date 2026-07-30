# ATSInfer Real Implementation Status and Action Plan

This document records the current source-level state of the ATSInfer work in this repository and lays out an implementation plan to close the gap between the current prototype and a functional ATSInfer-style hybrid CPU/GPU inference runtime.

## Current code status

### What is currently implemented

The repository currently contains a small isolated ATSInfer prototype under `src/atsinfer/`:

- `atsinfer-placement.cpp` implements a static tensor placement dynamic-programming solver over tensor profiles and a VRAM budget.
- `atsinfer-profiler.cpp` creates tensor profiles from `ggml_tensor` metadata using fixed heuristic formulas.
- `atsinfer-scheduler.cpp` implements a simplified dynamic transfer decision and a load-deviation reschedule trigger.
- `tests/test-atsinfer.cpp` exercises these components with synthetic tensor profiles.

These files are compiled into the `llama` target through `src/CMakeLists.txt`, but their logic is not connected to real model loading, real GGUF tensor placement, CUDA transfers, or the `llama_decode` execution path.

### What is integrated into llama.cpp today

The current integration into the main inference code is minimal:

- `src/CMakeLists.txt` adds the ATSInfer `.cpp` files to the `llama` library build.
- `src/llama-model.h` adds `llama_model::buft_tensor`, a per-tensor buffer-type map.
- `src/llama.cpp` forces `cparams.scheduler_async` to true when the model has devices.
- `tests/CMakeLists.txt` registers `test-atsinfer.cpp`.

No current code path applies the placement decisions from `atsinfer_compute_static_placement()` to actual `ggml_backend_buffer_type_t` choices during model loading. No current code path invokes the dynamic scheduler during inference.

### Current limitations

The current implementation should be treated as a component prototype, not as a complete ATSInfer implementation. The following major pieces are missing:

- Real hardware profiling for PCIe bandwidth and per-op/per-tensor latency.
- Integration with the GGUF model loader and tensor allocation path.
- Per-tensor backend placement applied to actual model weights.
- Runtime migration of tensors between CPU and GPU.
- Dedicated CUDA transfer stream and compute stream coordination for ATSInfer.
- CUDA events for dependency tracking between transfer and compute.
- Actual overlap of CPU compute, GPU compute, and host-to-device transfers.
- Tensor cache, eviction, pinning, and residency tracking.
- Online adaptive policy connected to measured runtime latency.
- Benchmark scripts and generated artifacts that reproduce reported results.

## Target architecture

A complete implementation should be organized around five cooperating subsystems:

1. **Profiler**: measures hardware and model execution characteristics.
2. **Placement solver**: computes initial tensor residency under VRAM constraints.
3. **Loader integration**: applies placement decisions to real GGUF tensors and backend buffers.
4. **Runtime scheduler and migration engine**: moves tensors asynchronously and schedules compute with overlap.
5. **Benchmark and validation harness**: verifies correctness, performance, and reproducibility.

## Action plan

### Phase 1: Establish reliable profiling data

#### Goals

- Replace fixed constants with measured values.
- Collect enough per-tensor/per-op data to drive placement decisions.
- Make profiling reproducible and cacheable.

#### Tasks

1. Add a hardware profiling backend that measures host-to-device and device-to-host bandwidth with pinned host memory and `cudaMemcpyAsync`.
2. Measure PCIe bandwidth for multiple transfer sizes and record median/percentile values.
3. Add CUDA event timing around representative GGML operations for CPU and GPU execution.
4. Collect tensor metadata from the real loaded model: name, size, type, layer, op role, MoE/non-MoE classification, backend compatibility, and expected access frequency.
5. Persist profile results in a small structured format keyed by model hash, quantization, device, driver/runtime version, and backend configuration.
6. Add a CLI/debug flag to print profiling summaries without running generation.

#### Expected output

- `atsinfer_profile_hardware()` reports measured bandwidth instead of a fixed default.
- `atsinfer_profile_tensors()` is backed by real model metadata and measured or calibrated latency estimates.
- Tests cover profile serialization/deserialization and fallback behavior when profiling is unavailable.

### Phase 2: Connect static placement to real model loading

#### Goals

- Turn the abstract placement map into actual tensor residency.
- Make `llama_model::buft_tensor` functional or replace it with a clearer placement structure.

#### Tasks

1. Identify the tensor loading path where `ggml_backend_buffer_type_t` is selected for model tensors.
2. Call the ATSInfer profiler and placement solver before final buffer assignment.
3. Apply `tensor_name -> backend` decisions to real tensors by selecting CPU, GPU, or split backend buffer types.
4. Ensure fallback behavior preserves existing llama.cpp placement when ATSInfer is disabled.
5. Add command-line parameters for enabling ATSInfer, setting VRAM budget, controlling profiling mode, and dumping placement decisions.
6. Add detailed logs that show tensor name, size, selected backend, expected latency reduction, and cumulative VRAM usage.
7. Validate that the final GPU allocation stays under the configured physical VRAM budget.

#### Expected output

- Model loading uses ATSInfer placement when explicitly enabled.
- Per-tensor placement decisions affect actual buffer allocation.
- A loaded model can report its real ATSInfer placement map.

### Phase 3: Implement tensor residency tracking and cache management

#### Goals

- Track which tensors are resident on CPU, GPU, or both.
- Support temporary GPU residency for tensors promoted by the dynamic scheduler.
- Avoid unsafe lifetime and synchronization behavior.

#### Tasks

1. Introduce an `atsinfer_tensor_state` structure with tensor identity, host pointer, device pointer, size, current residency, dirty state, last-use timestamp, and lock/pin state.
2. Add an `atsinfer_tensor_cache` with a fixed GPU memory budget and accounting for static and dynamic residents.
3. Implement cache insertion, lookup, eviction, and pinning APIs.
4. Define eviction policies: LRU baseline, layer-distance heuristic, and MoE expert priority.
5. Prevent eviction of tensors required by in-flight CUDA work using events or reference counts.
6. Add correctness checks that a tensor is resident on the required backend before compute dispatch.

#### Expected output

- Runtime has an authoritative view of tensor residency.
- Dynamic promotion can reserve GPU space safely.
- Eviction decisions are explicit, testable, and logged.

### Phase 4: Implement asynchronous migration

#### Goals

- Move tensors between CPU and GPU without blocking the compute stream unnecessarily.
- Prepare future tensors while current compute continues.

#### Tasks

1. Add an ATSInfer CUDA runtime layer that owns at least one compute stream and one transfer stream per CUDA device, or integrates cleanly with existing ggml CUDA streams.
2. Use pinned host memory or existing pinned staging buffers for host-to-device transfers.
3. Implement `atsinfer_migrate_to_gpu()` using `cudaMemcpyAsync` on the transfer stream.
4. Record a CUDA event after each migration completes.
5. Make compute wait only on the event for the specific tensor it needs, not on a global synchronization.
6. Add error handling for allocation failure, copy failure, and stream/event creation failure.
7. Add CPU fallback when a scheduled migration cannot be completed under memory or timing constraints.

#### Expected output

- Tensor promotion performs real asynchronous host-to-device copies.
- CUDA events encode tensor readiness.
- The compute path consumes migrated tensors only after their transfer event completes.

### Phase 5: Implement compute/transfer overlap and CPU/GPU pipeline

#### Goals

- Schedule future transfers during current CPU/GPU compute windows.
- Avoid making migration overhead fully visible on the critical path.

#### Tasks

1. Extend the runtime scheduler to inspect the upcoming graph/layer sequence and identify future CPU-resident tensors that may be beneficial to promote.
2. Estimate transfer cost and available overlap windows using measured runtime timing, not only static formulas.
3. Enqueue migrations early enough to overlap with CPU compute or GPU compute from independent graph sections.
4. Use CUDA events to enforce only required dependencies.
5. Integrate with ggml backend scheduler split decisions so that a tensor promoted to GPU is actually used by GPU kernels.
6. Track exposed transfer cost, hidden transfer cost, and realized overlap in runtime metrics.
7. Add debug traces that can reconstruct the compute/transfer timeline.

#### Expected output

- Transfers can overlap with useful compute.
- Runtime metrics distinguish hidden transfer time from blocking wait time.
- Scheduler decisions are observable and reproducible.

### Phase 6: Implement adaptive runtime policies

#### Goals

- Make the scheduler respond to changing workload, prompt length, batch size, active MoE experts, and system load.
- Replace isolated reschedule tests with runtime decisions.

#### Tasks

1. Feed measured per-token latency, per-layer timing, migration wait time, and cache hit/miss statistics into `ATSInferRescheduler`.
2. Trigger rescheduling when measured latency diverges from expected latency beyond a configured threshold.
3. Add rate limits and hysteresis to prevent thrashing.
4. Recompute dynamic promotion candidates using current cache state and upcoming graph requirements.
5. Add MoE-specific policies that prioritize active experts and avoid promoting inactive experts.
6. Add guardrails to disable dynamic promotion when it hurts latency over a moving window.
7. Expose runtime counters through logs or an introspection API.

#### Expected output

- Adaptive policy is part of inference, not only a unit-tested class.
- Runtime can change promotion choices based on measured behavior.
- Bad promotion decisions can be detected and suppressed.

### Phase 7: Validation and correctness testing

#### Goals

- Prove the implementation preserves model correctness.
- Catch synchronization and residency bugs early.

#### Tasks

1. Add unit tests for placement budget accounting, MoE classification, cache eviction, and migration state transitions.
2. Add integration tests that load a small GGUF model with ATSInfer enabled and disabled.
3. Compare logits/token outputs between baseline and ATSInfer runs under deterministic settings.
4. Run stress tests with small artificial VRAM budgets to force migration and eviction.
5. Add CUDA synchronization tests that detect use-before-copy and event ordering bugs.
6. Add tests for fallback behavior when CUDA is unavailable or allocation fails.

#### Expected output

- CI can verify ATSInfer correctness on CPU-only and CUDA-capable environments where available.
- Deterministic tests catch placement and migration regressions.

### Phase 8: Benchmark and reproducibility harness

#### Goals

- Replace static markdown benchmark tables with reproducible scripts and generated artifacts.
- Quantify real gains and regressions.

#### Tasks

1. Add benchmark scripts that run baseline llama.cpp, static ATSInfer, and dynamic ATSInfer modes with identical prompts and sampling settings.
2. Capture command line, git commit, model path/hash, GPU name, driver version, CUDA runtime version, CPU model, memory size, OS, and environment variables.
3. Store raw logs and parsed JSON/CSV metrics.
4. Track prefill throughput, decode throughput, TPOT, total latency, VRAM usage, RAM usage, migration bytes, cache hits/misses, exposed transfer time, and overlap ratio.
5. Add a parser that turns raw `llama-cli` logs into machine-readable metrics.
6. Add documentation that explains how to reproduce each reported table.
7. Require benchmark claims in docs to reference generated artifacts.

#### Expected output

- Benchmarks become auditable and repeatable.
- README performance claims can be supported or corrected based on generated data.

## Suggested implementation order

1. Add feature flags and logging scaffolding.
2. Implement real profiling and placement dump mode.
3. Apply static placement to real model loading.
4. Add residency tracking and cache accounting.
5. Add asynchronous migration with CUDA streams/events.
6. Integrate dynamic scheduler into inference.
7. Add adaptive policies.
8. Add reproducible benchmark harness and update documentation.

## Acceptance criteria for a complete implementation

The project should not claim full ATSInfer integration until all of the following are true:

- ATSInfer can be enabled and disabled through explicit runtime configuration.
- Static placement decisions are applied to actual GGUF model tensors.
- The runtime reports actual tensor residency and VRAM accounting.
- Dynamic promotion performs real asynchronous tensor migration.
- CUDA streams and events are used to overlap transfer and compute.
- The scheduler runs during inference and adapts based on measured runtime behavior.
- Correctness is validated against baseline outputs.
- Benchmark results are generated by reproducible scripts and stored as artifacts.

## Documentation cleanup required

Until the missing runtime pieces are implemented, documentation should avoid stating that the repository has complete ATSInfer integration. It should instead describe the current status as:

> An isolated ATSInfer prototype containing a static placement solver, heuristic profiler, simplified dynamic scheduler, and synthetic unit tests. Runtime integration with GGUF loading, CUDA migration, tensor cache, and adaptive inference scheduling is still pending.
