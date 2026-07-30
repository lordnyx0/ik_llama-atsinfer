# ATSInfer Architecture Mapping & Implementation Audit in ik_llama.cpp

This document describes the technical architecture mapping, implementation scope, and audited status of **ATSInfer** (*Automated Tensor Scheduling for Hybrid CPU-GPU LLM Inference on Consumer Devices*, arXiv:2607.10183v2) within `ik_llama.cpp`.

> **Superseded in part.** The mechanisms described below as standalone have since been wired into
> the loader, the context and the decode loop, and measured end to end. None of them improves
> decode throughput on this hardware, for reasons that are structural rather than defects. See
> **[atsinfer-findings.md](atsinfer-findings.md)** for the measurements, the root causes, and what
> to do if the work is picked up again. Read that document first.

---

## 1. Audited Implementation Scope

The codebase contains two distinct levels of implementation:

### A. Integrated Runtime Pipeline Modifications
- **Asynchronous Scheduling (`cparams.scheduler_async = 1`)**: Automatically forced in `src/llama.cpp` for CUDA devices, enabling asynchronous graph scheduling via native `ggml` CUDA backend streams.
- **Granular MoE Buffer Overrides (`-ncmoe`)**: Configured in `src/llama-load-tensors.cpp` to route MoE expert weights (`ffn_exps`) to Pinned Host DRAM (`CUDA_Host`) while retaining Attention projections and LayerNorms in GPU VRAM (`CUDA0`).
- **Physical VRAM Budget Enforcement (`--fit --fit-margin 256`)**: Prevents Windows WDDM CUDA System Memory Paging (which previously collapsed decode down to 7.95 tok/s under VRAM overflow).

### B. Standalone ATSInfer Research Prototype (`src/atsinfer/`)
- **`atsinfer-placement.cpp`**: Implements 3D Knapsack DP tensor placement algorithm under physical VRAM budgets ($M \le 10.5\text{ GiB}$).
- **`atsinfer-profiler.cpp`**: Implements heuristic tensor operator and hardware profiler.
- **`atsinfer-scheduler.cpp`**: Implements dynamic transfer decision logic and rate-limited reschedulers.
- **Audit Note**: The `src/atsinfer/` standalone module is compiled into the `llama` library and validated via unit tests (`tests/test-atsinfer.cpp`). Its placement decisions are currently evaluated via synthetic test routines rather than directly driving `ggml_decode` runtime kernel graph dispatch or dynamic H2D CUDA memcpy loops.

---

## 2. Codebase Structure Map

| Component | Files | Execution Context | Status |
| :--- | :--- | :--- | :--- |
| **Standalone Profiler** | `src/atsinfer/atsinfer-profiler.h`, `.cpp` | Unit tests (`test-atsinfer.exe`) | Implemented (Heuristic) |
| **Knapsack Placement Solver** | `src/atsinfer/atsinfer-placement.h`, `.cpp` | Unit tests (`test-atsinfer.exe`) | Implemented (Standalone DP) |
| **Dynamic Rescheduler** | `src/atsinfer/atsinfer-scheduler.h`, `.cpp` | Unit tests (`test-atsinfer.exe`) | Implemented (Decision Logic) |
| **Runtime Async CUDA Scheduling** | `src/llama.cpp` | `llama_init_from_model` / Runtime | **Active in Runtime** |
| **MoE Tensor Buffer Overrides** | `src/llama-load-tensors.cpp` | `llama_model_loader` / Runtime | **Active in Runtime** |
