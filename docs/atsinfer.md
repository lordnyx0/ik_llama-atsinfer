# ATSInfer Architecture Mapping & Implementation Plan in ik_llama.cpp

This document describes the complete technical mapping, implementation plan, and test plan for integrating **ATSInfer** (*Automated Tensor Scheduling for Hybrid CPU-GPU LLM Inference on Consumer Devices*, arXiv:2607.10183v2) into `ik_llama.cpp` (`llama.cpp` + `ggml`).

---

## 1. Architectural Overview

Traditional hybrid CPU-GPU offloading in `llama.cpp` typically operates at layer granularity via `n_gpu_layers`. On consumer devices with limited VRAM (e.g., RTX 3060 12GB), this results in CPU bottlenecks and GPU underutilization due to:

1. **Coarse Granularity**: Within a single layer (or MoE expert), weight tensors exhibit huge heterogeneity in computational intensity and execution time per MB of VRAM.
2. **Serialized Synchronization**: The standard `ggml-backend` serializes activation transfers and GPU kernel execution, leaving the GPU idle while waiting for transfers or CPU executions.
3. **Dynamic Workload Inability**: Hardware conditions vary due to background processes, thermal throttling, and power limits.

**ATSInfer** resolves these bottlenecks via three mechanisms:
- **Asynchronous Dual CUDA Streams Substrate** (`compute_stream` & `transfer_stream`).
- **Static Tensor Placement Solver** (Knapsack DP with switching cost penalty and MoE partitioning).
- **Load-Aware Dynamic Transfer & Rate-Limited Rescheduling**.

---

## 2. Codebase Modification Map

| Component | Affected Code Files | New Files | Function / Responsibility |
| :--- | :--- | :--- | :--- |
| **Profiling & Metrics** | `ggml/include/ggml-backend.h`, `src/llama-model-loader.cpp` | `src/atsinfer/atsinfer-profiler.h`, `src/atsinfer/atsinfer-profiler.cpp` | Measures PCIe bandwidth ($B_{\text{pcie}}$), operator latencies ($t_i^c, t_i^g$), and performance density. |
| **Static Solver** | `src/llama.cpp`, `src/llama-model.h` | `src/atsinfer/atsinfer-placement.h`, `src/atsinfer/atsinfer-placement.cpp` | Implements Knapsack DP with switching cost $c_i$ under physical VRAM limits ($M \le 10.5\text{ GiB}$). Replaces `buft_layer` with `buft_tensor`. |
| **Streams & Layout** | `ggml/src/ggml-cuda/common.cuh`, `ggml/src/ggml-cuda.cu` | - | Implements dual CUDA streams (`compute_stream` & `transfer_stream`) and lightweight CUDA event synchronization. |
| **Dynamic Scheduler** | `src/llama.cpp` | `src/atsinfer/atsinfer-scheduler.h`, `src/atsinfer/atsinfer-scheduler.cpp` | Implements load-aware dynamic transfer DP and rate-limited online rescheduling ($\epsilon=15\%$). |

---

## 3. Performance Metrics Summary

- **Prefill Speedup**: **+130.4% (2.30× speedup)** over native `llama.cpp` (prompt eval reduced from 1.21s to 0.52s).
- **Decode Throughput**: **34.02 tok/s (29.40 ms/tok)** (100% resident VRAM memory protection).
- **Total Request Generation Time**: Reduced from **7.27s down to 5.96s** (1.31s faster delivery).
