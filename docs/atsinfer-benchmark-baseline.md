# Benchmark Report: Native Baseline vs ATSInfer CUDA

This document records the inference benchmark results for the **Qwen3.6-35B-A3B** model comparing the official `llama.cpp` baseline with **ATSInfer** (*Automated Tensor Scheduling for Hybrid CPU-GPU LLM Inference*, arXiv:2607.10183v2).

---

## 1. Test Setup

- **Hardware**: NVIDIA GeForce RTX 3060 (12GB VRAM), x86-64 CPU (8 Threads).
- **Model**: `Qwen3.6-35B-A3B-Uncensored-HauhauCS-Aggressive-Q4_K_M.gguf` (34.66B parameters, 19.70 GiB).
- **Inference Parameters**:
  - Context ($n_{ctx}$): 32,000 tokens.
  - Offloading: Physical VRAM optimization (`--fit --fit-margin 256` / `-ncmoe 18`).
  - CPU Threads: 8 threads (`-t 8`).
  - Flash Attention: Enabled (`-fa on`).
  - KV Cache Quantization: `q8_0` (`-ctk q8_0 -ctv q8_0`).
  - Test Prompt: `"Write a detailed and inspiring paragraph about the Moon, its history, beauty, and impact on Earth."`
  - Tokens Generated ($n_{predict}$): 256 tokens.

---

## 2. Benchmark Results Comparison

| Performance Metric | Official Baseline (`llama.cpp`) | ATSInfer CUDA (`task-490`) | Winner / Speedup Gain |
| :--- | :--- | :--- | :--- |
| **Prefill Throughput (tok/s)** | 16.52 tok/s | **38.06 tok/s** | **ATSInfer (+130.4% / 2.30× Speedup)** |
| **Prompt Eval Time (20 tokens)** | 1,210.53 ms | **525.50 ms** | **ATSInfer (56.6% Faster Prompt Eval)** |
| **Decode Throughput (tok/s)** | 32.46 tok/s | **34.02 tok/s** | **ATSInfer (+4.8% / 29.40 ms per token)** |
| **Time per Token (TPOT)** | 30.81 ms/tok | **29.40 ms/tok** | **ATSInfer (29.40 ms/tok)** |
| **Total Execution Time** | 7,270.24 ms | **5,964.22 ms** | **ATSInfer (1.31 Seconds Faster Total Delivery)** |
| **GPU VRAM Allocation** | 9.74 GiB | **9.86 GiB** | 100% Resident in Physical VRAM |
| **CPU Pinned RAM** | 10.19 GiB | **9.83 GiB** | Host Pinned Memory for MoE Experts |
| **Async Scheduling (`sched_async`)** | Disabled (`0`) | **Enabled (`1`)** | Dual CUDA Streams Active |

---

## 3. Technical Breakdown & Hardware Analysis

1. **Prefill Acceleration (+130.4% Speedup)**:
   - Evaluated prompt processing in **525.50 ms** (down from 1,210.53 ms), achieving over **double the prefill throughput** (38.06 tok/s vs 16.52 tok/s).
   - Dual CUDA Streams (`compute_stream` and `transfer_stream`) pipeline host-to-device transfers while GPU SMs compute Flash Attention kernels.

2. **Decode Bandwidth & VRAM Protection**:
   - Without ATSInfer granular placement, attempting full offloading (`-ngl 999`) on a 12GB card allocated 19.9GB, triggering Windows WDDM PCIe paging and collapsing decode down to **7.95 tok/s**.
   - ATSInfer granular tensor allocation (`buft_tensor`) keeps VRAM usage at **9.86 GiB**, completely eliminating PCIe paging and delivering **34.02 tok/s** decode speed (29.40 ms/tok).
