# Audited Benchmark Report: ik_llama.cpp CUDA Inference

This document records the empirical inference benchmark results for the **Qwen3.6-35B-A3B** model comparing official `llama.cpp` baseline with `ik_llama.cpp` CUDA optimizations.

---

## 1. Test Setup

- **Hardware**: NVIDIA GeForce RTX 3060 (12GB VRAM), x86-64 CPU (8 Threads).
- **Model**: `Qwen3.6-35B-A3B-Uncensored-HauhauCS-Aggressive-Q4_K_M.gguf` (34.66B parameters, 19.70 GiB).
- **Inference Parameters**:
  - Context ($n_{ctx}$): 32,000 tokens.
  - VRAM Offloading: Physical budget fitting (`--fit --fit-margin 256` / `-ncmoe 18`).
  - CPU Threads: 8 threads (`-t 8`).
  - Flash Attention: Enabled (`-fa on`).
  - KV Cache Quantization: `q8_0` (`-ctk q8_0 -ctv q8_0`).
  - Test Prompt: `"Write a detailed and inspiring paragraph about the Moon, its history, beauty, and impact on Earth."`
  - Tokens Generated ($n_{predict}$): 256 tokens.

---

## 2. Empirical Benchmark Results

| Performance Metric | Official Baseline (`llama.cpp`) | `ik_llama.cpp` CUDA (`task-490`) | Technical Cause & Analysis |
| :--- | :--- | :--- | :--- |
| **Prefill Throughput (tok/s)** | 16.52 tok/s | **38.06 tok/s** | **+130.4% (+2.30× Speedup)** (Driven by `sched_async = 1` + Flash Attention) |
| **Prompt Eval Time (20 tokens)** | 1,210.53 ms | **525.50 ms** | **56.6% Faster Prompt Processing** |
| **Decode Throughput (tok/s)** | 32.46 tok/s | **34.02 tok/s** | **+4.8%** (Bounded by CPU DRAM bandwidth on MoE active experts) |
| **Time per Token (TPOT)** | 30.81 ms/tok | **29.40 ms/tok** | **29.40 ms/tok** |
| **Total Execution Time** | 7,270.24 ms | **5,964.22 ms** | **1.31 Seconds Faster Total Delivery** |
| **GPU VRAM Allocation** | 9.74 GiB | **9.86 GiB** | Enforced physical budget (Avoids WDDM 7.95 tok/s thrashing collapse) |
| **CPU Pinned RAM** | 10.19 GiB | **9.83 GiB** | Host Pinned Memory for MoE Experts |
