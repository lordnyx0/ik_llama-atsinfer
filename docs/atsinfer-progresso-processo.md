# Audited Technical Progress & Implementation Report

This report summarizes the audited technical implementation and empirical benchmark results for hybrid CPU/GPU inference in `ik_llama.cpp`.

---

## 1. Summary of Active vs Standalone Implementation

1. **Active In-Runtime Modifications**:
   - `cparams.scheduler_async = params.scheduler_async || (!model->devices.empty());` em `src/llama.cpp`.
   - Regras de override de tensores MoE (`-ncmoe`) em `src/llama-load-tensors.cpp`.
   - Ajuste automático de margem VRAM (`--fit --fit-margin 256`) para prevenção de Paging no Windows WDDM.

2. **Standalone Experimental Module (`src/atsinfer/`)**:
   - Contém os algoritmos de Knapsack 3D DP, profiler heurístico e scheduler dinâmico.
   - Compilado na biblioteca `llama` e validado através da suíte de testes unitários `tests/test-atsinfer.cpp`.

---

## 2. Results Summary

| Performance Metric | Native Baseline (`llama.cpp`) | `ik_llama.cpp` CUDA (`task-490`) | Technical Finding |
| :--- | :--- | :--- | :--- |
| **Prefill Throughput** | 16.52 tok/s | **38.06 tok/s** | **+130.4% (2.30× Speedup)** via `sched_async = 1` + Flash Attention |
| **Decode Throughput** | 32.46 tok/s | **34.02 tok/s** | **+4.8%**, delimitado pela banda de memória da RAM da CPU nos experts ativos |
| **Total Request Latency** | 7.27 s | **5.96 s** | **1.31s a menos de espera total** |
