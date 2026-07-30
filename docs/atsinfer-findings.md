# ATSInfer in ik_llama.cpp: implementation and measured results

Empirical findings from implementing *Automated Tensor Scheduling for Hybrid CPU-GPU LLM Inference
on Consumer Devices* (arXiv:2607.10183v2) in this fork.

**Bottom line: none of the paper's three mechanisms beats the manual controls this fork already
has**, on either a MoE or a dense model. On MoE the reasons are structural; on dense the solver is
held back by two calibration defects that are identified and fixable. The measurement
infrastructure built along the way is the part worth keeping.

## Test setup

| | |
|---|---|
| GPU | RTX 3060, 12288 MiB |
| RAM | 32 GiB |
| Model | Qwen3.6-35B-A3B-Uncensored Q4_K_M, 19.70 GiB, 40 MoE layers, 128 experts, 8 used |
| Command | `llama-bench -p 512 -n 64 -r 3 -t 8 -ngl 99 --n-cpu-moe 19 -fa 1 -ctk q8_0 -ctv q8_0 -mmp 0` |

## Measurement methodology

**Do not benchmark MoE decode with `llama-cli`.** Three runs of the identical command gave 36.67,
34.78 and 21.50 tok/s. `llama-bench` on the same configuration gives **38.34 ± 0.23** (σ = 0.6%).

The cause is not noise. With host-resident experts, routing decides how many expert bytes cross
PCIe per token, so throughput depends on *which tokens are generated*. Changing placement changes
rounding, which changes sampling, which changes the token stream, which changes PCIe traffic — the
measured variable is coupled to one you do not control. `--ignore-eos` does not fix it: the model
terminates on a different EOG token and the run still stops early, and short runs inflate ms/token
through warmup.

`llama-bench` sidesteps all of this with synthetic tokens and repetitions. It has been given
`--atsinfer`, `--atsinfer-vram-budget` and `--atsinfer-dynamic` for this work.

Note that prefill (`pp512`) has σ of 10–25% even with `llama-bench`; prefill numbers below are not
conclusive. Decode is clean.

## Round composition

Measured with the per-split profiler added in this work (`ggml_backend_sched_set_profiling`), on a
steady-state decode round:

| exec CPU | exec GPU | input copy | round | copy share |
|---|---|---|---|---|
| 14.68 ms | 11.40 ms | 1.98 ms | 28.06 ms | 7.1% |
| 12.83 ms | 11.68 ms | 4.31 ms | 28.82 ms | 14.9% |
| 13.41 ms | 11.93 ms | 1.85 ms | 27.19 ms | 6.8% |
| 13.17 ms | 11.08 ms | 1.64 ms | 25.89 ms | 6.3% |

Two things follow.

**~87% of the round is dependent, sequential compute.** CPU and GPU work sums to the round rather
than overlapping, because layer N+1 consumes layer N's output. At batch size 1 this dependency is
strict and no scheduling mechanism removes it. Only the ~7% spent in transfers is overlappable at
all, which caps everything the paper's coordination mechanism could recover.

**Per expert-layer group, `t_c` ≈ 0.74 ms.**

## Results

### Static tensor placement (section 4.3)

| config | pp512 | tg64 |
|---|---|---|
| `--n-cpu-moe 19` | 183.07 ± 20.76 | **38.34 ± 0.23** |
| `--atsinfer` | 204.05 ± 10.35 | **9.15 ± 0.03** |

The solver loses 4.2x on decode. It picks tensors independently by performance density and never
co-locates a layer's expert weights: layer 12 gets `ffn_up` on the GPU with `ffn_gate` and
`ffn_down` on the CPU. That breaks `GGML_OP_MOE_FUSED_UP_GATE` and raises graph splits from 44 to
64. Prefill is marginally better, matching the paper's own ablation (Figure 15), where static
placement helps prefill and does little for decode.

Fixing this means treating a layer's three expert tensors as a single knapsack item.

Three integration defects were fixed along the way and are worth recording because each was silent:

- The regex escaper re-escaped the backslashes it had just inserted, so `blk.0.x` became
  `blk\\.0\\.x` and matched nothing — 0 of 733 overrides applied.
- `gpu_buft` was read from `buft_layer[0]`, but layers `[0, i_gpu_start)` hold the CPU buffer type,
  so under partial offload every tensor silently landed on the host.
- The solver dragged `token_embd.weight` (272.81 MiB) and `output.weight` (397.85 MiB) onto the
  GPU, violating the invariant at `llama.cpp:4042` that the input layer stays on the CPU. Cost:
  33.85 → 20.15 tok/s.

Also: `--atsinfer-vram-budget` was neither auto-detected nor bounded. With 15000 MiB on a 12288 MiB
card the loader placed 14700 MiB of weights on CUDA0; WDDM spilled to system memory over PCIe and
decode collapsed to 5.78 tok/s. It is now clamped against free VRAM with a 1 GiB reservation.

### Load-aware dynamic transfer (section 4.4)

Algorithms 2 and 3 are implemented (`src/atsinfer/atsinfer-scheduler.cpp`) and unit tested, and
wired into the decode loop (`src/llama-atsinfer.cpp`). The mechanism works but has no headroom.

Promoting one host-resident expert layer to the GPU saves `t_c - t_g` ≈ 0.65 ms and costs `w_i` ≈
1.4 ms of PCIe at 11.8 GB/s. Promoting all 19 would need ~27 ms of transfer inside a ~26 ms round.
The DP correctly declines.

`--atsinfer-dynamic` is off by default and should stay off on this configuration.

### Asynchronous CPU-GPU coordination (section 4.2)

**Structurally blocked, and this is the central finding.**

Moving host-to-device weight copies to a dedicated CUDA stream was implemented and measured A/B in
the same binary:

| | tg64 |
|---|---|
| copies on the compute stream | **39.60 ± 0.23** |
| copies on a dedicated stream | **38.95 ± 0.39** |

No gain. The reason is in `ggml_backend_sched_copy_inputs()`: to copy only the routed experts, the
scheduler reads the routing ids back to the host, and does so with a full device synchronize
(`ggml_backend_synchronize(ids_backend)`, `ggml-backend.cpp`). By the time the expert copies are
issued the compute stream is already drained, so there is no in-flight work to overlap with.

This also rules out the Zero-Copy copy-kernel path (section 2.1). That mechanism exists to stop
small activation transfers from queuing behind large weight transfers on the single copy engine of
a consumer RTX — but with no concurrency there is no contention to relieve.

It cannot be worked around by starting layer N+1's transfer earlier either: N+1's router consumes
N's output.

The change was reverted; a note recording the experiment sits at the call site in `ggml-cuda.cu`.

## Dense models

The blockers above are MoE-specific — there is no routing, so no readback synchronize — so a dense
model that exceeds VRAM was tested separately: **Qwen3.6-27B Q4_K_S, 15.90 GiB, 65 layers**, same
RTX 3060.

| Configuration | weights on GPU | graph splits | pp512 | tg64 |
| :--- | ---: | ---: | ---: | ---: |
| `--fit 1 --fit-margin 256` | 7121 MiB | — | 248.00 ± 17.37 | 2.09 ± 0.02 |
| `-ngl 38` | 7758 MiB | 363 | **285.23 ± 16.40** | 2.30 ± 0.02 |
| `--atsinfer` | 9448 MiB | 645 | 279.54 ± 9.33 | 2.34 ± 0.01 |
| `-ngl 42` | 8576 MiB | 308 | 196.67 ± 13.22 | **2.56 ± 0.07** |

`--atsinfer` beats `--fit` by 13.6% on decode, but loses to a hand-tuned `-ngl 42` by 9.4% **while
using 872 MiB more VRAM**. It put more bytes on the GPU and got less throughput out of them.

The cause is the same failure as on MoE, by a different route: tensor-granularity placement
scatters tensors across backends and fragments the graph — 645 splits against 308 for `-ngl 42`.

Two things follow.

**The gain over `--fit` is packing, not placement.** With the heuristic profiler
(`t = size × constant`) the performance density `t/s` is identical for every tensor, so the
knapsack cannot discriminate and degenerates into "maximise bytes on the GPU". That happens to beat
`--fit` here only because `--fit` is badly calibrated on this model — it left about 4.5 GiB of VRAM
idle. That is worth investigating on its own, independently of ATSInfer.

**The switching penalty is calibrated too low.** The paper's objective does penalise backend
switches (`- Σ c_i·1{b_i ≠ b_{i-1}}`) and the solver carries a `last_backend` dimension for it, but
`c_i = S_in,i / B_pcie` only prices the activation transfer. It ignores kernel launch overhead,
synchronization, and the pipeline break — which is what a split actually costs. 645 splits is the
symptom.

So: **dense models are not a use case for this code as it stands either.** They would become one
only after (a) real measured `t_c` / `t_g` so the density metric discriminates, and (b) a switching
cost calibrated against measured split overhead rather than transfer size.

## Why the paper's speedups do not transfer

The paper's baseline is stock llama.cpp, which moves **all** experts of a layer. This fork has
`only_active_experts`, which reads the routing and moves only the ~8 selected ones. Most of the
transfer volume the paper's overlap mechanism hides has already been eliminated here.

The irony is that the same feature closes the door on the paper's coordination mechanism:
selective transfer requires knowing the routing, knowing the routing requires a synchronize, and
the synchronize destroys the overlap window. The paper acknowledges the constraint in section 4.2.2
("expert-weight transfer must wait until routing has been resolved") and relies on decode
activating few experts to keep the exposed cost small — which is exactly the regime this fork is
already in.

## What was kept

- `ggml_backend_sched_set_profiling` / `ggml_backend_sched_get_split_timings`: per-split execution
  and input-copy timing, zero cost when disabled. This produced every number above.
- `ggml_backend_sched_backend_index`: previously a static helper.
- Algorithms 2 and 3, unit tested, inactive by default.
- `llama-bench` support for the ATSInfer flags.

## If this is picked up again

Ranked by expected value:

1. **Calibrate the switching cost against measured split overhead**, not `S_in,i / B_pcie`. Graph
   fragmentation is what sinks the solver on both model families — 44→64 splits on MoE, 308→645 on
   dense. The profiling API added here reports per-split time and can price a split directly.
2. **Replace the `size_mb * 0.45` / `* 0.06` heuristics** in `atsinfer-profiler.cpp` with measured
   `t_c` / `t_g`. Section 4.3.1 requires measurement; with a size-proportional model the density
   metric is constant across tensors and the knapsack has nothing to discriminate on.
3. **Group each layer's three expert tensors as one knapsack item** for MoE models, since splitting
   them breaks `GGML_OP_MOE_FUSED_UP_GATE`.

Items 1 and 2 are the ones that matter — without them the solver is a VRAM filler, and a
fragmenting one.

Separately and independently of ATSInfer: **`--fit` left ~4.5 GiB of VRAM idle** on the dense
27B model (7121 MiB of weights on a 12287 MiB card). If that reproduces on other dense models it is
a calibration bug worth chasing on its own; it is the entire reason `--atsinfer` appears to win
there.

Expect single-digit percentages at best from the ATSInfer work itself. On MoE the binding
constraint is 12 GiB of VRAM against a 19.70 GiB model, and no scheduling decision removes it.
