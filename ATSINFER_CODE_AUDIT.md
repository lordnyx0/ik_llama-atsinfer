# Auditoria técnica do ATSInfer no código-fonte

Data da auditoria: 2026-07-30.

## Conclusão executiva

Classificação: **B) Implementação parcial, mas real**.

O repositório não é apenas README: existe um módulo C++ `src/atsinfer/` compilado na biblioteca `llama`, com testes unitários, um solver de placement, heurísticas de profiling, um scheduler simplificado, cache de residência e uma camada CUDA para streams/eventos/cópia assíncrona. Porém, a implementação é majoritariamente **standalone**: a auditoria não encontrou integração de ATSInfer ao carregamento real de tensores GGUF, ao `ggml_backend_sched`, ao loop de inferência/decode, nem flags CLI reais `--atsinfer`, `--atsinfer-vram-budget` ou `--atsinfer-dynamic`. Portanto, não há evidência de que o runtime faça scheduling adaptativo de pesos durante a inferência.

Confiança: **90%**. A confiança vem de buscas diretas por símbolos ATSInfer fora de `src/atsinfer/` e `tests/test-atsinfer.cpp`, inspeção dos arquivos C++ do módulo, CMake, script de benchmark e histórico Git. A incerteza restante é que a auditoria não executou uma comparação completa contra o upstream remoto atual; a quantificação de mudanças usa o ancestral local `6647db9c` imediatamente anterior ao primeiro commit ATSInfer nesta branch.

## Evidências positivas de código real

### Build e testes

- `src/CMakeLists.txt` adiciona `atsinfer/atsinfer-profiler.cpp`, `atsinfer/atsinfer-placement.cpp`, `atsinfer/atsinfer-scheduler.cpp`, `atsinfer/atsinfer-cache.cpp` e `atsinfer/atsinfer-cuda.cpp` à biblioteca `llama`.
- `tests/CMakeLists.txt` registra `test-atsinfer.cpp` com label `atsinfer`.

### Tensor-level placement / placement solver

- Arquivo: `src/atsinfer/atsinfer-placement.cpp`.
- Funções: `solve_knapsack_dp`, `atsinfer_compute_static_placement`, `atsinfer_map_placement_to_buft`.
- A lógica existe: `solve_knapsack_dp` discretiza o orçamento em MiB, cria uma DP tridimensional `dp[i][w][last_backend]`, avalia alternativas CPU/GPU por tensor, aplica penalidade de troca de backend e faz backtracking para preencher `result.placement`.
- Para MoE, `atsinfer_compute_static_placement` separa tensores por nome em experts e não-experts, força não-experts na GPU quando cabem no orçamento e roda DP no restante; quando não cabem, mantém experts na CPU e roda DP nos não-experts.
- Limitação crítica: `atsinfer_map_placement_to_buft` apenas converte um mapa para `ggml_backend_buffer_type_t`; não foi encontrada chamada desse mapa no carregador real de tensores.

### Profiling para decidir placement

- Arquivo: `src/atsinfer/atsinfer-profiler.cpp`.
- Funções: `atsinfer_profile_hardware`, `atsinfer_profile_tensors`, `atsinfer_save_profile_cache`, `atsinfer_load_profile_cache`.
- A lógica de hardware mede H2D/D2H com CUDA quando habilitado: aloca host pinned/device, usa `cudaMemcpyAsync`, eventos CUDA e `cudaEventElapsedTime`. Sem CUDA, retorna fallback fixo.
- A lógica de profiling de tensores é heurística: lê `ggml_nbytes(tensor)`, extrai `layer_id` de nomes como `blk.X`, classifica attention/FFN/MoE por substrings e estima latências por constantes `size_mb * 0.45f` CPU e `size_mb * 0.06f` GPU.
- Limitação: não mede latência real por op/tensor no grafo de inferência; usa estimativas por tamanho e nome.

### Adaptive tensor scheduling / runtime scheduler simplificado

- Arquivo: `src/atsinfer/atsinfer-scheduler.cpp`.
- Funções/classes: `ATSInferRescheduler::should_reschedule`, `ATSInferRescheduler::record_reschedule_event`, `atsinfer_dynamic_transfer_schedule`.
- A lógica existe como decisão abstrata: o rescheduler compara desvio relativo de latência com um threshold e impõe intervalo mínimo em múltiplos de TPOT. O schedule dinâmico acumula tempo de compute CPU anterior como janela de overlap e promove tensores CPU se `latency_reduction > exposed_transfer_cost`.
- Limitação crítica: não foi encontrada chamada dessas rotinas no decode/runtime real. Elas retornam nomes de tensores a promover, mas não executam migração nem alteram despacho de kernels.

### Asynchronous tensor migration, dual CUDA streams e eventos

- Arquivo: `src/atsinfer/atsinfer-cuda.cpp`.
- Classe: `ATSInferCudaManager`.
- A lógica existe: `init` cria dois streams CUDA non-blocking (`compute_stream` e `transfer_stream`), `migrate_h2d_async` usa `cudaMemcpyAsync(..., cudaMemcpyHostToDevice, transfer_stream)` e grava evento, e `wait_for_transfer_event` faz o compute stream esperar pelo evento de transferência.
- Limitação crítica: não há integração encontrada com buffers reais de pesos GGUF nem com kernels ggml. Fora dos testes, a camada não é chamada por código de inferência.

### Cache de tensores e gerenciamento de memória

- Arquivo: `src/atsinfer/atsinfer-cache.cpp`.
- Classe: `ATSInferTensorCache`.
- A lógica existe: registra tensores com tamanho/layer/residência, mantém contador de bytes residentes na GPU, reserva espaço sob orçamento, escolhe vítimas, marca vítimas como CPU-only, suporta pin/unpin, refcount e timestamp de uso.
- Políticas encontradas: LRU, distância de layer e prioridade para experts MoE.
- Limitação: o cache controla metadados; não foi encontrada ligação com alocação/liberação real de buffers ggml/CUDA no runtime.

## Comparação com funcionalidades esperadas do paper ATSInfer

| Funcionalidade | Status no código | Evidência/observação |
| --- | --- | --- |
| Tensor-level placement | Parcial | Solver por tensor existe em `atsinfer-placement.cpp`, mas não há uso no loader real. |
| Adaptive tensor scheduling | Parcial/fraco | Há rescheduler e heurística de promoção, sem integração no loop de inferência. |
| Asynchronous tensor migration | Parcial | `cudaMemcpyAsync` em classe isolada; sem migração de tensores GGUF reais. |
| Dual CUDA streams | Parcial | Dois streams são criados em `ATSInferCudaManager`; não conectados ao scheduler ggml. |
| Overlap compute/transfer | Parcial/teórico | Scheduler calcula janela de overlap; CUDA manager permite evento/espera. Não há evidência de overlap efetivo no runtime. |
| Pipeline CPU/GPU | Ausente para ATSInfer | O repositório herda capacidades híbridas do llama.cpp/ik_llama, mas ATSInfer não controla pipeline real por tensor. |
| Placement solver knapsack/equivalente | Implementado standalone | DP sob orçamento de VRAM e custo de troca. |
| Runtime scheduler | Ausente como runtime integrado | Apenas funções standalone/testadas sinteticamente. |
| Profiling para placement | Parcial | Hardware PCIe medido; tensor latency é heurística. |
| Cache de tensores | Parcial | Metadados/eviction existem; sem movimentação real de pesos. |
| Políticas adaptativas durante inferência | Ausente | Não há chamadas no decode/inferência para recomputar placement/promover tensors. |

## Mudanças em relação ao llama.cpp/ik_llama base local

Comparando `6647db9c..HEAD` nesta branch local:

- **24 arquivos alterados**.
- **Aproximadamente 1.940 inserções e 2 remoções**.
- Principais subsistemas afetados:
  - novo módulo `src/atsinfer/`;
  - integração CMake mínima em `src/CMakeLists.txt`;
  - teste unitário `tests/test-atsinfer.cpp`;
  - documentação `README.md` e `docs/atsinfer*.md`;
  - script `scripts/benchmark-atsinfer.py`;
  - alterações muito pequenas em `src/llama.cpp` e `src/llama-model.h`.

A única alteração observada em `src/llama.cpp` próxima ao runtime torna `scheduler_async` verdadeiro quando há devices, mas isso é o scheduler assíncrono genérico de graph split, não uma chamada ao módulo ATSInfer. Em `src/llama-model.h`, foi adicionado `buft_tensor`, mas a auditoria não encontrou uso funcional dele para placement ATSInfer.

## Benchmarks

Existe `scripts/benchmark-atsinfer.py`, que executa três modos (`baseline`, `static_atsinfer`, `dynamic_atsinfer`) e salva JSON. Entretanto, o script adiciona flags `--atsinfer`, `--atsinfer-vram-budget` e `--atsinfer-dynamic`, e a auditoria não encontrou essas flags implementadas em `common`, `src` ou `examples`. Assim, o harness parece preparado para um CLI futuro ou inexistente nesta árvore, não uma reprodução funcional garantida.

Não foram encontrados dados gerados automaticamente versionados que comprovem reprodução; há documentos com tabelas e métricas, mas eles não substituem evidência de execução integrada.

## Histórico do projeto

O histórico local mostra uma base de muitos commits upstream de `ik_llama.cpp`, seguida por commits específicos de ATSInfer:

- `56ce3a64 feat(atsinfer): integrate ATSInfer automated tensor scheduling and dual CUDA streams`
- `4a63ba0e docs: add ATSInfer implementation action plan`
- `138a017 docs: audit README and documentation to reflect active runtime vs standalone research module status`
- `c570a017 feat(atsinfer): implement real profiling, cache eviction, async CUDA streams, and benchmark harness`
- commits posteriores documentais de métricas.

Há implementação incremental no módulo standalone, mas a evolução visível não demonstra integração completa ao runtime de inferência.

## Sinais de alerta

- As funções ATSInfer quase não aparecem fora de `src/atsinfer/`, CMake, testes e documentação.
- O script de benchmark usa flags ATSInfer não encontradas no parser CLI.
- `atsinfer_map_placement_to_buft` sugere ponte para buffers ggml, mas não há chamada no loader real.
- `ATSInferCudaManager` implementa migração H2D assíncrona, mas não há chamada conectada a tensores reais.
- `ATSInferTensorCache` controla residência como metadado, sem `cudaMalloc`/`cudaFree`/buffer ggml integrado no cache.
- O profiling por tensor é heurístico, não medição real de operadores.
- Não foram encontrados caminhos de runtime que atualizem políticas adaptativas durante geração.

## Verificações de busca executadas

- `rg -n "ATSInfer|atsinfer|cudaMemcpyAsync|cudaStream|cudaEvent|scheduler|placement|knapsack|migration|tensor migration|cache|profile|profil" -S --glob '!build/**' --glob '!vendor/**' --glob '!models/**'`
- `rg -n "atsinfer_|ATSInfer" src ggml examples common tests CMakeLists.txt -S --glob '!src/atsinfer/**' --glob '!tests/test-atsinfer.cpp'`
- `rg -n -e "--atsinfer" -e "atsinfer-vram" -e "atsinfer-dynamic" common src examples -S --glob '!examples/server/webui/dist/**'`
- `git diff --stat 6647db9c..HEAD`
- `git diff --shortstat 6647db9c..HEAD`
- `git log --oneline --decorate --max-count=30`
