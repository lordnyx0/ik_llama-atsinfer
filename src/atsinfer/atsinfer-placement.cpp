#include "atsinfer-placement.h"
#include <algorithm>
#include <cmath>
#include <limits>

static atsinfer_placement_decision solve_knapsack_dp(
    const std::vector<atsinfer_tensor_profile> & tensors,
    size_t vram_budget_bytes) {
    
    atsinfer_placement_decision result;
    result.total_vram_used_bytes = 0;
    result.expected_total_latency_ms = 0.0f;

    size_t n = tensors.size();
    if (n == 0) return result;

    // Discretize budget to MB to keep DP memory and time complexity O(n M)
    constexpr size_t MB = 1024 * 1024;
    size_t budget_mb = vram_budget_bytes / MB;

    // State dp[i][w][last_backend]: Max reduction in latency
    // last_backend: 0 = CPU, 1 = GPU
    std::vector<std::vector<std::vector<float>>> dp(
        n + 1, std::vector<std::vector<float>>(budget_mb + 1, std::vector<float>(2, -1e9f)));
    
    std::vector<std::vector<std::vector<int>>> parent_w(
        n + 1, std::vector<std::vector<int>>(budget_mb + 1, std::vector<int>(2, 0)));
    std::vector<std::vector<std::vector<int>>> parent_b(
        n + 1, std::vector<std::vector<int>>(budget_mb + 1, std::vector<int>(2, 0)));

    // Base case: 0 items
    dp[0][0][0] = 0.0f; // Start on CPU
    dp[0][0][1] = 0.0f; // Start on GPU

    for (size_t i = 1; i <= n; ++i) {
        const auto & t = tensors[i - 1];
        size_t size_mb = (t.size_bytes + MB - 1) / MB;
        float r_i = t.latency_reduction;
        float c_i = t.switching_cost_ms;

        for (size_t w = 0; w <= budget_mb; ++w) {
            for (int prev_b = 0; prev_b < 2; ++prev_b) {
                if (dp[i - 1][w][prev_b] < -1e8f) continue;

                // Option 1: Place on CPU (curr_b = 0)
                float penalty_cpu = (prev_b == 1) ? c_i : 0.0f;
                float score_cpu = dp[i - 1][w][prev_b] - penalty_cpu;
                if (score_cpu > dp[i][w][0]) {
                    dp[i][w][0] = score_cpu;
                    parent_w[i][w][0] = (int)w;
                    parent_b[i][w][0] = prev_b;
                }

                // Option 2: Place on GPU (curr_b = 1)
                if (w + size_mb <= budget_mb) {
                    float penalty_gpu = (prev_b == 0) ? c_i : 0.0f;
                    float score_gpu = dp[i - 1][w][prev_b] + r_i - penalty_gpu;
                    size_t next_w = w + size_mb;
                    if (score_gpu > dp[i][next_w][1]) {
                        dp[i][next_w][1] = score_gpu;
                        parent_w[i][next_w][1] = (int)w;
                        parent_b[i][next_w][1] = prev_b;
                    }
                }
            }
        }
    }

    // Find best final state
    float best_score = -1e9f;
    size_t best_w = 0;
    int best_b = 0;

    for (size_t w = 0; w <= budget_mb; ++w) {
        for (int b = 0; b < 2; ++b) {
            if (dp[n][w][b] > best_score) {
                best_score = dp[n][w][b];
                best_w = w;
                best_b = b;
            }
        }
    }

    // Backtrack to recover decisions
    size_t curr_w = best_w;
    int curr_b = best_b;
    for (size_t i = n; i >= 1; --i) {
        const auto & t = tensors[i - 1];
        ATSInferBackend backend = (curr_b == 1) ? ATSInferBackend::GPU : ATSInferBackend::CPU;
        result.placement[t.tensor_name] = backend;
        if (backend == ATSInferBackend::GPU) {
            result.total_vram_used_bytes += t.size_bytes;
        }

        int prev_w = parent_w[i][curr_w][curr_b];
        int prev_b = parent_b[i][curr_w][curr_b];
        curr_w = (size_t)prev_w;
        curr_b = prev_b;
    }

    result.expected_total_latency_ms = best_score;
    return result;
}

atsinfer_placement_decision atsinfer_compute_static_placement(
    const std::vector<atsinfer_tensor_profile> & tensor_profiles,
    size_t vram_budget_bytes,
    bool is_moe_model) {
    
    if (!is_moe_model) {
        return solve_knapsack_dp(tensor_profiles, vram_budget_bytes);
    }

    // MoE specific partitioning: T_nonexp vs T_exp
    std::vector<atsinfer_tensor_profile> t_nonexp;
    std::vector<atsinfer_tensor_profile> t_exp;
    size_t nonexp_size = 0;

    for (const auto & p : tensor_profiles) {
        if (p.tensor_name.find("exps") != std::string::npos || 
            p.tensor_name.find("expert") != std::string::npos) {
            t_exp.push_back(p);
        } else {
            t_nonexp.push_back(p);
            nonexp_size += p.size_bytes;
        }
    }

    atsinfer_placement_decision result;
    if (vram_budget_bytes >= nonexp_size) {
        // Non-expert tensors fit in VRAM, assign them to GPU and solve DP for experts
        for (const auto & p : t_nonexp) {
            result.placement[p.tensor_name] = ATSInferBackend::GPU;
        }
        result.total_vram_used_bytes += nonexp_size;
        
        size_t remaining_budget = vram_budget_bytes - nonexp_size;
        auto exp_decision = solve_knapsack_dp(t_exp, remaining_budget);
        
        for (const auto & kv : exp_decision.placement) {
            result.placement[kv.first] = kv.second;
        }
        result.total_vram_used_bytes += exp_decision.total_vram_used_bytes;
    } else {
        // Budget limited: keep experts on CPU, use DP for non-experts
        for (const auto & p : t_exp) {
            result.placement[p.tensor_name] = ATSInferBackend::CPU;
        }
        auto nonexp_decision = solve_knapsack_dp(t_nonexp, vram_budget_bytes);
        for (const auto & kv : nonexp_decision.placement) {
            result.placement[kv.first] = kv.second;
        }
        result.total_vram_used_bytes += nonexp_decision.total_vram_used_bytes;
    }

    return result;
}
