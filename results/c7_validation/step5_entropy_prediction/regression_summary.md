# C7 v3 Step 5: predicting per-task pass-rate shift from α=2 entropy

Target: `Δp_i = c_i^(α)/10 − c_i^(α=2)/10` for each task `i`. Features extracted from per-position entropy log at α=2 only (streaming aggregate over all 500 tasks × 10 samples).

## R² comparison

- **null**: predict Δp = constant (just intercept) — sanity check, should be ~0
- **baseline**: Δp = intercept + β·p_α2 — does pass-rate at α=2 alone predict shift?
- **full**: Δp = intercept + β·p_α2 + Σ entropy features
- **Δ-only**: Δp = intercept + Σ entropy features (no p_α2 covariate)
- **single**: Δp = intercept + β·p_α2 + γ·mean_log_ratio (a single 'distance' feature)

| Model | α target | n | null R² | baseline R² | full R² | Δ-only R² | single-feat R² |
|---|---:|---:|---:|---:|---:|---:|---:|
| Qwen2.5-Coder-7B-Instruct | 2.5 | 500 | +0.0000 | +0.1126 | +0.1399 | +0.0204 | +0.1132 |
| Qwen2.5-Coder-7B-Instruct | 3.0 | 500 | +0.0000 | +0.1286 | +0.1740 | +0.0422 | +0.1287 |
| Qwen2.5-Coder-7B-Instruct | 5.0 | 500 | +0.0000 | +0.1564 | +0.1900 | +0.0303 | +0.1598 |
| CodeLlama-7B-Instruct | 2.5 | 500 | +0.0000 | +0.0835 | +0.0952 | +0.0088 | +0.0848 |
| CodeLlama-7B-Instruct | 3.0 | 500 | +0.0000 | +0.1221 | +0.1404 | +0.0164 | +0.1278 |
| CodeLlama-7B-Instruct | 5.0 | 500 | +0.0000 | +0.1777 | +0.2020 | +0.0254 | +0.1843 |

## Full-model coefficients (per cell)

### Qwen2.5-Coder-7B-Instruct / α=2.5

| Feature | Coefficient |
|---|---:|
| `intercept` | -3.324488 |
| `p_alpha2` | -0.138612 |
| `mean_H2` | -3.200460 |
| `mean_H3` | +5.334716 |
| `mean_H5` | -1.791825 |
| `var_log_sigma_p2` | -0.875076 |
| `mean_log_ratio_23` | +13.869893 |
| `mean_log_ratio_25` | -3.966840 |
| `var_log_ratio_25` | -0.023695 |
| `frac_high_entropy_positions` | +0.209689 |
| `mean_max_p` | +3.484639 |
| `mean_positions_per_sample` | -0.000474 |

### Qwen2.5-Coder-7B-Instruct / α=3.0

| Feature | Coefficient |
|---|---:|
| `intercept` | -12.501228 |
| `p_alpha2` | -0.156656 |
| `mean_H2` | +0.667194 |
| `mean_H3` | +3.469613 |
| `mean_H5` | +0.071869 |
| `var_log_sigma_p2` | -1.956342 |
| `mean_log_ratio_23` | +6.272032 |
| `mean_log_ratio_25` | -0.379718 |
| `var_log_ratio_25` | -0.099674 |
| `frac_high_entropy_positions` | +0.128730 |
| `mean_max_p` | +12.688555 |
| `mean_positions_per_sample` | -0.000707 |

### Qwen2.5-Coder-7B-Instruct / α=5.0

| Feature | Coefficient |
|---|---:|
| `intercept` | +2.786708 |
| `p_alpha2` | -0.213351 |
| `mean_H2` | -2.623203 |
| `mean_H3` | +9.135595 |
| `mean_H5` | -3.017254 |
| `var_log_sigma_p2` | -2.943545 |
| `mean_log_ratio_23` | +20.894392 |
| `mean_log_ratio_25` | -9.445815 |
| `var_log_ratio_25` | +0.476594 |
| `frac_high_entropy_positions` | +0.281554 |
| `mean_max_p` | -2.579410 |
| `mean_positions_per_sample` | -0.000636 |

### CodeLlama-7B-Instruct / α=2.5

| Feature | Coefficient |
|---|---:|
| `intercept` | +2.887537 |
| `p_alpha2` | -0.081577 |
| `mean_H2` | -0.542778 |
| `mean_H3` | -5.768578 |
| `mean_H5` | +1.187572 |
| `var_log_sigma_p2` | +2.751183 |
| `mean_log_ratio_23` | -10.994377 |
| `mean_log_ratio_25` | +5.293067 |
| `var_log_ratio_25` | -0.458193 |
| `frac_high_entropy_positions` | -0.103483 |
| `mean_max_p` | -2.836669 |
| `mean_positions_per_sample` | -0.000200 |

### CodeLlama-7B-Instruct / α=3.0

| Feature | Coefficient |
|---|---:|
| `intercept` | +4.614936 |
| `p_alpha2` | -0.122170 |
| `mean_H2` | -0.121683 |
| `mean_H3` | -5.558401 |
| `mean_H5` | +1.092101 |
| `var_log_sigma_p2` | +2.288997 |
| `mean_log_ratio_23` | -10.995118 |
| `mean_log_ratio_25` | +4.490089 |
| `var_log_ratio_25` | -0.217340 |
| `frac_high_entropy_positions` | -0.764753 |
| `mean_max_p` | -4.545855 |
| `mean_positions_per_sample` | -0.000211 |

### CodeLlama-7B-Instruct / α=5.0

| Feature | Coefficient |
|---|---:|
| `intercept` | +1.244897 |
| `p_alpha2` | -0.160524 |
| `mean_H2` | +2.310393 |
| `mean_H3` | -1.256865 |
| `mean_H5` | +0.668158 |
| `var_log_sigma_p2` | -1.670877 |
| `mean_log_ratio_23` | -4.824123 |
| `mean_log_ratio_25` | +0.362238 |
| `var_log_ratio_25` | +0.480806 |
| `frac_high_entropy_positions` | -0.669246 |
| `mean_max_p` | -1.158887 |
| `mean_positions_per_sample` | -0.000244 |

## Reading the result

If **full R² » baseline R²** and **Δ-only R² > 0**, entropy features add real predictive signal beyond knowing `pass-rate at α=2`. That would justify Step 5b (building a closed-form prediction of `(a_α, b_α)` from entropy).

If **full R² ≈ baseline R²** (entropy adds nothing) **AND** baseline itself is low (<0.2), the per-task pass-rate shift is mostly noise at our n=10 sample size, and Step 5 should pivot to population-level prediction or accept that the empirical ν(α) regularity is the deliverable.
