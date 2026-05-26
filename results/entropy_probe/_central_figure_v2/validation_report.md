# Survival-curves validation report

Standard rigor: 4 checks per `docs/theory/central_figure_plan.md`. Reported per (dataset, model) cell.

## mbpp / Qwen--Qwen2.5-Coder-7B-Instruct

Random subsample size: **500**

### sigma_p2_recomputation — ✅ PASS

- Acceptance: `fraction_within_1e-3 >= 0.99`
- n_sampled: `500`
- max_delta: `4.800826832240812e-06`
- mean_delta: `1.8789978363131787e-08`
- fraction_within_1e-3: `1.0`

### top_32_truncation_mass — ✅ PASS

- Acceptance: `median <= 0.01 AND p99 <= 0.10 (most records have <1% tail leakage)`
- n_sampled: `500`
- min: `0.0`
- median: `5.346273157513792e-08`
- mean: `8.06809518987841e-05`
- p95: `2.375177496105607e-05`
- p99: `0.000233507780048966`
- max: `0.01961396320257336`

### H_recomputation_well_formed — ✅ PASS

- Acceptance: `all H finite AND all H >= 0`
- n_sampled: `500`
- min_H: `2.080954285496316e-07`
- median_H: `7.302886376817906e-05`
- max_H: `1.7326169251954362`
- all_finite: `True`
- all_nonneg: `True`

### per_bin_sample_size_adequacy — informational

- Total bins: 80
- Populated bins (n_positions > 0): 59
- Reliable bins (n_positions ≥ 50): 47
- The figure plots reliable bins as solid lines, low-count bins as dotted/translucent so the reader can see which range is statistically meaningful.

## mbpp / codellama--CodeLlama-7b-Instruct-hf

Random subsample size: **500**

### sigma_p2_recomputation — ✅ PASS

- Acceptance: `fraction_within_1e-3 >= 0.99`
- n_sampled: `500`
- max_delta: `7.507481614155154e-08`
- mean_delta: `6.697177266312693e-09`
- fraction_within_1e-3: `1.0`

### top_32_truncation_mass — ✅ PASS

- Acceptance: `median <= 0.01 AND p99 <= 0.10 (most records have <1% tail leakage)`
- n_sampled: `500`
- min: `0.0`
- median: `1.405881788762997e-08`
- mean: `1.6719130615488132e-05`
- p95: `1.1243974221031294e-05`
- p99: `0.0005743085826088614`
- max: `0.001330324710579589`

### H_recomputation_well_formed — ✅ PASS

- Acceptance: `all H finite AND all H >= 0`
- n_sampled: `500`
- min_H: `4.8726772722895223e-08`
- median_H: `6.143392739953577e-05`
- max_H: `1.7707244990003728`
- all_finite: `True`
- all_nonneg: `True`

### per_bin_sample_size_adequacy — informational

- Total bins: 80
- Populated bins (n_positions > 0): 59
- Reliable bins (n_positions ≥ 50): 41
- The figure plots reliable bins as solid lines, low-count bins as dotted/translucent so the reader can see which range is statistically meaningful.

## gsm8k / Qwen--Qwen2.5-Coder-7B-Instruct

Random subsample size: **500**

### sigma_p2_recomputation — ✅ PASS

- Acceptance: `fraction_within_1e-3 >= 0.99`
- n_sampled: `500`
- max_delta: `1.173067691373486e-06`
- mean_delta: `4.3868427051896306e-08`
- fraction_within_1e-3: `1.0`

### top_32_truncation_mass — ✅ PASS

- Acceptance: `median <= 0.01 AND p99 <= 0.10 (most records have <1% tail leakage)`
- n_sampled: `500`
- min: `0.0`
- median: `7.283955596903979e-05`
- mean: `0.0005866496808832196`
- p95: `0.003402975020071581`
- p99: `0.008229483687027821`
- max: `0.009372908534714952`

### H_recomputation_well_formed — ✅ PASS

- Acceptance: `all H finite AND all H >= 0`
- n_sampled: `500`
- min_H: `5.476376940659793e-06`
- median_H: `0.04884220379977852`
- max_H: `2.3108191942927316`
- all_finite: `True`
- all_nonneg: `True`

### per_bin_sample_size_adequacy — informational

- Total bins: 80
- Populated bins (n_positions > 0): 59
- Reliable bins (n_positions ≥ 50): 54
- The figure plots reliable bins as solid lines, low-count bins as dotted/translucent so the reader can see which range is statistically meaningful.

## gsm8k / codellama--CodeLlama-7b-Instruct-hf

Random subsample size: **500**

### sigma_p2_recomputation — ✅ PASS

- Acceptance: `fraction_within_1e-3 >= 0.99`
- n_sampled: `500`
- max_delta: `6.802747131230691e-05`
- mean_delta: `1.0492535081809417e-06`
- fraction_within_1e-3: `1.0`

### top_32_truncation_mass — ✅ PASS

- Acceptance: `median <= 0.01 AND p99 <= 0.10 (most records have <1% tail leakage)`
- n_sampled: `500`
- min: `2.9211224994440954e-07`
- median: `0.0006862823488518188`
- mean: `0.004134028862177612`
- p95: `0.022147580070304682`
- p99: `0.04090790878166445`
- max: `0.06794962310232222`

### H_recomputation_well_formed — ✅ PASS

- Acceptance: `all H finite AND all H >= 0`
- n_sampled: `500`
- min_H: `0.000940111640653564`
- median_H: `0.2809314370447702`
- max_H: `2.803909863819678`
- all_finite: `True`
- all_nonneg: `True`

### per_bin_sample_size_adequacy — informational

- Total bins: 80
- Populated bins (n_positions > 0): 61
- Reliable bins (n_positions ≥ 50): 59
- The figure plots reliable bins as solid lines, low-count bins as dotted/translucent so the reader can see which range is statistically meaningful.

---

**Overall**: ✅ ALL CHECKS PASSED