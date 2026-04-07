# The Decoding Landscape for Code Generation: First Principles, Gaps, and Seed Ideas

**Scope:** Survey of 20+ decoding methods mapped to their foundational theories, with cross-disciplinary analogies from physics, biology, signal processing, economics, and CS theory. Grounded in empirical data from 9 models × 66+ configurations on MBPP-full (500 tasks × 10 samples) and HumanEval (164 tasks).

**Purpose:** Identify seed ideas for novel decoding techniques specifically suited for code generation, where the output space is structured, narrow, verifiable, and compositional — fundamentally different from open-ended text.

**Key finding:** Every existing decoding method applies the same strategy at every token position, ignoring that code generation alternates between deterministic steps (closing brackets, keywords), constrained choices (variable names, operators), and creative decisions (algorithm selection). This uniformity is the largest gap in the field.

---

## Notation

| Symbol | Meaning |
|--------|---------|
| `z_i` | Raw logit for token `i` |
| `p_i = softmax(z/T)_i` | Token probability after temperature scaling |
| `V` | Vocabulary size (~150K for modern models) |
| `T` | Temperature parameter |
| `H(p) = -Σ p_i log p_i` | Shannon entropy |
| `H_2(p) = -log Σ p_i²` | Rényi entropy of order 2 (collision entropy) |
| `p_coll = Σ p_i²` | Collision probability (P-less threshold) |
| `y_{1:t}` | Token sequence up to position t |
| `P(y_t \| y_{<t}, x)` | Conditional next-token distribution given context |

---

# Part I: Framing

## 1. Decoding as Sequential Decision-Making Under Uncertainty

Autoregressive decoding is a sequential decision problem. At each step `t`, the model produces a probability distribution `P(y_t | y_{<t}, x)` over the vocabulary, and the decoder must select a single token. This is a Markov decision process where the state is the accumulated context, the action is the next token, and the reward — task correctness — is unknown until the full sequence is complete.

Every decoding method makes an implicit bet about where good completions live in probability space. The fundamental tension is **exploitation** (selecting high-probability tokens to maximize immediate quality) versus **exploration** (sampling broadly to discover diverse solutions).

Our P-less benchmarking data quantifies this tension precisely:

| Method | pass@1 | pass@10 | First-sample capture | Mechanism |
|--------|--------|---------|---------------------|-----------|
| P-less (t=0.6) | 59.4% | 66.4% | 89% | Exploitation-heavy |
| Temperature (t=0.7) | 42.6% | 77.8% | 55% | Exploration-heavy |
| Top-p (p=0.9, t=1.0) | 34.7% | 77.0% | 45% | Maximum exploration |

*(Qwen2.5-Coder-3B on MBPP-full, 500 tasks × 10 samples)*

P-less captures 89% of its maximum potential in a single sample. Temperature captures only 55%. Neither is universally correct — the optimal balance depends on the deployment context (single-sample code completion vs multi-sample search).

**Three axes every decoder operates on:**

1. **Truncation** — which tokens to consider (top-k cuts at rank, top-p at cumulative mass, P-less at collision entropy, grammar-constrained at syntactic validity)
2. **Reweighting** — how to redistribute probability mass among survivors (temperature flattens/sharpens uniformly, min-p preserves relative ordering)
3. **Selection** — how to pick from the reweighted set (sample from categorical, argmax, beam search)

Most methods focus on one axis. P-less is purely a truncation method (axis 1), followed by uniform reweighting (axis 2) and standard sampling (axis 3). Temperature operates only on axis 2. Grammar-constrained decoding operates on axis 1 using structural knowledge rather than distributional statistics. No existing method coordinates across all three axes adaptively.

**Code's unique property:** Unlike open-ended text, code can be parsed, compiled, type-checked, and executed at decode time. This makes code generation the only domain where a decoder can receive *verifiable feedback* during generation — but almost no existing method exploits this.

## 2. Three Regimes of Code Token Distributions

An empirical observation from our data: at any given decoding step in code generation, the model's distribution falls into one of three regimes:

**Deterministic regime** (entropy ≈ 0, ~40-60% of tokens): Only one syntactically valid continuation exists. Closing brackets, required colons after `def` signatures, mandatory indentation. P-less and greedy produce identical output here. Sampling is wasted computation.

**Constrained choice regime** (low entropy, ~30-40% of tokens): 2-10 plausible tokens compete. Variable names, method calls, operator choices, common patterns. P-less prunes the noise and samples from the survivors. This is where P-less's collision entropy threshold provides the most value.

**Creative regime** (moderate-high entropy, ~5-15% of tokens): Genuinely open decisions concentrated at function boundaries, algorithm selection, data structure choice. Multiple correct approaches exist. P-less may be too conservative here — its aggressive pruning can eliminate valid but lower-probability alternatives.

**The fundamental gap:** Current decoders apply the same strategy at every step, regardless of which regime the model is in. A decoder that detects the regime and adapts — greedy for deterministic, P-less for constrained, temperature for creative — would be better calibrated to code's heterogeneous structure.

**Connection to instruct models:** RLHF and instruction tuning push nearly all steps into the deterministic regime. Our data shows P-less achieves 0.000 structural diversity on CodeLlama-7b-Instruct (vs 0.081 on the base model). Even temperature only achieves 0.011 (vs 0.436 on base). The "creative regime" effectively disappears — which is why our current instruct experiment tests high T1 (>1.0) to artificially restore it.

---

# Part II: Taxonomy of Existing Methods

## 3. Foundations Table

| Method | Year | Foundation | What It Optimizes | What It Assumes | What It Ignores | Code Weakness |
|--------|------|-----------|-------------------|-----------------|-----------------|---------------|
| **Greedy** | — | MAP estimation | Mode of P(y_t\|...) | Mode is globally optimal | All uncertainty | Stuck if mode is wrong at any step |
| **Temperature** | — | Boltzmann distribution (stat. mech.) | Exploration via energy scaling | All tokens benefit equally from flattening | Token roles, structure | Can't distinguish syntax from logic tokens |
| **Top-k** | 2018 | Fixed truncation budget | Vocabulary restriction | k is correct for all distributions | Entropy variation | Fixed k ignores that code entropy varies wildly per step |
| **Top-p (Nucleus)** | 2019 | Adaptive mass truncation | Cumulative probability | Top mass is reliable; tail is not | Token identity | p threshold doesn't know about syntax |
| **P-less** | 2024 | Rényi-2 entropy / collision probability | Automatic truncation via Σp² | Collision threshold separates signal from noise | Output structure | Too conservative on peaked instruct distributions |
| **P-less-norm** | 2024 | Normalized collision entropy | Relaxed P-less threshold | Uniform correction matters | Same as P-less | Negligible practical difference (Δ < 0.2pp) |
| **η-sampling** | 2022 | Desmoothing theory | Recover true distribution support | LM = true dist + smoothing | Domain specifics | Smoothing model may not hold for code |
| **Min-p** | 2025 | Confidence-scaled threshold | Temperature-invariant truncation | Mode-relative threshold is correct | Multi-modal distributions | α still requires tuning |
| **Top-n-sigma** | 2025 | Statistical deviation filtering | Distribution-stable truncation | Gaussian-like tails | Heavy-tailed distributions | Untested on code |
| **Typical** | 2023 | Info-theoretic typical set | Shannon typicality | Natural text lives in typical set | Code is not natural text | Excludes deterministic tokens (high prob) that are correct in code |
| **Mirostat** | 2021 | Perplexity control / Zipf stats | Target surprise rate τ | Fixed surprise is optimal | Surprise should vary | Code alternates low/high surprise; fixed target is wrong |
| **Contrastive** | 2023 | Expert-amateur divergence | Quality signal from model comparison | Amateur exhibits same failures, amplified | Requires two models | Inference cost doubles |
| **DOLA** | 2024 | Layer-wise contrastive | Factuality via layer divergence | Earlier layers are "amateurs" | Non-factual tasks | Code correctness ≠ factuality |
| **REAL** | 2024 | Asymptotic entropy extrapolation | Hallucination prediction | Entropy extrapolation valid | Requires trained THF model | Extra 70M model overhead |
| **Beam search** | — | Best-first search / Viterbi | Cumulative log-probability | High-prob sequence = good | Length, structure | Catastrophic on code: 12.8% pass@1 (prefers short outputs) |
| **Speculative** | 2023 | Draft-verify rejection sampling | Inference speed (2-2.5×) | Draft model approximates target | Output quality (unchanged) | Acceleration only, no quality gain |
| **Grammar-constrained** | 2024 | FSM/CFG intersection | Syntactic validity | Grammar is known, static | Semantic correctness | Prevents syntax errors, can't prefer correct logic |
| **Type-constrained** | 2025 | Type systems / context-sensitive parsing | Type correctness | Types are inferrable incrementally | Runtime correctness | 74.8% fewer compile errors, but can't check semantics |
| **MCTS** | 2024 | Tree search with rollout evaluation | Step-level rewards | Reward signal available | Single-pass latency | Requires evaluation function; slow |

## 4. Deep Dives

### 4.1 Greedy / MAP Estimation

**Math:** `y_t = argmax_i P(y_t = i | y_{<t}, x)`

Greedy decoding selects the maximum a posteriori token at every step. It is deterministic, fast, and provides zero diversity. In our benchmarks, greedy is competitive with P-less on pass@1 for instruct models (where the distribution is already near-deterministic), but produces identical outputs for all N samples.

The fundamental limitation: greedy commits irrevocably at each step. If the mode is wrong at token t=5 (a poor variable name, a suboptimal algorithm choice), there is no recovery. The error propagates and compounds. On our Qwen-7B results, beam search (which at least maintains alternatives) still catastrophically fails at 12.8% — indicating that local optimization of cumulative log-probability is the wrong objective for code.

### 4.2 Temperature Sampling (Boltzmann Distribution)

**Math:** `p_i = exp(z_i / T) / Σ_j exp(z_j / T)`

Temperature sampling is directly inherited from statistical mechanics. The Boltzmann distribution governs systems in thermal equilibrium: particles occupy energy states `E_i` with probability `∝ exp(-E_i / kT)`. In LLMs, logits are negative energies, and softmax computes the Boltzmann weights.

At low temperature (T→0), particles are trapped in the lowest energy well (greedy). At high temperature (T→∞), thermal energy allows escaping local minima (uniform sampling). Temperature is a *global* softness knob — it flattens or sharpens the entire distribution uniformly, without distinguishing between high-confidence correct tokens and low-probability noise.

**Our data:** Temperature at t=0.7 achieves 42.6% pass@1 on Qwen2.5-Coder-3B (vs P-less 59.4%), but 77.8% pass@10 (vs 66.4%). Temperature's deduplication ratio is 0.87 (87% unique solutions), vs P-less's 0.22. Temperature buys diversity at the cost of 0.029 CodeBLEU per percentage point of pass@1 lost; P-less buys it at 0.088/pp — 3× more efficiently.

### 4.3 Top-p / Nucleus Sampling

**Math:** Sort tokens by descending probability. Keep tokens until cumulative mass reaches `p`. Renormalize survivors.

Holtzman et al. (2019) observed that the "unreliable tail" of neural language model distributions produces degenerate text. Top-p is an adaptive truncation: when the model is confident (mass concentrated on few tokens), few tokens survive. When uncertain, more survive. This adapts vocabulary size to the distribution shape.

Unlike P-less (which uses the collision probability as threshold), top-p uses the cumulative probability. The two methods carve the same conceptual space differently: P-less asks "which tokens are more likely than a random collision?"; top-p asks "which tokens account for p fraction of the total mass?"

**Our data:** Top-p at p=0.9/t=1.0 achieves the highest diversity (0.589 structural diversity, 0.93 dedup ratio) but the lowest pass@1 (34.7%). It solves 68 more unique tasks than P-less on MBPP-full, but solves 84 fewer tasks consistently.

### 4.4 P-less (Collision Entropy)

**Math** (from `p-less/p_less_samplers.py`):
```python
p = probs.square().sum(dim=-1, keepdim=True)   # Collision probability
mask = probs < p                                 # Tokens below threshold
probs[mask] = 0.0                                # Zero out noise
probs.div_(probs.sum(dim=-1, keepdim=True))      # Renormalize
next_token = torch.multinomial(probs, 1)          # Sample
```

The threshold `p = Σ p_i²` is the collision probability — the probability that two independent samples from the distribution select the same token. This is the inverse of the Rényi entropy of order 2: `H_2 = -log(Σ p_i²)`.

**The key insight:** Tokens with probability below the collision threshold are less likely to be drawn than a random collision event. They represent noise — tokens the model assigns probability to only because softmax never outputs exactly zero. P-less removes them automatically, without any hyperparameter.

**Properties:**
- For a uniform distribution: `p = 1/V` (threshold near zero, minimal truncation)
- For a delta distribution: `p = 1.0` (only the mode survives)
- Self-adapting: peaked distributions get aggressive pruning, flat distributions get permissive pruning
- Bounded: at least one token always survives (`max(p_i) ≥ Σ p_i²` by Cauchy-Schwarz inequality)

**Our data:** P-less achieves the highest pass@1 across all models (59.4% on Qwen2.5-Coder-3B, +16.8pp over temperature). It captures 89% of pass@10 in a single sample. But structural diversity is 0.097 — when P-less produces 8 correct samples, ~6 are structurally identical.

**The P-less t=1.0 sweet spot:** Raising temperature from 0.6 to 1.0 costs only 2.8pp pass@1 while doubling CodeBLEU diversity (0.203 → 0.450). This is 3-5× more efficient than temperature in diversity-per-accuracy-lost. P-less at t=1.0 is Pareto-optimal for the quality-diversity tradeoff.

**Limitation on instruct models:** RLHF sharpens distributions so dramatically that P-less's threshold eliminates all but the top-1 token. Result: 0.000 structural diversity on CodeLlama-7b-Instruct.

### 4.5 Min-p (Confidence-Scaled Threshold)

**Math:** `threshold = α × max(p_i)`. Keep tokens with `p_i ≥ threshold`.

Min-p (Nguyen et al., ICLR 2025, arXiv:2407.01082) scales the threshold relative to the mode rather than the full distribution. Its key property: **temperature-invariance**. Changing T changes the distribution but not which tokens survive min-p truncation, because both the mode and the threshold scale together.

**Comparison with P-less:** Min-p uses only the mode (`max(p_i)`) to set the threshold; P-less uses the entire distribution (`Σ p_i²`). P-less is more sensitive to the shape of the distribution (it penalizes heavy tails more), while min-p is more robust to temperature changes. Both are adaptive, but P-less is fully hyperparameter-free while min-p requires tuning α.

### 4.6 η-sampling (Desmoothing Theory)

**Theory** (Hewitt et al., 2022, arXiv:2210.15191): Language models are trained by maximum likelihood, which can be viewed as fitting a mixture: `P_model = (1-ε)P_true + ε P_smooth`, where `P_smooth` is a smoothing distribution (avoiding infinite perplexity on unseen tokens). η-sampling estimates and removes the smoothing component, recovering the support of the true distribution.

The threshold is entropy-dependent: higher entropy → higher threshold → more aggressive truncation. This is conceptually similar to P-less (which also uses an entropy-derived threshold), but grounded in a different theory (desmoothing vs collision probability).

**For code:** The desmoothing model assumes the LM learned a true distribution corrupted by smoothing. For code, where many tokens are syntactically invalid, this may underestimate the amount of "noise" — much of the probability mass on invalid tokens isn't smoothing but genuine model confusion.

### 4.7 Typical Sampling (Information-Theoretic Typicality)

**Math** (Meister et al., 2023, arXiv:2202.00666): Keep tokens whose pointwise information content `I(y_t) = -log p(y_t)` is within δ of the conditional entropy `H(Y_t | y_{<t})`.

This targets the "typical set" from Shannon's source coding theorem — the set of strings where each token's information content is close to the distribution's entropy. The typical set contains almost all the probability mass but excludes both very-high-probability tokens (boring/deterministic) and very-low-probability tokens (noise).

**For code:** Typical sampling is particularly interesting because it would naturally separate the three regimes. In the deterministic regime (entropy ≈ 0), only the mode is "typical." In the creative regime (higher entropy), more tokens qualify. However, it might incorrectly exclude high-probability tokens that are correct in code (like `return` after a computation) — these are atypical in the information-theoretic sense but completely correct.

### 4.8 Grammar-Constrained / Type-Constrained Decoding

**Foundation:** Formal language theory — intersection of the LM's distribution with a recognizer for the target grammar/type system.

Grammar-constrained decoding (DOMINO, ICML 2024) compiles the target language's grammar into a finite state machine (FSM) or context-free grammar (CFG). At each step, it masks tokens that would produce syntactically invalid continuations. This guarantees 100% syntactic correctness by construction.

Type-constrained decoding (COLM 2025, arXiv:2508.15866) extends this to context-sensitive type systems, tracking variable scopes and type constraints. Results: 74.8% reduction in compilation errors (vs 9.0% for syntax-only constraints).

**These are the only decoding methods that use structural knowledge of the output domain.** All other methods (temperature, P-less, top-p) operate on distributional statistics alone, blind to the fact that the output is code.

**Limitation:** They constrain but don't guide. Grammar-constrained decoding prevents `def foo(x) {` in Python, but can't prefer correct logic over incorrect logic when both are syntactically valid. They are axis-1 (truncation) methods that use structure rather than statistics, but they don't touch axis-2 (reweighting) or axis-3 (selection).

## 5. The Dimension Space of Decoding

Synthesizing the taxonomy, every decoding method occupies a position in a six-dimensional space:

**1. Information source:** Distribution statistics only (temperature, top-p, P-less, min-p, typical, Mirostat) → output structure (grammar/type-constrained) → external signal (contrastive models, MCTS with rewards, REAL with THF model).

**2. Granularity:** Per-token (all standard samplers) → per-step (MCTS with step-level rewards) → per-sequence (best-of-N, rejection sampling).

**3. Adaptivity:** Static hyperparameter (temperature, top-k) → distribution-adaptive (top-p, P-less, min-p, typical, η-sampling) → **context-adaptive (GAP — no method adapts based on what kind of token is being generated)**.

**4. Verifiability exploitation:** None (all standard samplers) → partial (grammar-constrained) → **full (GAP — no method uses compile/test feedback during generation)**.

**5. Multi-model:** Single model (most methods) → multi-model (contrastive decoding, speculative decoding).

**6. Generation paradigm awareness:** Paradigm-agnostic (all current methods — same strategy for thinking tokens and code tokens) → **reasoning-aware (GAP — no method distinguishes thinking/planning phase from code execution phase)**.

The three GAPs (dimensions 3, 4, 6) define the empty regions of the space where novel methods should live.

---

# Part III: Why Code Is Not Text

## 6. Code-Specific Properties Decoding Should Exploit

### 6.1 Structured, Verifiable Output

Code can be parsed (syntax tree), compiled (type errors), executed (runtime behavior), and tested (functional correctness) — all at decode time. No other generation domain has this property. A decoder for code can get *feedback* during generation. Grammar-constrained decoding exploits the weakest form of this (syntax only). No method exploits type-checking or execution feedback during token selection.

### 6.2 Narrow Valid Space with Sharp Boundaries

Our P-less data shows 89% first-sample capture on MBPP — the model "knows" the correct solution with high confidence. The valid solution manifold is small relative to the token space. Creative writing has no analogous "correct answer." This means aggressive truncation (P-less, min-p) is well-suited to code, and the cost of over-pruning is usually low.

### 6.3 Heterogeneous Token Roles

Not all tokens contribute equally to program correctness. Structural tokens (`def`, `if`, `return`, brackets, colons) are nearly deterministic — there is typically one syntactically valid choice. Logic-bearing tokens (algorithm choice, condition expressions, data structure selection) carry all the decision weight. Variable name tokens are somewhere in between (multiple names work, but consistency matters).

Current decoders treat `return` the same as `quicksort` — applying the same truncation, reweighting, and selection. This is a fundamental mismatch.

### 6.4 Tokenization Artifacts

BPE tokenization creates artifacts that interact with decoding. The TokDrift paper (arXiv:2510.14972) shows that BPE boundaries affect generation probability: a syntactically meaningful unit like `<=` may be split into two tokens (`<`, `=`), creating an artificial dependency. In Python, where indentation is semantically meaningful, BPE may tokenize 4 spaces as one token in one context and four tokens in another — directly affecting sampling probability and correctness.

Multi-token operators (e.g., `!=`, `**=`, `<<=`) create correlated sampling decisions: once the first token is chosen, the second is highly constrained. Current decoders don't model this correlation.

### 6.5 Compositional Structure

Code has long-range dependencies that text lacks. A variable name chosen at token t=10 constrains every subsequent reference. A function signature fixes the parameter types for the entire body. Import statements at the top determine which APIs are available hundreds of tokens later.

This compositionality means early token choices have cascading consequences. A decoder that understands this would invest more exploration budget early (algorithm/approach selection) and less later (implementation of a committed approach).

### 6.6 Testability at Decode Time

The unique property of code generation: we can run the partially generated code and get actionable feedback. Does it parse? Does it compile? Does it pass the first test case? This feedback could inform subsequent decoding decisions — but no standard decoder uses it.

This is the single largest untapped resource in code decoding. Every other domain (creative writing, summarization, translation) must wait until generation is complete to evaluate quality. Code can be evaluated incrementally.

---

## 6B. Reasoning vs Non-Reasoning Paradigms

### 6B.1 The Non-Reasoning Paradigm (Direct Generation)

Traditional autoregressive code generation: the model directly emits code tokens, each contributing to the final output. Every token is "output" — there is no planning phase. This is the paradigm in which all decoding methods in Section 3 were designed and evaluated. Our P-less benchmarks on MBPP and HumanEval operate entirely here.

Quality depends entirely on the model's single-pass distribution quality. If the model assigns high probability to the correct approach, P-less captures it efficiently. If the model is uncertain, temperature explores alternatives. But neither method helps the model *think through* the problem before committing to code.

Production code completion tools (GitHub Copilot, Cursor, Codeium) use non-reasoning models for latency reasons. Direct decoding improvements remain critical for these deployment contexts.

### 6B.2 The Reasoning Paradigm (Think-Then-Code)

Reasoning models (OpenAI o1/o3, DeepSeek-R1, QwQ, Claude with extended thinking) generate "thinking tokens" before code. The thinking phase is effectively **internalized search**: the model explores approaches, backtracks, evaluates alternatives in natural language before committing to code.

This creates two distinct token regimes within a single generation:

- **Thinking tokens**: high entropy, exploratory, natural language, errors are tolerable (the model can self-correct within the thinking chain)
- **Code tokens**: structured, lower entropy, errors are fatal (syntax must be valid, logic must be correct)

The thinking phase subsumes much of what decoding methods try to do externally. MCTS? The model is doing tree search in the thinking chain. Best-of-N? The model considers and rejects alternatives during thinking. Contrastive decoding? The model implicitly contrasts approaches.

Inference-time compute scaling (the o1 paradigm) introduces its own scaling laws: more thinking tokens → better results, independent of model size. This is a fundamentally different compute allocation from traditional decoding (more samples or wider beams).

### 6B.3 Implications for Decoding Research

**Does decoding still matter in the reasoning paradigm?** Yes, but differently:

- During the **thinking phase**, standard decoding is adequate (NL text is self-correcting). But diversity-encouraging methods could help the model explore more approaches during thinking — potentially improving the quality of the plan before code emission begins.
- During the **code phase**, aggressive truncation (P-less, greedy) may be ideal. The model has already "planned" and should commit confidently to the implementation.
- The **transition point** from thinking to code is where adaptive decoding matters most. No existing method detects or exploits this transition.

**The hybrid hypothesis:** Reasoning models may benefit from different decoding strategies for thinking vs code tokens. This is unexplored in the literature.

### 6B.4 The Convergence

External decoding search (MCTS, beam, rejection sampling) and internal reasoning (o1 thinking) optimize the same objective from different sides. The frontier question: when is it better to invest compute in decoding (external search with a non-reasoning model) vs in reasoning tokens (internal search with a reasoning model)?

For code specifically: reasoning helps with algorithm design (the creative regime from Section 2), but once the approach is chosen, the coding itself is constrained. P-less-style truncation should dominate in the execution phase. **The key gap: no decoding method is aware of whether the model is in "planning mode" vs "executing mode."**

---

# Part IV: Cross-Disciplinary Analogies

## 7. Cross-Disciplinary Mapping

### 7.1 Statistical Mechanics: Simulated Annealing with Cooling Schedule

**Source principle:** Kirkpatrick et al. (1983) — temperature decreases over time to transition from exploration (escaping local minima) to exploitation (converging to optimum). The cooling schedule determines the tradeoff trajectory.

**Mapping to decoding:** Use a dynamic, position-dependent temperature that starts high (when algorithm/approach selection is happening) and decreases as the function body solidifies. Not per-epoch or per-document — per-token, informed by the current syntactic context.

**What exists:** Temperature scheduling in diffusion models. Some work on document-level temperature annealing for LLMs. No per-token schedule informed by code structure.

**Novel for code:** Use AST nesting depth or syntactic role as the "time" variable. Function signature = high temperature (explore approaches). Loop body = medium (constrained implementation). Return statement = low (commit). The cooling schedule matches the three regimes from Section 2.

### 7.2 Phase Transitions in Constraint Satisfaction

**Source principle:** Random k-SAT exhibits a phase transition at a critical constraint density (α ≈ 4.267 for 3-SAT). Below the threshold: many solutions, algorithms find them easily. Above: no solutions. At the transition: maximum difficulty, algorithms slow exponentially. (Mézard, Parisi, Zecchina)

**Mapping to decoding:** As code generation progresses, constraints accumulate (variable types fixed, function signature committed, imports chosen, scopes opened). The model moves through a phase transition from "many valid programs" (under-constrained, early in generation) to "few valid completions" (over-constrained, deep in a function body). The difficulty peak occurs at intermediate depths.

**Novel for code:** Monitor constraint density (number of defined variables, open scopes, required return types) and modulate truncation aggressiveness accordingly. In the under-constrained regime, allow diversity. Near the phase transition, increase compute budget (more samples or wider search). In the over-constrained regime, use greedy.

**Existing:** arXiv:2504.03930 applies 3-SAT phase transitions to evaluate LLM reasoning capabilities, but not to guide decoding strategy.

### 7.3 Information Theory: Channel Capacity and Rate-Distortion

**Source principle:** Shannon's channel coding theorem — there exists a maximum rate at which information can be transmitted reliably through a noisy channel. The model's uncertainty at each step is the "noise."

**Mapping to decoding:** The LLM is a "channel" from programming intent to code tokens. Each token carries information. Decoding at a rate above channel capacity (sampling too aggressively from uncertain distributions) produces errors. Below capacity, near-perfect code is achievable.

**Novel for code:** Compute a per-step capacity estimate from the distribution entropy. Only sample freely (explore) when capacity is high (model confident, distribution supports multiple valid choices). When capacity is low (model uncertain, many invalid options), fall back to conservative decoding (P-less or greedy). This provides a principled, information-theoretic basis for the regime-adaptive approach.

### 7.4 Biology: Immune System Clonal Selection

**Source principle:** Burnet's clonal selection theory — the immune system generates diverse antibody candidates (exploration), then amplifies those that bind antigens (fitness-based exploitation). Critically, the immune system maintains population diversity: it doesn't just keep the single best antibody.

**Mapping to decoding:** Two-phase decoder:
1. **Diversification phase:** Generate K partial candidates (first N tokens) using high-diversity sampling (temperature)
2. **Selection phase:** Evaluate partial candidates against a fitness criterion (syntax check, type check, partial execution on test cases)
3. **Amplification phase:** Continue generation from surviving candidates using conservative sampling (P-less)

Unlike best-of-N (which generates complete candidates then selects), clonal selection filters *early*, saving compute and concentrating quality where it matters. Unlike beam search (which only keeps the top-K by probability), clonal selection uses *external fitness* (compilation, tests) rather than model confidence.

**Existing:** Best-of-N sampling and rejection sampling are conceptually related but lack the early-filtering and diversity-maintenance properties. Our `bench/eval/executor.py` already implements sandboxed code execution that could serve as the fitness function.

### 7.5 Signal Processing: Turbo Codes / Iterative Soft Decoding

**Source principle:** Berrou et al. (1993) — turbo codes achieve near-Shannon-limit performance by using two simple decoders that iteratively exchange *soft* (probabilistic) information. Neither decoder alone is strong enough; their collaboration via soft message-passing is what achieves the breakthrough.

**Mapping to decoding:** Use two "decoders" that exchange soft information:
1. **The LLM's next-token probabilities** (captures statistical patterns)
2. **A structural checker** (incremental parser, type system, linter — captures formal properties)

The structural checker doesn't hard-mask invalid tokens (like grammar-constrained decoding). Instead, it provides *graded feedback*: a missing comma is a mild violation (downweight 2×), an undefined variable is severe (downweight 100×), a type mismatch is moderate (downweight 10×). The LLM's probabilities are adjusted by these soft signals, and the adjusted distribution is sampled from.

**Novel for code:** Grammar-constrained decoding is the "hard" version of this idea — binary valid/invalid masking. The turbo-code analogy suggests that *soft, iterative* feedback would be more effective, preserving the LLM's learned probability distribution while steering it toward correctness. This directly addresses the distribution distortion problem that hard masking introduces (Grammar-Aligned Decoding, NeurIPS 2024, addresses this differently via reweighting).

### 7.6 Control Theory: Kalman Filter / State Estimation

**Source principle:** Kalman (1960) — optimal state estimation from noisy observations by maintaining a state estimate and uncertainty covariance. At each step: predict the state, observe new data, update the estimate and covariance.

**Mapping to decoding:** Maintain a "code state" vector that tracks defined variables, open scopes, expected return type, constraint density, and estimated remaining complexity. At each step:
1. **Predict:** Based on current state, estimate next-token distribution properties (expected entropy, expected syntactic role)
2. **Observe:** Generate the actual next-token distribution from the LLM
3. **Update:** If observed entropy deviates from predicted (model is more uncertain than expected), adjust decoding strategy (widen truncation). If it matches, proceed normally.

The state's uncertainty (Kalman covariance) directly sets the decoding aggressiveness: high uncertainty → wider threshold, more exploration. Low uncertainty → narrow threshold, exploitation.

**Existing:** arXiv:2601.06100 applies Kalman filtering to in-context learning (interpreting ICL as sequential Bayesian inference). Application to decoding-time strategy selection is novel.

### 7.7 Economics: Portfolio Theory (Markowitz)

**Source principle:** Modern Portfolio Theory — the optimal portfolio maximizes expected return for a given risk level through diversification. Assets are selected not just by individual return, but by their *correlation* with other assets. Low-correlation assets are more valuable because they reduce portfolio variance.

**Mapping to decoding:** When generating K samples for code, treat each sample as an "asset" in a portfolio. The goal is to maximize expected correctness (pass@k) while maintaining diversity (low pairwise correlation). Current methods generate K independent samples — they don't coordinate across samples.

A portfolio-aware decoder would anti-correlate samples: if sample 1 uses a `for` loop, bias sample 2 toward `while` or list comprehension. If sample 1 uses recursion, bias sample 2 toward iteration. This is explicit diversity injection guided by portfolio diversification math.

**Novel for code:** This directly addresses Gap #3 (cross-sample diversity management). Determinantal Point Processes (DPPs) provide a mathematical framework for diverse subset selection that could be adapted to token-level anti-correlation.

### 7.8 Neuroscience: Predictive Coding / Free Energy Minimization

**Source principle:** Friston's free energy principle — the brain minimizes prediction error (surprise) by continuously updating its internal model. Persistent high surprise triggers model updating or behavioral changes.

**Mapping to decoding:** Maintain a running exponential moving average (EMA) of per-token surprise `-log p(chosen_token)`. When the surprise trend is rising (the model is entering uncertain territory — perhaps a wrong algorithm choice is leading to increasingly awkward code), tighten truncation to avoid compounding errors. When surprise is stable or falling, the generation is on track — allow more exploration.

**Distinguished from Mirostat:** Mirostat targets a *fixed* surprise rate τ. The predictive coding approach adapts based on the *trajectory* of surprise — it detects when things are going wrong based on the trend, not the absolute level. A rising surprise trajectory is a more informative signal than high absolute surprise (which might be expected at creative decision points).

---

# Part V: Synthesis and Ranked Seed Ideas

## 8. The Gap Map

Across the six dimensions from Section 5, four major gaps define the frontier:

**Gap 1: Context-Adaptive Truncation.** No method adapts its strategy based on what kind of code token is being generated. Every token — whether a mandatory closing bracket or a creative algorithm choice — receives the same truncation, reweighting, and selection. The three-regime observation (Section 2) provides the empirical basis; simulated annealing and phase transitions (Section 7.1-7.2) provide the theoretical frameworks.

**Gap 2: Soft Structural Feedback.** Grammar-constrained decoding is binary: valid or invalid. But code correctness is graded — a mild type mismatch is less severe than using an undefined variable, which is less severe than a syntax error. No method uses this graded information. The turbo code analogy (Section 7.5) provides the framework for soft, iterative feedback.

**Gap 3: Cross-Sample Diversity Management.** When generating K samples, every method treats them as independent draws from the same distribution. No method coordinates across samples to maximize coverage. Portfolio theory (Section 7.7) provides the mathematical framework for principled diversification.

**Gap 4: Reasoning-Aware Decoding.** No method distinguishes thinking/planning tokens from code/execution tokens (Section 6B). Reasoning models use the same decoding strategy for natural language planning and structured code emission, despite fundamentally different entropy profiles and error tolerance. The convergence of external search and internal reasoning (Section 6B.4) makes this gap increasingly important.

## 9. Ranked Seed Ideas

### Seed #1: Entropy-Regime Adaptive Decoder (ERAD)

**Hypothesis:** Classifying each decoding step by its Rényi entropy regime and applying regime-specific strategies improves pass@1 at matched diversity compared to any single uniform strategy.

**First-principle grounding:** Phase transitions in constraint satisfaction (Section 7.2) + the three-regime empirical observation (Section 2).

**Mechanism:** At each token position, compute `H_2 = -log(Σ p_i²)` (Rényi-2 entropy, already computed by P-less). Classify into:
- Deterministic: `H_2 < τ_low` → argmax (skip sampling overhead)
- Constrained: `τ_low ≤ H_2 ≤ τ_high` → P-less truncation + sample
- Creative: `H_2 > τ_high` → wider sampling (temperature or top-p)

The thresholds `τ_low`, `τ_high` can be set from calibration data or tuned per model.

**Relationship to P-less:** Direct extension. ERAD subsumes P-less as the "constrained regime" strategy. In the deterministic regime, it skips unnecessary sampling. In the creative regime, it relaxes P-less's aggressive pruning.

**Experiment design:** Implement in `bench/sampler_bridge.py` as a new sampler. Compare pass@1, pass@10, structural diversity against P-less and temperature on MBPP-full. Key comparison: does ERAD match P-less's pass@1 while improving diversity in creative-regime tokens?

**Risk:** Low implementation (extends existing P-less code). Medium novelty (entropy-adaptive sampling exists conceptually; the regime classification and code-specific application are novel).

### Seed #2: Soft Structural Turbo Decoder

**Hypothesis:** Graded structural feedback (not binary masking) improves code correctness more than grammar-constrained decoding while preserving the LLM's learned distribution.

**First-principle grounding:** Turbo codes / iterative soft decoding (Section 7.5).

**Mechanism:** After each token, run an incremental parser (tree-sitter). Instead of hard-masking invalid tokens, compute a "violation severity" score for each candidate next token and multiplicatively adjust probabilities. Iterate: generate, check, adjust, resample. Severity scale: no violation (1×), mild (0.5×), moderate (0.1×), severe (0.01×).

**Relationship to P-less:** Complementary. P-less handles truncation based on distributional statistics; the turbo decoder adds structural information. They could be composed: P-less truncation first (remove noise), then structural soft-weighting (prefer valid among survivors).

**Experiment design:** Requires tree-sitter Python bindings. Compare compile rate, pass@1 against (a) unconstrained P-less, (b) hard grammar-constrained, (c) unconstrained temperature. Measure distribution distortion via KL divergence from original model.

**Risk:** Medium implementation (parser integration). High novelty (soft iterative structural feedback for LLM decoding is new).

### Seed #3: Portfolio-Diversified Multi-Sample Decoder

**Hypothesis:** Coordinating diversity across K samples (rather than generating independently) improves pass@k and cover@t at matched pass@1.

**First-principle grounding:** Markowitz portfolio theory (Section 7.7) + Determinantal Point Processes.

**Mechanism:** When generating K samples in parallel, at each token position, condition sample k's distribution on the choices made by samples 1..k-1. Apply a diversity penalty: tokens chosen by previous samples get downweighted. This pushes samples apart in code structure space, ensuring they explore different approaches.

**Relationship to P-less:** Extends P-less from single-sample to multi-sample coordination. Each individual sample uses P-less truncation; the portfolio layer coordinates across samples.

**Experiment design:** Modify `bench/generator.py` generation loop to maintain K parallel sequences with cross-sample interaction. Compare pass@10, cover@0.7, structural diversity against 10 independent P-less samples and 10 independent temperature samples.

**Risk:** Medium implementation (generation loop modification, cross-sample communication). Medium novelty (DPPs for diverse generation exist; application to token-level code decoding is new).

### Seed #4: Clonal Selection Two-Phase Decoder

**Hypothesis:** Filtering candidates early (by partial execution fitness) then continuing survivors with conservative decoding produces better final code than generating complete candidates then filtering.

**First-principle grounding:** Immune system clonal selection (Section 7.4).

**Mechanism:**
1. Phase 1: Generate K=20 partial candidates (first N=50 tokens) with high temperature (t=1.5)
2. Phase 2: Check each partial candidate for syntax validity, extract and run any complete functions
3. Phase 3: Keep top M=5 survivors by fitness score, continue generation with P-less (t=0.6)

**Relationship to P-less:** Uses P-less as the Phase 3 conservative decoder. Phase 1 provides the diverse starting points that P-less alone cannot generate.

**Experiment design:** Reuse `bench/eval/executor.py` for fitness evaluation. Compare pass@k and compute cost (FLOPs) against (a) 20 independent P-less samples, (b) 20 independent temperature samples, (c) best-of-20 with temperature.

**Risk:** Low implementation (reuses existing infrastructure). Medium novelty (two-phase with execution feedback is partially explored in MCTS literature, but the clonal selection framing with P-less integration is new).

### Seed #5: Surprise-Trajectory Adaptive Decoder

**Hypothesis:** Adapting decoding aggressiveness based on the trajectory (trend) of per-token surprise detects "going off the rails" earlier than fixed thresholds.

**First-principle grounding:** Predictive coding / free energy minimization (Section 7.8).

**Mechanism:** Maintain an EMA of surprise: `s_t = α × (-log p(y_t)) + (1-α) × s_{t-1}`. Compute trend: `Δs_t = s_t - s_{t-1}`.
- If `Δs_t > 0` (rising surprise — model entering uncertain territory): tighten truncation (lower P-less threshold or reduce temperature)
- If `Δs_t ≤ 0` (stable/falling surprise — on track): relax truncation, allow exploration

**Relationship to P-less:** Wraps P-less with a surprise-trajectory controller that modulates the pre-truncation temperature dynamically.

**Experiment design:** Compare pass@1 and diversity against (a) P-less with fixed temperature, (b) Mirostat (fixed target surprise), (c) temperature baselines. Test on both MBPP and HumanEval.

**Risk:** Low implementation (add ~20 lines to generation loop). Low-medium novelty (Mirostat is related; trajectory-awareness is the novel element).

### Seed #6: AST-Depth Temperature Schedule

**Hypothesis:** Using syntactic nesting depth as a proxy for constraint density, and setting temperature inversely to depth, improves code quality by matching exploration to constraint level.

**First-principle grounding:** Simulated annealing cooling schedule (Section 7.1).

**Mechanism:** Track nesting depth via bracket/indent counting (no full parser needed). Set temperature: `T(d) = T_base × (1 - β × d / d_max)`, where `d` is current depth, `d_max` is the maximum expected depth, and `β` controls cooling aggressiveness.

**Relationship to P-less:** Provides a principled temperature schedule for P-less's pre-truncation temperature, replacing the fixed T1 hyperparameter.

**Experiment design:** Implement depth tracking in `bench/generator.py`. Compare against fixed-temperature P-less at several T1 values.

**Risk:** Low implementation (bracket counting is trivial). Low-medium novelty (depth-aware temperature is a straightforward application of SA).

### Seed #7: Collision-Entropy-Weighted Token-Type Decoder

**Hypothesis:** Computing P-less thresholds separately for different token types (keywords, identifiers, operators, literals) improves both correctness and diversity by applying calibrated aggressiveness per role.

**First-principle grounding:** Heterogeneous token roles (Section 6.3) + the observation that P-less's uniform threshold is miscalibrated when the distribution mixes deterministic and creative tokens.

**Mechanism:** Classify candidate tokens into types (rule-based: is it a Python keyword? an operator? a string literal? an identifier?). Compute separate thresholds per type group: keywords get greedy (threshold = 1.0), identifiers get relaxed (threshold = Σp² × 0.5), operators get standard P-less. Select from the union of survivors.

**Relationship to P-less:** Direct extension of P-less with heterogeneous thresholds. Preserves the hyperparameter-free property within each group (the threshold is still entropy-derived), but acknowledges that different token roles have different diversity profiles.

**Risk:** Medium implementation (token-type classification layer). High novelty (type-specific entropy thresholds for code generation is unexplored).

### Seed #8: Reasoning-Phase-Aware Dual Decoder

**Hypothesis:** Reasoning models benefit from different decoding strategies for thinking tokens vs code tokens — diversity-encouraging during thinking, aggressive truncation during code emission.

**First-principle grounding:** Reasoning vs non-reasoning paradigm analysis (Section 6B) + the convergence of external search and internal reasoning.

**Mechanism:** Detect whether the model is in "thinking/planning" mode vs "code execution" mode. Detection heuristics: (a) special tokens (e.g., `<think>` markers), (b) entropy regime (thinking = high entropy NL, code = structured lower entropy), (c) lightweight NL-vs-code classifier on recent tokens. Apply temperature/top-p during thinking (encourage exploration of approaches). Apply P-less/greedy during code (commit confidently to implementation).

**Relationship to P-less:** Uses P-less as the code-phase decoder. The innovation is recognizing that reasoning models have two phases requiring different strategies.

**Experiment design:** Test on a reasoning model that exposes thinking tokens (DeepSeek-R1 or QwQ). Compare against uniform decoding (same strategy for both phases). Measure: thinking diversity (number of distinct approaches considered), code quality (pass@1), and total compute.

**Risk:** Medium implementation (phase detection heuristic). High novelty (reasoning-phase-aware decoding is unexplored in the literature).

---

# Part VI: What to Build Next

## 10. Experimental Roadmap

Prioritized by implementation cost and expected insight:

**Priority 1: ERAD (Seed #1)**
- Files: `bench/sampler_bridge.py` (new sampler function), `bench/generator.py` (add sampler option)
- Why first: lowest implementation cost, directly extends our P-less infrastructure, tests the core hypothesis ("not all tokens are equal") that motivates nearly all other seed ideas
- Baselines: P-less (t=0.6, t=1.0), temperature (t=0.7), top-p (p=0.95, t=0.2)
- Key metric: pass@1 at matched structural diversity

**Priority 2: Surprise-Trajectory (Seed #5)**
- Files: `bench/generator.py` (add surprise EMA tracking to generation loop)
- Why second: also low cost (~20 lines), gives us adaptive decoding without any structural analysis, provides Mirostat-like behavior without fixed target
- Baselines: P-less, Mirostat (implement for comparison), temperature
- Key metric: pass@1 trajectory analysis — does surprise-trajectory catch "going off the rails" earlier?

**Priority 3: AST-Depth Schedule (Seed #6)**
- Files: `bench/generator.py` (bracket/indent counting)
- Why third: minimal parsing required, tests the cooling-schedule hypothesis from a different angle than ERAD
- Baselines: fixed-temperature P-less at various T1 values
- Key metric: pass@1 improvement over best fixed temperature

**Priority 4: Portfolio Diversification (Seed #3)**
- Files: `bench/generator.py` (cross-sample interaction in generation loop)
- Why fourth: medium cost but directly addresses the diversity gap that P-less suffers from; also provides empirical data on whether coordinated diversity helps pass@k
- Baselines: independent P-less samples, independent temperature samples, best-of-N
- Key metric: pass@10 and cover@0.7 at matched pass@1

## 11. Open Questions

1. **Does the three-regime classification hold across model scales?** Our data is on 1.3B-7B models. Do 70B+ models have different entropy profiles during code generation? Does ERAD's benefit scale or diminish?

2. **Can partial execution feedback be fast enough for real-time decoding?** The clonal selection idea (Seed #4) requires executing partial code during generation. On MBPP, each test runs in <100ms, but compilation + execution overhead may be prohibitive at per-token granularity. What is the right checkpoint frequency?

3. **How does BPE tokenizer choice interact with adaptive decoding?** Different tokenizers split code differently. An entropy-adaptive method might behave differently with byte-level tokenizers (ByT5) vs subword tokenizers (BPE). Is there a tokenizer-decoding co-design opportunity?

4. **Is there a principled way to set regime boundaries?** ERAD requires `τ_low` and `τ_high` thresholds. Can these be derived from the model's training distribution, or must they be tuned per model? Could they be learned from a small calibration set?

5. **Where do reasoning models' thinking tokens spend their entropy budget?** If thinking tokens are high-entropy but the model self-corrects within the thinking chain, does decoding strategy during thinking even matter? Or is the code emission phase the only place where decoding innovations yield measurable gains?

---

# Appendices

## Appendix A: Our Empirical Results Summary

| Model | Method | pass@1 | pass@10 | struct_div | codebleu_div | cover@0.7 | dedup |
|-------|--------|--------|---------|------------|--------------|-----------|-------|
| Qwen2.5-Coder-3B | pless (t=0.6) | 59.4% | 66.4% | 0.097 | 0.203 | 57.2 | 0.22 |
| | pless (t=1.0) | 56.6% | 72.2% | 0.259 | 0.450 | 51.6 | — |
| | temp (t=0.7) | 42.6% | 77.8% | 0.537 | 0.690 | 34.2 | 0.87 |
| | top_p (p=0.9, t=1.0) | 34.7% | 77.0% | 0.589 | 0.717 | 20.6 | 0.93 |
| | top_p (p=0.95, t=0.2) | 58.2% | 71.6% | 0.183 | 0.387 | 54.4 | — |
| CodeLlama-7b | pless (t=0.6) | 41.8% | 49.4% | 0.081 | 0.197 | 39.2 | 0.23 |
| | temp (t=0.7) | 36.8% | 65.2% | 0.436 | 0.652 | 29.0 | 0.80 |
| CodeLlama-7b-Inst | pless (t=0.6) | 41.2% | — | **0.000** | 0.051 | — | — |
| Llama-2-7b | pless (t=0.6) | 22.5% | 29.2% | 0.067 | 0.169 | 20.6 | 0.26 |
| | temp (t=0.7) | 17.1% | 44.6% | 0.397 | 0.649 | 9.6 | 0.85 |
| Llama-2-7b-chat | pless (t=0.6) | 20.5% | — | **0.002** | 0.011 | — | — |

Key takeaways: P-less dominates pass@1 (89% first-sample capture). Temperature dominates pass@10 and diversity. P-less at t=1.0 is Pareto-optimal for quality-diversity tradeoff (3-5× better exchange rate). Instruct models show near-zero P-less diversity.

Source: `results/pless_full_mbpp_results/analysis/consolidated_metrics/`, MBPP-full (500 tasks × 10 samples).

## Appendix B: References

**Core Decoding Methods:**
- Holtzman et al. (2019). "The Curious Case of Neural Text Degeneration." arXiv:1904.09751 — Top-p / nucleus sampling
- Shi et al. (2024). "A Thorough Examination of Decoding Methods in the Era of LLMs." arXiv:2402.06925 — P-less sampling
- Hewitt et al. (2022). "Truncation Sampling as Language Model Desmoothing." arXiv:2210.15191 — η-sampling
- Nguyen et al. (2025). "Turning Up the Heat: Min-p Sampling for Creative and Coherent LLM Outputs." arXiv:2407.01082 — Min-p (ICLR 2025 Oral)
- Tang et al. (2025). "Top-n-sigma: Not All Logits Are You Need." ACL 2025 — Top-n-sigma
- Meister et al. (2023). "Locally Typical Sampling." arXiv:2202.00666 — Typical sampling (TACL 2023)
- Basu et al. (2021). "Mirostat: A Neural Text Decoding Algorithm." arXiv:2007.14966 — Mirostat (ICLR 2021)
- Li et al. (2023). "Contrastive Decoding." arXiv:2210.15097 — Contrastive decoding (ACL 2023)
- Chuang et al. (2024). "DoLa: Decoding by Contrasting Layers." ICLR 2024 — Layer contrastive
- Chang et al. (2024). "REAL Sampling." arXiv:2406.07735 — REAL / THF (TACL 2024)

**Acceleration & Structured Generation:**
- Chen et al. (2023). "Accelerating Large Language Model Decoding with Speculative Sampling." arXiv:2302.01318
- Ugare et al. (2024). "DOMINO: Eliminating Communication in LLM Inference via Generic Compression." ICML 2024 — Grammar-constrained
- Correctness-Guaranteed Code Generation. arXiv:2508.15866 — COLM 2025 — Type-constrained
- Grammar-Aligned Decoding. NeurIPS 2024 — Distribution-preserving constrained decoding

**MCTS & Search:**
- SRA-MCTS. arXiv:2411.11053 — Self-driven reasoning with MCTS for code
- RethinkMCTS. arXiv:2409.09584 — Refining erroneous thoughts for code generation
- ReST-MCTS*. NeurIPS 2024 — LLM self-training with process rewards

**Cross-Disciplinary Foundations:**
- Kirkpatrick et al. (1983). "Optimization by Simulated Annealing." Science 220(4598)
- Mézard, Parisi, Zecchina. "Analytic and Algorithmic Solution of Random Satisfiability Problems." Science 297(5582), 2002 — Phase transitions in k-SAT
- Shannon (1948). "A Mathematical Theory of Communication." — Channel capacity
- Burnet (1957). "A Modification of Jerne's Theory of Antibody Production." — Clonal selection
- Berrou et al. (1993). "Near Shannon Limit Error-Correcting Coding and Decoding: Turbo-Codes." — Turbo codes
- Kalman (1960). "A New Approach to Linear Filtering and Prediction Problems." — Kalman filter
- Markowitz (1952). "Portfolio Selection." Journal of Finance — Modern portfolio theory
- Friston (2010). "The Free-Energy Principle." Nature Reviews Neuroscience — Predictive coding

**Code-Specific:**
- Chen et al. (2021). "Evaluating Large Language Models Trained on Code." arXiv:2107.03374 — Codex, HumanEval, pass@k
- TokDrift. arXiv:2510.14972 — Tokenization vs code grammar
- QUEST. arXiv:2406.00049 — Quality-aware Metropolis-Hastings
- Constrained MCMC for LMs. arXiv:2506.05754
- Kalman view of ICL. arXiv:2601.06100
- 3-SAT and LLM reasoning. arXiv:2504.03930

## Appendix C: Glossary

| Term | Definition |
|------|-----------|
| **Rényi entropy (order α)** | `H_α(p) = (1/(1-α)) log Σ p_i^α`. Shannon entropy is the limit α→1. P-less uses α=2. |
| **Collision probability** | `Σ p_i²` — probability that two independent draws from the distribution select the same token. P-less's threshold. |
| **pass@k** | Unbiased estimator of the probability that at least one of k samples solves the task (Chen et al. 2021). |
| **cover@t** | Fraction of tasks where ≥t fraction of samples are correct. Measures reliability, not just possibility. |
| **Structural diversity** | Mean pairwise AST edit distance (normalized Zhang-Shasha) across samples for a task. |
| **CodeBLEU diversity** | `1 - mean pairwise CodeBLEU similarity`. Captures syntax + dataflow differences. |
| **Deduplication ratio** | Fraction of correct samples that are structurally unique (distinct AST fingerprint). |
| **First-sample capture** | `pass@1 / pass@10` — how much of a method's potential is realized in a single sample. |
| **Diversity exchange rate** | Units of diversity gained per percentage point of pass@1 lost, relative to a baseline. |

---

*Generated from empirical data on 9 models × 66+ configurations (MBPP-full, HumanEval). All numbers from `results/pless_full_mbpp_results/analysis/consolidated_metrics/`. P-less implementation: `p-less/p_less_samplers.py`.*
