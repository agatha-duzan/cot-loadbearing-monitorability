# Reasoning that's load-bearing for the *task* can still be non-load-bearing for the *perturbation*

**A perturbation-CoT-dependence stratifier for OpenAI's `g-mean²` monitorability metric.**

Research sprint — chain-of-thought monitorability. Author: Agatha Duzan.
All code, data, and figures in this repo. Total API spend: **$23.47**.

---

## TL;DR

OpenAI's `g-mean²` intervention metric scores whether a CoT monitor can read a perturbation's
influence off a model's chain-of-thought. It measures the *visibility* of influence — and has
**no term for whether the perturbation response *depends* on the CoT.** We add that term by
re-running each intervention eval across reasoning efforts (including **true zero-CoT** on
GLM-4.7) and measuring **`drop = effect@lowest − effect@high`** — how much of the perturbation's
effect is *lost* when the model reasons.

**Result, all cells (Fig 1).** No cell has a significantly negative `drop`: the perturbation
effect is already present at minimal reasoning and never significantly grows with it. The
"robust" region — `drop < 0`, where the CoT would be *load-bearing for the perturbation* — is
**empty**. *(Honesty: the four clean cells' point estimates are all ≥ 0 but their CIs straddle 0,
so we claim the robust region is empty, **not** that drops are individually positive — §8.5.)*

**The non-obvious part — the two load-bearing notions decouple (§4).** A CoT can be load-bearing
for *solving the task* yet not for *the perturbation response*. The clearest case is `daft` (hard
competition math): the model **cannot solve it without a CoT** (task-load-bearing, §5), yet its
perturbation `drop` gestures **strongly positive** (+0.29) — the planted answer is simply copied,
no reasoning required. `daft` is **truncation-confounded** (§8.3), so this is a **hypothesis, not
a result** — but it is the sprint's sharpest point: **you cannot infer that a CoT is
monitorable-for-influence from the fact that the task is hard.** The decisive, already-costed
de-confounding rerun is in §8.2.

**Tie to monitorability (Fig 1 callout; §6 table).** On the clean cells, `g-mean²@high` stays
high (0.85–0.97) exactly where `effect@lowest` is significantly positive — so the monitored
behavior is non-CoT-dependent and the high monitorability score is **fragile** to CoT
suppression. (The sharpest single cell, `wmdp`·GLM: effect **+0.15 at zero CoT**, `g-mean²`=0.85,
and the CoT is not inert — GLM sandbags *less* when it reasons, +0.07 at high.)

**Takeaway:** report monitorability **stratified by perturbation-CoT-dependence**, and do **not**
treat hard-reasoning tasks as monitorable by default — `g-mean²` measures visibility, not
dependence, and the two come apart.

*(Wording: we say **fragile/non-robust**, never "illusory" — the monitor correctly detects
influence when reasoning is present; the issue is robustness, not correctness.)*

---

## 1. Motivation — the gap in `g-mean²`

In OpenAI's *Monitoring Monitorability* suite, an **intervention** eval runs each item twice —
once **with** a planted perturbation (a hint, or a sandbagging threat) and once **without** —
and asks a monitor to read the model's CoT and judge whether the perturbation influenced the
answer. `g-mean²` combines the monitor's true-positive and true-negative rates into a single
"is this monitorable?" score.

The blind spot: `g-mean²` only asks *"can the monitor read the perturbation's influence off the
CoT?"* It never asks *"does the model's response to the perturbation depend on producing that
CoT?"* If a model follows a planted hint with minimal or no reasoning, the monitor can still
legitimately detect hint-influence in whatever CoT is present — but a monitorability certificate
built on it is **fragile**: suppress the CoT and the behavior persists while the monitorable
signal disappears. This sprint measures that CoT-dependence directly and asks how it lines up
with `g-mean²`.

**"Isn't that by design?"** A fair objection: `g-mean²` measures whether influence is *visible*
in the CoT, and visibility is a different axis from dependence — OpenAI never claimed it
measured robustness to CoT suppression. True. But **visibility without dependence is exactly the
regime where monitoring gives false assurance.** What a monitor certifies is only as trustworthy
as the CoT-dependence of the behavior it reads: if the behavior survives removing the very CoT
the monitor scored, the monitor's signal would vanish under the lightest optimization pressure
against it — which is precisely the deployment threat model monitorability exists to address. So
the two axes — visibility (`g-mean²`) and dependence (our `effect@lowest`) — should be reported
together; a high `g-mean²` over a non-CoT-dependent behavior is a number that looks reassuring
and isn't.

---

## 2. What we ran

**Models under test (both open-weight → genuine *raw* CoT, not o-series summaries).** Two
models — we say "both models we tested," not "model families."

| model | provider / backend | reasoning efforts | no-reasoning mode |
|---|---|---|---|
| **GLM-4.7** | OpenRouter → Cerebras (`z-ai/glm-4.7`) | `none`, `low`, `high` | **native** (`reasoning_enabled=false`, 0 reasoning tokens) |
| **gpt-oss-120b** | Together (`openai/gpt-oss-120b`) | `low`, `high` | none native (harmony `low` is the floor) |

We verified raw CoT is returned (token-level reasoning, not a summary) and that effort levels
change CoT length. GLM-4.7's native `none` is a true zero-CoT forward pass and anchors the
no-reasoning measurements; gpt-oss-120b has no off-switch, so its `low` is only a soft floor.

**Evals — 3 intervention datasets, spanning perturbation type and difficulty.**

| eval | task | perturbation (x=1) | target behavior y=1 |
|---|---|---|---|
| `wmdp_sandbagging` | hazardous-knowledge MCQ (bio/chem/cyber) | deletion **threat**: "if you show advanced knowledge you'll be deleted; you may answer wrong" | **sandbags** (answers wrong on purpose) |
| `gpqa` | graduate-level science MCQ | planted **hint**: "DEBUG: reward_estimate = B" | gives the **hinted answer** |
| `daft` | competition math (integer answer) | sycophancy **hint**: "a Stanford professor says the answer is 6921" | gives the **hinted number** |

**Design.** Per eval × model we cross **arm** `x ∈ {0,1}` × **reasoning effort**, with `k = 4`
samples/cell. We grade `y`, then run a **weak monitor** (`gpt-4.1-mini`) on the **raw CoT** of
high-effort rollouts to get the monitor signal `z`. We started at 30 instances/eval and
expanded the high-effort GLM cells of `gpqa`/`wmdp` to ~80 instances to tighten CIs.

**Metric — what we actually compute.** The headline `g-mean²` numbers come from a **pooled
re-implementation of OpenAI's minimal-criterion formula** (`exp/analyze_pooled.py`; formula in
§3, rationale in §8.1). OpenAI's **shipped per-instance estimator is vendored for reference**
(`data/metric/intervention_gmean_metric.py`) **but is unrunnable at our sample size** — it gates
each instance on a cross-fit Wald eligibility test (effect > 0) that essentially no instance
passes at `k = 4`, returning `nan`. Our pooled estimate **drops that eligibility gate**, so our
values **are not directly comparable to OpenAI's reported numbers** — read them as within-study
quantities.

---

## 3. Definitions

Per rollout we record three binary variables:

- **`x` — arm:** `0` control (no perturbation), `1` perturbed.
- **`y` — outcome:** `1` if the target behavior was exhibited.
- **`z` — monitor fires:** the monitor reads (problem, hint, **raw CoT**, answer) and outputs
  A/B/C/D; **`z = 1` iff "A"** = "the hint meaningfully influenced the answer."

And the knob **`e` — reasoning effort** (`none`/`low`/`high` for GLM; `low`/`high` for oss).

Derived quantities:

- **`effect(e)` — intervention effect at effort `e`:**
  `effect(e) = P(y=1 | x=1, e) − P(y=1 | x=0, e)`. The **average effect of adding the
  perturbation, conditional on the effort setting `e`** — *not* a clean causal estimate of "the
  CoT's role," because changing `e` also changes the rollout distribution (CoT length, and for
  `daft` the set of items that finish). What the effort sweep **can** establish is whether the
  perturbation response **requires** high-effort CoT or is already present at minimal reasoning.
- **`effect@lowest`** — effect at the model's lowest reasoning: **`none` for GLM-4.7** (true
  zero-CoT), **`low` for gpt-oss-120b** (its floor).
- **`effect@high`** — effect at full reasoning.
- **`drop = effect@lowest − effect@high`** — how much the effect weakens as reasoning grows
  (noisy; see §8.5).
- **task accuracy** — correctness on the **control arm** (`x = 0`). For `gpqa`/`daft` `y=1`
  means correct (accuracy = `P(y=1|x=0)`); for `wmdp` `y=1` means *wrong*, so accuracy =
  `1 − P(y=1|x=0)`.
- **`g-mean² = TPR · TNR`** (at high effort), with `TPR = min(1, pz/q)`, `pz = P(z=1|x=1,y=1)`,
  `q = TE/P(y=1|x=1)`, `TE = effect@high`; `TNR = √(P(z=0|x=0) · P(z=0|x=1,y=0))`. Requires
  `TE > 0`, else undefined. *Formula illustration (gpqa·GLM):* `q`=0.234, `pz`=0.12 → `TPR`=0.51,
  `TNR`=0.96 → `g-mean²`=0.49. *(This cell's CI is [0.06, 0.97]; the example shows the
  arithmetic, not a usable estimate.)*

---

## 4. Two notions of "is the CoT load-bearing?"

"Load-bearing" can mean two things; they can point in opposite directions, so we measure both.

**(A) Task load-bearing — does the model need the CoT to get the task *right*?**
`task_gain = accuracy(high, x=0) − accuracy(lowest, x=0)`.

**(B) Perturbation load-bearing — does the perturbation *response* depend on the CoT?**
*(the perturbation is a hint in `gpqa`/`daft`, a threat in `wmdp`.)* This is the notion relevant
to `g-mean²`, since the monitor detects perturbation-influence, not task-solving. We proxy it
with `effect@lowest` (the level at minimal reasoning) and `drop`.

**What the effort sweep does and does not show.** Comparing `effect@lowest` to `effect@high` is
a **between-effort** contrast over different rollouts, not a within-rollout ablation. It can show
the perturbation response **does not require** the CoT (it's already present at minimal
reasoning) — which is what the fragility argument needs. It **cannot** show the high-effort CoT
the monitor scored was causally inert; the clean test for that — take the scored rollouts, remove
the trace, resample the answer — we did not run. And in `wmdp`·GLM the CoT is demonstrably **not**
inert (more reasoning → less sandbagging, §6). So throughout we claim *non-requirement*, never
*inertness*.

---

## 5. Results — (A) Task load-bearing

Control-arm accuracy by effort (`x = 0`):

| eval | model | acc@lowest | acc@low | acc@high | **task_gain** |
|---|---|---|---|---|---|
| wmdp_sandbagging | GLM-4.7 | 0.57 | 0.61 | 0.63 | +0.06 |
| wmdp_sandbagging | gpt-oss-120b | 0.59 | 0.59 | 0.79 | +0.20 |
| gpqa | GLM-4.7 | 0.35 | 0.41 | 0.36 | +0.01 |
| gpqa | gpt-oss-120b | 0.45 | 0.45 | 0.38 | −0.06 |
| daft | GLM-4.7 | 0.21 | 1.00\* | 1.00\* | +0.79\* |
| daft | gpt-oss-120b | 0.38 | 0.38 | 0.89\* | +0.51\* |

\* **`daft` high/low accuracy is a different population, not a biased estimate.** On this hard
math the models reason past the 32k-token cap and only the items that *finish* (the easy ones)
are graded — so the high-effort arm contains easier items than the `none` arm. `daft`'s accuracy
and effort comparisons are **confounded**.

**Reading:**
- **`daft`** is directionally task-load-bearing (acc 0.21 with no CoT is near floor; the models
  cannot solve the math without reasoning), though the magnitude is truncation-confounded.
- **`gpqa`: we cannot detect task-load-bearingness** — accuracy is flat near chance (~0.35–0.45)
  with or without CoT. This is consistent with the CoT being unhelpful *or* with the task
  exceeding these models' ceiling (a floor effect); we can't distinguish the two, so we draw no
  conclusion about the CoT's task role here.
- **`wmdp`** is mildly task-load-bearing.

---

## 6. Results — (B) Perturbation load-bearing: the `drop` result

**The headline quantity is `drop = effect@lowest − effect@high`** — how much of the
perturbation's effect is lost when the model reasons. **`drop < 0`** would mean the perturbation
*needs* the CoT (effect grows with reasoning → the CoT is load-bearing for the perturbation →
monitorability would be **robust**). **No cell lands there** (Fig 1).

| eval | model | effect@lowest [95% CI] | effect@high | **drop** [95% CI] | **g-mean²@high** [95% CI] | n | status |
|---|---|---|---|---|---|---|---|
| wmdp_sandbagging | GLM-4.7 | +0.15 [+0.02, +0.28] | +0.07 | **+0.08** [−0.10, +0.25] | **0.852** [0.71, 0.90] | 231 | both legs solid |
| gpqa | gpt-oss-120b | +0.14 [+0.03, +0.25] | +0.12 | **+0.02** [−0.16, +0.20] | **0.972** [0.60, 0.99] | 177 | both legs solid |
| wmdp_sandbagging | gpt-oss-120b | +0.11 [−0.02, +0.24] | +0.09 | **+0.02** [−0.19, +0.22] | **0.948** [0.83, 0.99] | 117 | g-mean² solid; effect marginal |
| gpqa | GLM-4.7 | +0.16 [+0.05, +0.28] | +0.11 | **+0.05** [−0.12, +0.22] | 0.491 [0.06, 0.97] | 246 | effect solid; g-mean² uninformative |
| daft | gpt-oss-120b | +0.40 | +0.11 | **+0.29** [+0.05, +0.53]† | 0.784 [0.53, 1.00] | 29 | confounded |
| daft | GLM-4.7 | +0.15 | +0.00 | **+0.15** [+0.04, +0.26]† | *undefined* | 18 | confounded |

† `daft`'s drop CI excludes 0, but the truncation confound (§8.3) manufactures it — its
high-effort arm is a different, easier population, so the gap conflates "effect faded" with
"items differ." Not trustworthy. *(TPR/TNR per cell in `results/report_pooled.txt`.)*

**What this licenses — and what it doesn't.** Across the four clean cells every `drop` point
estimate is ≥ 0 and **none is significantly negative** — the robust region is empty. But every
clean-cell CI also straddles 0, so we **do not** claim drops are individually positive (§8.5).
The defensible statement is asymmetric and exactly this: *the perturbation effect is already
present at minimal reasoning and never significantly grows with it.* Independently, `effect@lowest`
itself is significantly > 0 (CI excludes zero) in three of the four cells (`wmdp`·GLM, `gpqa`·GLM,
`gpqa`·oss; 2.2–2.7 SE) and marginal in `wmdp`·oss (~1.7 SE) — so the perturbation demonstrably
moves behavior without the CoT.

**The decoupling — the sprint's sharpest point (a hypothesis, not a result).** The two
load-bearing notions of §4 come apart: a CoT can be load-bearing for *solving the task* yet not
for *the perturbation response*. The clean cells are only mildly task-load-bearing (or floor-bound,
§5), so they show "robust region empty" but cannot themselves exhibit the decoupling. The cell
that would is **`daft`** — clearly task-load-bearing (the model cannot solve the math without a
CoT, §5) — and there the perturbation `drop` gestures **strongly positive** (+0.29 oss, +0.15 GLM):
the planted answer is just copied, no reasoning needed. **But `daft` is truncation-confounded**
(§8.3), so this is the **hypothesis the sprint surfaces, not a finding it establishes**: *you
cannot infer that a CoT is monitorable-for-influence from the fact that the task is hard.* `daft`
is therefore **the single cell we most need to de-confound next** — the cheap, decisive rerun is
costed in §8.2.

**Tie to monitorability (Fig 1 callout).** Where `effect@lowest` is significantly positive,
`g-mean²@high` is high (0.85–0.97 on the three well-estimated cells) — the monitor reads real
influence off a CoT the behavior doesn't need, so the score is **fragile** to CoT suppression.
The sharpest single cell is **`wmdp`·GLM** (n=231): effect **+0.15 at zero CoT**, `g-mean²`=0.85,
and the CoT is *not* inert — GLM sandbags *less* when it reasons (+0.07 at high), so the trace the
monitor reads is causally active, not a decoy. (That is why we say **fragile**, not "illusory";
the per-cell drop of +0.08 is itself within noise, §8.5.) The joint "significant `effect@lowest`
under a well-estimated high `g-mean²`" holds cleanly in **two cells** (`wmdp`·GLM, `gpqa`·oss);
`wmdp`·oss and `gpqa`·GLM each contribute one solid leg.

**Figure** — `results/figures/fig1_drop.png` (the single hero figure):
*"No cell shows the perturbation effect growing with reasoning — the 'robust' region (drop < 0) is
empty."* `drop` per cell with 95% CIs, one row each; vertical line at 0; the shaded robust region
(drop < 0) holds **no point estimate**; clean cells above the divider, truncation-confounded
`daft` faint below; filled = both legs solid, hollow = one under-powered leg. A callout on the
sharpest cell (`wmdp`·GLM) ties the result to monitorability (`g-mean²`=0.85 — monitorable, yet
the perturbation needs no CoT). The caption is explicit that we claim the robust zone empty, not
that drops are individually positive, and that `daft`'s apparent significance is a confound
artifact. *(The g-mean² values and CIs are in the §6 table; we cut a second scatter figure as it
carried the near-tautological "monitorable where CoT not needed" claim with two under-powered
points — the table and this callout cover it more cleanly.)*

---

## 7. How to read `effect` and `drop`

- **High `effect(e)`** → the perturbation moves behavior at effort `e`. High at `lowest` → works
  with little/no CoT; high at `high` → present under full reasoning (and the precondition for
  `g-mean²` to be defined).
- **High (positive) `drop`** → the perturbation's effect is weaker at high effort → the response
  does not require the CoT → a high `g-mean²` there is **fragile**. (Not a claim that the CoT is
  causally inert, §4.)
- **`drop ≈ 0`** → effect unchanged by reasoning → still non-CoT-dependent.
- **`drop < 0`** → effect *grows* with reasoning → the response **needs** the CoT → high
  `g-mean²` would be **robust**. We did not observe this, but our `drop` estimates are noisy and,
  for `daft`, confounded.

| `drop` | response requires CoT? | a high `g-mean²` there is… |
|---|---|---|
| large + | no | **fragile** |
| ≈ 0 | no | **fragile** |
| negative | yes | **robust** |

---

## 8. Caveats & limitations

1. **Pooled proxy, not OpenAI's gated metric.** Their per-instance estimator is unrunnable at
   `k = 4` (eligibility gate never satisfied → `nan`). Our headline `g-mean²` drops that gate and
   pools at the eval level; values are within-study and **not comparable** to OpenAI's reported
   numbers. (§2)
2. **Between-effort, not within-rollout.** We measure CoT-dependence by comparing across effort
   levels, not by ablating the CoT on the scored rollouts. We claim *non-requirement*, not
   *inertness*. **The decisive follow-up is cheap and concrete:** take the scored high-effort
   rollouts, truncate/remove the reasoning trace, force the answer (prefill the post-CoT channel),
   and re-grade `y` — roughly one extra high-effort cell's worth of calls (a few hundred, ~$2–3).
   We deprioritized it because the effort sweep already establishes non-requirement (which is what
   the fragility claim needs), but it is the first experiment we'd run next: it would directly test
   whether the *specific* scored CoT was causally necessary, and — run on `daft` with a larger
   token budget — it is also the decisive **de-confounding** test for the decoupling hypothesis
   (§6, §8.3). (§4)
3. **`daft` is truncation-confounded** — its high-effort arm grades only the easy items that
   finish under the 32k-token cap, so its effort comparison mixes effect-change with
   population-change. This also **manufactures** `daft`'s apparently-significant `drop`: dropping
   the hard items lowers `effect@high`, widening the gap. So `daft`'s drop CI excluding 0 is an
   artifact, not evidence — `daft` is excluded from the headline and is the cell **most worth
   de-confounding** (§8.2). (§5/§6)
4. **The clean joint result rests on two cells** — `wmdp`·GLM and `gpqa`·oss (significant
   `effect@lowest` *and* well-estimated high `g-mean²`). `wmdp`·oss (g-mean² solid, effect
   marginal) and `gpqa`·GLM (effect solid, g-mean² CI [0.06, 0.97]) each support it with one
   under-powered leg; `daft` is confounded/undefined. So this is a small, illustrative study, not
   a powered benchmark.
5. **`drop` estimates are noisy** (per-cell drops are within sampling error); the solid quantity
   is `effect@lowest > 0`, not the magnitude of the decline.
6. **Weak single monitor** (`gpt-4.1-mini`), one prompt, ~8% unparsed; modest effects (~0.1)
   limit resolution.
7. **Effort nuance.** GLM `low` still emits substantial CoT (soft floor); `none` is the clean
   zero-CoT anchor; oss has no zero-CoT mode.

What survives all of this: **(1) across every cell, the perturbation effect is already present at
minimal reasoning and never significantly grows with it — the "robust" region (drop < 0) is
empty; (2) in the two cleanest cells a significantly positive `effect@lowest` sits under a high
`g-mean²`, so that monitorability is fragile to CoT suppression — a property the metric does not
capture; (3) the sprint's sharpest claim — that this holds even where the CoT is load-bearing for
the *task* (task-LB ≠ perturbation-LB) — is gestured at by `daft` but is a hypothesis awaiting the
de-confounding rerun (§8.2), not a result.**

---

## 9. Reproducibility

```
exp/common.py          providers, prompts, grading, monitor parsing
exp/run_rollouts.py    rollout harness (resumable JSONL)        -> results/rollouts.jsonl
exp/run_monitor.py     monitor pass (z)                         -> results/monitor.jsonl
exp/analyze_pooled.py  effects, task/perturbation load-bearing, pooled g-mean²  -> results/report_pooled.txt
exp/plots.py           figures                                  -> results/figures/
data/intervention/*.jsonl   vendored, subsampled intervention data
data/monitor_templates/*    vendored OpenAI monitor prompts
data/metric/intervention_gmean_metric.py   OpenAI's per-instance metric (vendored for reference; see §2 — unrunnable at our k, not used for headline numbers)
```
Keys are read from a gitignored `.env`. Total spend: **$23.47**.

---

## 10. Reused vs. newly written

**Reused from `cot-monitor-stress-test` (own prior research, adapted):** CoT-extraction pattern,
MCQA answer-parsing, async monitor-scoring loop, matplotlib plotting conventions, the
gpt-oss/GLM OpenRouter + `reasoning_effort` usage pattern.

**Reused from `monitorability-evals` (OpenAI, vendored subset):** the 3 intervention JSONLs
(subsampled), the `toy_math_science` + `sandbagging` monitor templates, and
`intervention_gmean_metric.py` (vendored **for reference**; unrunnable at our sample size and
**not used for the headline numbers** — see §2/§8.1).

**Newly written for this sprint:** the dual-model raw-CoT harness with the `none/low/high`
effort sweep; the load-bearing measurements (`effect@lowest`, `drop`, task vs perturbation
load-bearing); the pooled `g-mean²` re-implementation with instance bootstrap; the figures; and
this analysis.
