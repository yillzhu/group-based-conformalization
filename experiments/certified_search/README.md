# ML proposals with pointwise simulation certification

This is a deliberately **non-RCC** implementation of the sequential algorithm discussed in the research conversation. A small gradient-boosted tree estimates the conditional 95th percentile of the scalar constraint response \(g(x,Y)\), with violation defined by \(g(x,Y)>0\). It proposes the cheapest point on a finite search mesh whose fitted quantile is nonpositive. If none exists, it proposes the point with the smallest fitted quantile. After each audit, the observations can improve later proposals. Neither the fit nor the mesh is part of the risk guarantee.

At round \(t\), the point \(x_t\) is chosen *before* drawing a new batch of \(m\) iid responses at that point. With \(K_t\) observed violations, its one-sided Clopper–Pearson risk bound is

\[
 U_t = \begin{cases}
  \operatorname{Beta}^{-1}(1-\delta/T;K_t+1,m-K_t), & K_t<m,\\
  1, & K_t=m.
 \end{cases}
\]

Only points with \(U_t\leq\alpha\) are accepted. The returned point is the least-cost accepted point; if none is accepted, the procedure returns no decision. Conditional on all prior data, the current batch is fresh Bernoulli data at the fixed \(x_t\). Thus the union bound gives probability at least \(1-\delta\) that **every accepted point has true risk at most \(\alpha\)**, regardless of how the search model adapts to previous audits. This is a confidence statement over the simulator draws, not a deterministic guarantee and not a claim of global optimality. It depends on a faithful iid simulator at each chosen decision. In particular, applying this procedure to a fixed observational dataset without the ability to sample at chosen decisions would not have this guarantee.

The examples are ordered by search difficulty:

1. `picture`: one-dimensional heteroscedastic response from the four-quarter coverage picture; minimize \(x\) with \(P(Y<0.35\mid x)\leq 0.1\).
2. `capacity`: the continuous capacity example with a quiet interior interval; minimize capacity \(x\) with \(P(1-x+\sigma(x)Z>0)\leq 0.1\).
3. `queue`: a two-dimensional continuous-FTE call-center approximation; minimize staffing cost under at most 10% overload probability.

The first two examples have analytic risk and a mesh oracle for evaluation only. Queue risks and its continuous benchmark are estimated on separate simulation draws. No RCC procedure is called. The search mesh is only a proposal device: an audited decision is a fixed continuous-domain point, and its certificate does not depend on the mesh resolution.

Run from the repository root with:

```bash
.venv/bin/python experiments/certified_search/run.py --case all --seeds 10 --output experiments/certified_search/results/ten-seed-study.json
.venv/bin/python tests/test_certified_search.py
```

Defaults: \(\alpha=0.1\), \(\delta=0.05\), ten rounds, 600 fresh audit scenarios per round, 2,000 initial observations over the full domain. A tenth-percentile chance boundary often will *not* pass a finite-sample 95% audit; the proposal quantile is set to 95% to create some risk slack. This is a heuristic efficiency choice, not a validity requirement. The queue example may still fail to return a point in some runs; that is an honest outcome of a finite budget, not evidence that no feasible point exists.

## Initial numerical check

Ten seeds (0–9) with the settings above gave:

| Example | Certified runs | Selected cost range | Reference cost |
| --- | ---: | ---: | ---: |
| Coverage picture | 10/10 | 0.140–0.151 | 0.136 (exact mesh oracle) |
| Interior-quiet capacity | 10/10 | 1.088–1.098 | 1.080 (exact mesh oracle) |
| Continuous queue | 6/10 | 17.34–17.86 | about 17.05 (simulation oracle) |

The four queue failures had final proposals too near or above the risk boundary; none was returned. As an exploratory sensitivity check, changing only `--proposal-quantile .99` produced certificates in 10/10 queue runs, with selected costs 17.92–18.92. That is a meaningful cost increase for higher certification reliability. These are small Monte Carlo studies, not a performance claim or a comparison to RCC. Full per-round decisions, observed violations, confidence bounds, and independent risk evaluations are saved in `results/ten-seed-study.json` and `results/queue-q99.json`.

## Exploiting the staffing response structure

The earlier staffing studies used the known functional form

\[
 g(d,f,\xi)=R(d,\xi)-f,
 \qquad R(d,\xi)=\frac{\sum_{i=1}^2\max\{r_i(\xi)-d,0\}}{\eta}.
\]

Our initial boosted tree ignored this exact identity and fitted a conditional quantile in both \(d\) and \(f\). This wastes data estimating a coefficient already known to be \(-1\), and its stepwise approximation in \(f\) can point the optimizer at spuriously cheap staffing plans. The `--queue-structured` option instead fits a one-dimensional boosted quantile model to observed requirements \(R=y+f\), then predicts the response quantile as \(\widehat q_R(d)-f\). The model is still only a proposal heuristic; each point receives the same independent binomial audit. This option is appropriate only when the exact additive capacity shift is known.

On exploratory seeds 0–9, the structured 95th-percentile model certified 10/10 runs at mean cost 17.714. It certified 29/30 on the earlier holdout range 10–39, at mean selected cost 17.804 and independently evaluated mean risk 0.0705. The failure on seed 12 had several apparently low-risk proposals whose finite batches nevertheless failed the prespecified confidence cutoff.

On the untouched confirmation range 40–139, using 2,000 initial observations, ten rounds of 600 fresh simulations, and the same 95th-percentile target for both models:

| Proposal model | Certified runs | Mean cost among certified | Mean independent risk among certified |
| --- | ---: | ---: | ---: |
| Raw two-input boosted tree | 27/100 | 17.747 | 0.0745 |
| Structured one-input boosted tree | 99/100 | 17.839 | 0.0694 |

The simulation-oracle mean cost was 17.059. The costs in the table condition on certification, so their direct comparison is affected by selection: on the 27 seeds where **both** methods certified a plan, the structured method cost only 0.064 more on average. The 72-point increase in certification frequency is the main gain. The prior buffered RCC study reported 100/100 reliable selections at mean cost 18.444 and mean risk 0.0672 on seeds 40–139, but it used a different training/calibration design and did not provide this pointwise optimizer-valid certificate; it is contextual evidence, not a controlled head-to-head comparison.

Reproduce the paired confirmation with:

```bash
.venv/bin/python experiments/certified_search/run.py --case queue --seed-start 40 --seeds 100 --output experiments/certified_search/results/queue-raw-seeds40-139.json
.venv/bin/python experiments/certified_search/run.py --case queue --seed-start 40 --seeds 100 --queue-structured --output experiments/certified_search/results/queue-structured-seeds40-139.json
```

The new saved JSON files also contain all proposals, audit counts and upper bounds, not just the selected points. The 95% guarantee refers to the probability, over fresh simulator draws, that any accepted point is unsafe; the observed 99/100 selection frequency is an empirical measure of search efficiency and is not itself part of the theorem.

## Structure-free neural proposals

The strong structured-tree result prompted a direct question: can a proposal model learn the staffing response surface **without being told** the exact capacity shift? The new `--proposal-model neural` option fits a two-hidden-layer, tanh conditional-quantile network to the raw two-dimensional inputs \((d,f)\) and response \(g\). It starts with 2,000 full-domain observations and warm-starts its fit after each audit using the audited response data. The network neither removes the flexible-staff coordinate nor uses \(R=y+f\). The same candidate mesh, ten rounds, 600 fresh audit scenarios per round, and Clopper–Pearson certificates are retained. PyTorch is needed only for this optional proposal model.

We also tried an explicit `--audit-guidance` rule: after a failed audit, transfer half of the observed model-versus-empirical quantile discrepancy into a global margin for later proposals. This was counterproductive. At the 95th-percentile target, it certified 30/30 on seeds 10–39 but raised mean cost from 18.305 to 18.865. Warm-starting the network on past audit observations already provides useful adaptation; the extra margin is **off by default**.

The initial 95th-percentile neural setting certified 10/10 exploratory seeds 0–9 and 30/30 seeds 10–39, but its mean cost was 18.305 on the latter range. We then tried only 94th and 93rd percentiles on seeds 0–39. Both certified every one of those 40 runs; the 93rd percentile was cheaper (mean cost 17.894 versus 18.120 on seeds 10–39). We froze the 93rd-percentile specification before evaluating its results on seeds 40–139. The 95th-percentile result on that range had already been inspected, so this is a confirmation of the *new* 93rd-percentile specification, not a claim that the entire project never looked at those seeds.

| Proposal model, seeds 40–139 | Certified runs | Mean cost among certified | Mean independent risk among certified |
| --- | ---: | ---: | ---: |
| Raw two-input boosted tree, 95th percentile | 27/100 | 17.747 | 0.0745 |
| Structure-aware boosted tree, 95th percentile | 99/100 | 17.839 | 0.0694 |
| Raw two-input neural net, 95th percentile | 100/100 | 18.328 | 0.0595 |
| Raw two-input neural net, 93rd percentile | 99/100 | 17.917 | 0.0720 |

The simulated oracle mean cost is 17.059. On the 99 seeds where both the structured tree and 93rd-percentile neural method certified a plan, the neural cost was only 0.078 higher on average. The 93rd-percentile neural method and structured tree both failed to certify on seed 137; neither returned an uncertified plan. These results show that a sufficiently smooth, flexible proposal model can nearly reproduce the benefit of the known structural feature in this example. They do not show that neural models dominate structured modeling generally, nor that the 93rd-percentile setting transfers to other simulators.

Reproduce the neural confirmation with:

```bash
.venv/bin/python experiments/certified_search/run.py --case queue --seed-start 40 --seeds 100 --proposal-model neural --output experiments/certified_search/results/queue-neural-plain-seeds40-139.json
.venv/bin/python experiments/certified_search/run.py --case queue --seed-start 40 --seeds 100 --proposal-model neural --proposal-quantile .93 --output experiments/certified_search/results/queue-neural-q93-seeds40-139.json
```

## Tests beyond staffing

`run_other_examples.py` applies the **same unstructured neural proposal model and independent pointwise audit**, at the prespecified 95th-percentile proposal target, to two other simulator-backed families. A raw boosted tree is run on the same seeds as a comparator. Each search starts with 2,000 full-domain observations and has ten audits of 600 fresh outcomes; the risk target is 10% and the confidence error budget is 5% per quadratic instance or complete tuition policy. No RCC method is called.

These are carefully scoped variants, not claims about the original chapter's data:

- **Quadratics.** The original Section 4.5 constraint is deterministic, observed with noise, so its true-feasibility check is not a chance constraint. Here we retain the original random coefficient *distributions* for objective and constraint, but define a stochastic response \(g(x,\xi)=\bar g(x)+\sigma Z\), where \(Z\) is standard normal and \(\sigma=0.2\operatorname{sd}_{X\sim U([-1,1]^d)}\bar g(X)\). The operational target is \(P(g(x,\xi)>0)\leq 0.1\), whose exact risk is \(\Phi(\bar g(x)/\sigma)\). Search and oracle both use a documented mesh of 801 points in 1-D, \(81^2\) in 2-D, or \(31^3\) in 3-D; no continuous global-optimality claim is made. Random instances with no chance-feasible mesh point are reported separately.
- **Tuition.** The historical JANOS observations cannot supply new cohorts at an offer selected by the optimizer. Instead we use the repository's published synthetic tuition oracle: for each of 12 fixed applicant segments, a fresh cohort enrollment count is binomial at the selected merit offer. The operational event is enrollment rate below \(\theta=0.4\). This is the synthetic oracle benchmark's predictive target, not the historical experiment's \(\theta=0.17\). Merit costs are in thousands of dollars. The 5% confidence budget is divided equally across the 12 segments, so the entire returned tuition policy—not merely each segment in isolation—has a simultaneous 95% pointwise risk certificate if all segments certify.

Fresh seed ranges, using the same initial training draws and audit RNG streams for the two proposal models, gave:

| Stochastic quadratic, seeds 10–39 | Chance-feasible instances | Certified, neural | Certified, tree | Mean objective gap on active certified instances, neural / tree |
| --- | ---: | ---: | ---: | ---: |
| One-dimensional | 22/30 | 21/22 | 20/22 | 0.063 / 0.127 |
| Two-dimensional | 24/30 | 24/24 | 24/24 | 0.189 / 0.974 |
| Three-dimensional | 29/30 | 29/29 | 27/29 | 0.245 / 3.991 |

The last column restricts to instances where the chance constraint changes the mesh optimum (11 feasible instances in 1-D, 16 in 2-D, and 18 in 3-D). No certified point in these runs exceeded 10% **exact** risk. The gap values are in each random quadratic objective's units, not a common monetary scale. The 1-D and 3-D comparisons condition on different certified subsets; the 2-D comparison uses all 16 active instances.

For synthetic tuition on seeds 5–14:

| Proposal model | Certified segment decisions | Complete 12-segment policies | Mean merit gap among certified segments |
| --- | ---: | ---: | ---: |
| Neural 95th percentile | 120/120 | 10/10 | \$444 |
| Boosted-tree 95th percentile | 115/120 | 7/10 | \$790 |

On the 115 segments certified by both, the tree's offer was \$351 higher on average. The neural policies' cohort-weighted mean merit was \$8,569, versus \$8,243 for the exact predictive-risk mesh oracle. All 120 neural selections had exact risks below 10% in this study. These results support the usefulness of smooth structure-free proposals in these *simulatable* variants; they do not establish an advantage on the original observational tuition dataset or the deterministic quadratic benchmark.

Reproduce with:

```bash
for case in quadratic1 quadratic2 quadratic3; do
  for model in neural tree; do
    .venv/bin/python experiments/certified_search/run_other_examples.py \
      --case "$case" --proposal-model "$model" \
      --seed-start 10 --seeds 30
  done
done
for model in neural tree; do
  .venv/bin/python experiments/certified_search/run_other_examples.py \
    --case tuition --proposal-model "$model" --seed-start 5 --seeds 10
done
```

The saved JSON in `results/` contains every audit count, upper confidence bound, selected point, and exact-risk evaluation. The independent simulation assumption remains indispensable: without the ability to generate new outcomes at a chosen decision, the pointwise certificate does not follow.
