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
