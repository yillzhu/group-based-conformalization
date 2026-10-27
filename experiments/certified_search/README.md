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
