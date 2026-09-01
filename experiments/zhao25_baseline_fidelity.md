# Zhao--Kadota'25 MMSE-MaxWeight baseline fidelity contract

Primary source: Z. Zhao and I. Kadota, "Optimizing Age of Information
without Knowing the Age of Information," 2025,
<https://arxiv.org/abs/2501.06688>.

## Published priority and specialization

The published Max-Weight priority is

```text
beta_i p_i^S p_i^D [hhat_i(t+theta_i) - zhat_i(t) - theta_i].
```

FreshSky's simulator is generate-at-will, uses a single uplink, delivers a
successful update in the attempted slot, and delays only ACK/NACK feedback.
Consequently, the matched specialization sets `p_i^S=1`, `zhat_i=0`, and
`theta_i=0`. The paper's end-to-end success probability is represented by
the common causal posterior-predictive success probability `q_i(t)`.

The resulting executable priority is

```text
beta_i q_i(t) Ahat_i(t).
```

`Ahat_i(t)` is the common pending-success receiver-age estimate. Under an
i.i.d. success process this generate-at-will recursion is an MMSE estimate;
under the tested hidden GE channel, multiplying marginal pending-success
probabilities is a conditional-independence adaptation and is not described
as an exact MMSE estimator.

## Fixed theorem-beta calibration

For fixed source weights `alpha_i=1`, the implementation uses the source
paper's Theorem-7 randomized-rate calibration:

```text
vartheta_i = alpha_i / sbar_i
mu_i^R = min(1, sqrt(vartheta_i / vartheta_star))
sum_i mu_i^R = M
beta_i = alpha_i / (sbar_i mu_i^R),
```

where `sbar_i = pi_i phi_L + (1-pi_i) phi_N` is the stationary geometry
success probability. `beta_i` remains fixed within a run. The online score
uses `q_i(t)` in place of the source paper's fixed reliability and then
applies the same individual queue-cap shield as every shielded comparator.

Time-varying event value is intentionally not multiplied into this score:
doing so by setting `beta_i=value_i(t)` would produce the existing Ji'24-style
score `value_i(t) q_i(t) Ahat_i(t)` exactly and would double-count one
executable policy. The new row instead preserves Zhao--Kadota's fixed-weight
calibration and is evaluated, but not optimized, on FreshSky's dynamic W-AoI
metric.

## Guarantee boundary

The posterior-predictive GE adaptation and the added individual-energy shield
are outside Zhao--Kadota's original model. Therefore, no asymptotic bound from
that paper is transferred to this implementation. FreshSky's queue-cap
certificate applies because the common admission shield is executed.

## Executable checks

`test_zhao25_specialization.py` checks the single- and two-pending age
mixtures, the Ji-equivalence condition, the Theorem-7 water filling and beta
identity, nonduplication under heterogeneous reliabilities, and rejection of
an infeasible highest-priority source by the common shield.
