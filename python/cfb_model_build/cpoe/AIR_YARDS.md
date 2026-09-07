# The air-yards CP model (Approach B, revived)

_Measured 2026-09-07. Supersedes the "Approach B INFEASIBLE" verdict in
[FEASIBILITY.md](FEASIBILITY.md), which was about a different data source._

## Why the old verdict no longer applies

FEASIBILITY.md checked **CFBD** for air yards and found the field absent from
the `/plays` response entirely — 0% fill across every name it tried
(`air_yards`, `yards_to_sticks`, `pass_length`, `passLength`). That finding is
still correct about CFBD.

It says nothing about ESPN. sdv-py's CFB parser now *derives* air yards from the
play text — `air_yards = start.yardsToEndzone - air_yardsToEndzone`, where the
catch/target spot comes from a `caught at …` / `thrown to …` regex. The blocker
was never "air yards are unknowable in college football"; it was "the API we
asked didn't carry them". A different source answers differently.

## What the game-state model is actually worth

The shipping CP model uses eight game-state features and no notion of how far
the ball travelled. On the plays where air yards *are* available, it performs
close to a coin flip:

| Arm | log-loss | AUC |
|---|---:|---:|
| intercept-only (base rate 0.627) | 0.6603 | 0.500 |
| **A — game-state only (ships today)** | **0.6539** | **0.567** |
| **B — game-state + air yards** | **0.5398** | **0.764** |
| C — PR #27's shape (4 game-state features dropped) | 0.5419 | 0.762 |

_5-fold CV grouped by `game_id`, 27,804 pass plays with a known outcome,
2025–2026, measured against the **published `sportsdataverse-data` release
parquet**. Reproduce: `ClaudeCowork/notes/2026-09-07-cfb-xcp-air-yards/`._

The fitting script itself trains on the **`cfbfastR-cfb-raw` finals corpus**
and reports 0.5395 / Brier 0.1806 over 27,673 plays — a slightly different
population because it is a different source, not a different result. The
registry quotes the fitting script's numbers; this file quotes the independent
confirmation. Both are recorded so neither looks like the other's restatement.

Arm A beats the intercept by 0.0064 of log-loss. That is the entire skill of the
currently-published CFB CP model, and by extension of the CPOE derived from it.
Throw depth is not a refinement here — it is most of the signal, and the model
shipped without it.

## Is that jump a leak?

An AUC of 0.567 → 0.764 from one feature is also what leakage looks like, so it
was checked rather than assumed. Air yards would be leaky if the field only
existed for completed passes — the model would then be reading the outcome.

It isn't. The regex matches `thrown to` (incompletions) as well as `caught at`
(completions), and coverage is only mildly outcome-dependent:

| | coverage | n |
|---|---:|---:|
| completions | 45.4% | 38,385 |
| incompletions | 37.6% | 27,541 |

A completion-only field would show ~100% vs ~0%. The mild skew does mean the
covered subset is slightly completion-rich (0.627 vs 0.582 overall), which is
why the air-yards model is only ever *applied* to covered rows — never
extrapolated onto plays whose air yards are unknown.

## Why two models instead of one

PR #27 proposed replacing the feature set and moving `MIN_SEASON` 2004 → 2025.
The features were the right instinct; the replacement was not. Coverage is the
reason:

| season | pass plays | with air yards |
|---|---:|---:|
| pre-2025 | ~1.3M | ~0% |
| 2025 | 61,660 | 38.9% |
| 2026 (to date) | 4,266 | 90.2% |

A single air-yards model would leave **CPOE undefined for 2004–2024 and for 61%
of 2025** — a regression for game-on-paper, which renders historical games. A
single game-state model wastes the best available signal on every modern play.

So both ship, and `compute_cpoe_hybrid()` routes each play to the best model
that can see it:

- `air_yards` present → **`cfb_cp_model_air_yards.ubj`** (2025+, 11 features)
- otherwise → **`cfb_cp_model.ubj`** (2004+, 8 features, unchanged)

Arm B is preferred over arm C because it keeps the four game-state features PR
#27 dropped. They cost 0.002 of log-loss to keep and nothing in data (identical
rows), and they still carry real information air yards can't — a throw on 3rd
and 18 down two scores is a different proposition from the same throw tied in
the first quarter.

## The one thing consumers must not do

`cp_model` on the scored output is load-bearing. CPOE from the two models is
**not on a comparable scale** — one is residual against a near-coin-flip
baseline, the other against a genuinely predictive one. A season leaderboard
that averages CPOE across both is mixing two different quantities, and because
coverage rose sharply from 2025 to 2026, that mix changes composition over time.
Group by `cp_model`, or restrict to one.

## Cross-validation

The air-yards model exists for 2025+ only, so LOSO would be two folds with one
partial season. It is gated on `GroupKFold(5)` by `game_id` instead: passes in
one game share a QB, an offense, an opponent and the weather, so a row-level
split would put near-duplicate plays on both sides and flatter the model. The
game-state model keeps LOSO, which its 22 seasons support.
