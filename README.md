[![make.fun gta6-twitch-ccu-forecast-01](https://make.fun/badge/gta6-twitch-ccu-forecast-01.svg)](https://make.fun/bounty/gta6-twitch-ccu-forecast-01)
[![rank](https://make.fun/badge/gta6-twitch-ccu-forecast-01/Zhoya0077/gta6-ccu-forecast/rank.svg)](https://make.fun/bounty/gta6-twitch-ccu-forecast-01)
[![reward](https://make.fun/badge/gta6-twitch-ccu-forecast-01/Zhoya0077/gta6-ccu-forecast/reward.svg)](https://make.fun/bounty/gta6-twitch-ccu-forecast-01)

# Twitch CCU forecaster — make.fun `gta6-twitch-ccu-forecast-01`

Forecasts how many people watch a game's Twitch category live in each hour of
its release day, hours 2 to 24 (UTC), given everything recorded up to the end
of hour 1. Entry point is `main.py` at the repo root: Python 3.13, standard
library only, no network, deterministic.

## How it runs

One case per invocation. The program reads every `*.in` file in the input
directory (in name order) and writes one `<id>.out` beside it:

```sh
mkdir -p my_input my_output
cp path/to/kit/input/p0*.in my_input/
MAKEFUN_INPUT_DIR=my_input MAKEFUN_OUTPUT_DIR=my_output python3 main.py
```

Directories come from `MAKEFUN_INPUT_DIR` / `MAKEFUN_OUTPUT_DIR`, defaulting to
`/input` and `/output`. A missing output directory is created. One bad case
cannot take the others down: each is wrapped, a failure is reported on stderr,
and the rest are still written.

Output is one object, at most 65,536 bytes:

```json
{"hours": [{"hour": 2, "q": [q05, q25, q50, q75, q95]}, ... 23 rows ...]}
```

Quantiles are finite, strictly positive, at most 1e9, non-decreasing within a
row, rounded to 2 decimals.

## The model

**1. Hour-1 print `p`.** From `history_hourly`, take the whole-hour bucket whose
`[from, from + 1h)` window contains the checkpoint end (the last complete hour
at or before the checkpoint). Fall back to the mean of `history` points inside
hour 1, then to the last usable history value. `p` is floored at 1.

**2. Comparable log-ratio curves.** For each comparable, bucket its `hourly`
points onto release-day hours 1..24 (`hour = floor((from - day_start)/1h) + 1`),
then form `lr[h] = ln(mean_h) - ln(mean_1)` for `h = 2..24`. Hours with no point
are skipped, never interpolated.

Two guards drop a comparable:

* fewer than 18 usable hours — too sparse to describe a launch day;
* `max_h lr[h] > ln(20)` — the "first hour" was a pre-unlock trickle, so the
  ratios describe a different phenomenon than a launch hour. Without this, a
  launch that sat at 164 viewers and peaked at 348k (2120x) contributes a
  `lr[24]` of +6.99 and drags the whole median upward.

**3. Unlock alignment.** Some `unlock` strings carry an absolute timestamp (a
single global unlock at 14:00Z, say). When one falls inside `[-6h, +24h]` of the
comparable's `day_start`, its curve is re-indexed so hour 1 means the first hour
at or after that unlock (shift capped to 1..12 h, otherwise ignored). Black
Myth: Wukong unlocked at 02:00Z, so its hour-1 print is 4,941 viewers of
pre-unlock waiting; aligned, its launch hour reads 129,864. Aligning makes a
global unlock comparable with a rolling-midnight launch instead of comparing a
launch hour against a quiet morning.

**4. Aggregate per hour** across the surviving comparables:

* centre `c[h] = median(lr[h])`
* spread `s[h] = clamp(1.4826 * MAD(lr[h]), 0.10, 1.2)`

With fewer than 3 comparables contributing, `c[h] = 0` and
`s[h] = 0.15 * sqrt(h - 1)` — the house baseline's shape.

**5. Forecast.** With `z = [-1.6449, -0.6745, 0, 0.6745, 1.6449]`:

```
q_tau(h) = p * exp(c[h] + z_tau * s[h] * 0.95)
```

then clamp to `(0, 1e9]`, round to 2 decimals, and enforce non-decreasing order
within the row. The `0.95` calibrates the spread: the MAD of a small sample of
curves slightly overstates the spread actually needed, and 0.95 buys the same
loss with coverage closer to the honest 0.90.

### The one refinement: a size slope

The median ratio curve is not the same for a launch that opens huge as for one
that opens small. Pooling every usable comparable curve across the four
practice cases and regressing `lr[h]` on `log10(p1)` shows a clear size effect
that grows through the day: roughly `+0.08` per decade at hour 2, `-1.2` at
hour 12, `-2.2` at hour 24. A launch that opens at 875k has spent its audience
in hour 1 and gives most of it back over the day; a launch that opens at 20k is
still climbing toward its evening peak. Flat ratios therefore over-predict big
launches late in the day and under-predict small ones.

So, per hour, fit

```
lr[h] ~ alpha[h] + beta[h] * log10(p1)
```

by least squares on this case's comparable set, with one robust pass that drops
points beyond 3 robust sigmas so a single freak launch cannot dictate the
slope, then apply

```
c'[h] = c[h] + 0.50 * beta[h] * clip(log10(p) - median(log10(p1)), -2.5, +2.5)
```

The `0.50` shrinks the correction. The clip stops a case far outside the
comparable range from extrapolating without limit; slopes are clipped to
`[-4, 4]` and a fit needs at least 8 points, else that hour's slope is 0 and the
model reduces to plain comparable ratios.

Everything is deterministic: total sort orders, exact medians, no randomness,
no hash-order dependence (verified identical output under differing
`PYTHONHASHSEED`).

## Validation

The practice case files do not contain their own answers, so the honest
backtest is leave-one-out over the real launch curves embedded in them: take
each comparable curve in turn as the "case", rebuild the forecast from the
others, and score pinball loss on the log scale against its actual hours 2..24,
exactly as the server scores. 152 raw comparable entries dedupe to 37 usable
curves; the spike filter keeps 30, of which 28 open above 20k viewers — the
regime the final lives in.

| test | model | house baseline | 90% coverage |
|---|---|---|---|
| leave-one-out, all 30 curves | **0.2715** | 0.3515 | 0.889 |
| leave-one-out, 28 curves opening >20k | **0.2568** | 0.3142 | 0.909 |
| extrapolation: train ≤100k, score >100k (17 curves) | **0.2429** | — | 0.878 |
| extrapolation: train ≤200k, score >200k (7 curves) | **0.1849** | — | 0.949 |

The extrapolation rows matter most, since the final is a launch above most
comparables: the size slope is what keeps that honest, and it still wins when it
has to extrapolate rather than interpolate.

Ablation, leave-one-out over the 28 curves opening above 20k:

| variant | loss | coverage |
|---|---|---|
| comparable medians only | 0.2809 | 0.889 |
| + spike filter | 0.2795 | 0.896 |
| + size slope, shrink 0.50 (**shipped**) | **0.2568** | 0.909 |
| size slope, no shrink (1.0) | 0.2550 | 0.912 |
| size slope, no spike filter | 0.2739 | 0.937 |

Shrink 0.50 is chosen on the extrapolation rows rather than the in-sample ones:
at 100k the sweep gives 0.2419 (0.50) versus 0.2979 (1.00) and 0.3389 (0.00),
and 0.50 is also the best point at 200k. In-sample the curve is flat from 0.5 to
1.0, so the extrapolation result is what decides it.

The shipped configuration also passes the output contract on all four practice
cases: 23 rows, hours 2..24 each once, five quantiles per row, non-decreasing,
all finite in `(0, 1e9]`, valid JSON, 1.4–1.6 KB, two runs byte-identical, and
0.38 s for the slowest run against an 8 s limit. Degenerate inputs all produce
valid output: no comparables, unusable comparables, empty history, zero hour-1,
5e8 hour-1, a minimal case with only the required keys, null and non-numeric
viewer points, and nine cases in one directory at once.

## Layout

```
main.py     the whole forecaster: time utils, robust stats, curve building,
            aggregation, the size slope, and the file loop
README.md   this file
```
