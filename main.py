#!/usr/bin/env python3
"""Twitch CCU forecaster for make.fun gta6-twitch-ccu-forecast-01.

Reads one case (<id>.in) per run from $MAKEFUN_INPUT_DIR (default /input) and
writes <id>.out into $MAKEFUN_OUTPUT_DIR (default /output): five quantiles of
each hour 2..24 of a launch day's Twitch category concurrent viewers.

Standard library only. Deterministic: the same case always produces the same
bytes. See README.md for the model and its justification.

Where each step of the model lives:

  hour-1 print        hour1_print()
  comparable curves   bucket_hours(), unlock_shift_hours(), comparable_series()
  aggregation         hour_stats()   (median ratio, MAD spread)
  the refinement      size_slopes()  (log-ratio on log10(p1), per hour)
  quantiles           forecast_rows() with q = p * exp(c + z*s)
  the file loop       main()
"""

from __future__ import annotations

import json
import math
import os
import sys
import tempfile
from datetime import datetime, timedelta, timezone

# Standard normal quantiles for the five levels the case asks for.
TAUS = (0.05, 0.25, 0.50, 0.75, 0.95)
Z = (-1.6448536269514722, -0.6744897501960817, 0.0, 0.6744897501960817, 1.6448536269514722)

HOURS = tuple(range(2, 25))

# --- model constants (see README: "The one refinement") -------------------
SPREAD_MULT = 0.95      # global calibration of the comparable dispersion
SPREAD_FLOOR = 0.10     # per-hour floor on the log spread
SPREAD_CAP = 1.2        # per-hour cap on the log spread
SLOPE_SHRINK = 0.50     # shrinkage on the size-slope correction
SIZE_CLIP = 2.5         # clip on log10(p) - median log10(p1) of comparables
BETA_CAP = 4.0          # clip on any fitted slope
MAX_SPIKE_LOG = math.log(20.0)  # drop curves whose peak exceeds 20x hour-1
MIN_Q = 0.01
MAX_Q = 1e9


# ---------------------------------------------------------------- time utils
def parse_iso(text):
    """Parse an ISO 8601 timestamp into an aware UTC datetime; None if unusable."""
    if not isinstance(text, str) or not text:
        return None
    s = text.strip()
    if s.endswith("Z") or s.endswith("z"):
        s = s[:-1] + "+00:00"
    try:
        dt = datetime.fromisoformat(s)
    except ValueError:
        return None
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)


def hour_offset(later, base):
    """Whole hours from base to later (float), or None."""
    if later is None or base is None:
        return None
    return (later - base).total_seconds() / 3600.0


# ------------------------------------------------------------- small robust
def median(values):
    """Median of a list of floats (mean of the middle pair when even)."""
    xs = sorted(values)
    n = len(xs)
    if n == 0:
        return 0.0
    mid = n // 2
    if n % 2:
        return xs[mid]
    return 0.5 * (xs[mid - 1] + xs[mid])


def mad(values):
    """Median absolute deviation, unscaled."""
    if not values:
        return 0.0
    m = median(values)
    return median([abs(v - m) for v in values])


def ols_line(pairs):
    """Least-squares slope and centre of [(x, y)]; (0, 0) when degenerate."""
    if len(pairs) < 2:
        return 0.0, 0.0, 0.0
    mx = median([p[0] for p in pairs])
    my = median([p[1] for p in pairs])
    num = sum((x - mx) * (y - my) for x, y in pairs)
    den = sum((x - mx) ** 2 for x, _ in pairs)
    return (num / den if den > 0 else 0.0), mx, my


# ------------------------------------------------------------ hour bucketing
def bucket_hours(rows, day_start, viewers_key="viewers", from_keys=("from", "at")):
    """Map hour index (1 = [day_start, day_start+1h)) -> mean of the points in it.

    A point at `from` covers [from, from + 1h); it lands in the bucket whose
    start is that same whole hour. Hours with no point are simply absent.
    """
    buckets = {}
    for row in rows or ():
        if not isinstance(row, dict):
            continue
        val = row.get(viewers_key)
        if not isinstance(val, (int, float)) or isinstance(val, bool):
            continue
        if not math.isfinite(val) or val < 0:
            continue
        stamp = None
        for key in from_keys:
            stamp = parse_iso(row.get(key))
            if stamp is not None:
                break
        if stamp is None:
            continue
        off = hour_offset(stamp, day_start)
        if off is None:
            continue
        idx = int(math.floor(off)) + 1
        if idx < 1 or idx > 24:
            continue
        buckets.setdefault(idx, []).append(float(val))
    return {h: sum(v) / len(v) for h, v in buckets.items()}


# --------------------------------------------------------------- hour-1 print
def hour1_print(case, day_start, checkpoint):
    """Viewers during the first hour: the last complete hour at the checkpoint.

    Order of preference:
      1. the history_hourly bucket that *ends* at the checkpoint,
      2. any history point inside [checkpoint - 1h, checkpoint),
      3. the last usable history value.
    """
    hourly = case.get("history_hourly") or []
    target = checkpoint - timedelta(hours=1) if checkpoint else day_start

    best = None
    for row in hourly:
        if not isinstance(row, dict):
            continue
        val = row.get("viewers")
        if not isinstance(val, (int, float)) or isinstance(val, bool):
            continue
        if not math.isfinite(val) or val < 0:
            continue
        start = parse_iso(row.get("from"))
        if start is None:
            continue
        end = start + timedelta(hours=1)
        if target is not None and start <= target < end:
            best = float(val)
            break
        if best is None and target is not None and start < target:
            best = float(val)

    if best is None:
        points = []
        lo = target
        for row in case.get("history") or []:
            if not isinstance(row, dict):
                continue
            val = row.get("viewers")
            if not isinstance(val, (int, float)) or isinstance(val, bool):
                continue
            if not math.isfinite(val) or val < 0:
                continue
            at = parse_iso(row.get("at"))
            if at is None or lo is None or checkpoint is None:
                continue
            if lo <= at < checkpoint:
                points.append(float(val))
        if points:
            best = sum(points) / len(points)

    if best is None:
        for row in reversed(case.get("history") or []):
            if not isinstance(row, dict):
                continue
            val = row.get("viewers")
            if isinstance(val, (int, float)) and not isinstance(val, bool):
                if math.isfinite(val) and val >= 0:
                    best = float(val)
                    break

    if best is None or not math.isfinite(best) or best < 1.0:
        return 1.0
    return best


# ------------------------------------------------- comparable curve handling
def unlock_shift_hours(comp, day_start):
    """Whole-hour shift of a comparable's release grid implied by its `unlock`.

    The unlock text carries absolute timestamps for some launches (a single
    global unlock at 14:00Z, say). When one sits within a day of the comparable's
    day_start we re-index its curve so "hour 1" means the first hour at or after
    the unlock. Text without a usable in-window timestamp yields 0 (no shift).
    """
    text = comp.get("unlock")
    if not isinstance(text, str) or not text:
        return 0
    found = []
    # Scan for ISO-8601 tokens directly. Slicing a fixed-width window and
    # handing it to the parser is unreliable: the token may be followed by
    # punctuation or may omit the seconds field.
    i = 0
    n = len(text)
    while i < n:
        if text[i] == "2" and text[i:i + 4].isdigit() and text[i + 4:i + 5] == "-":
            j = i + 5
            while j < n and (text[j].isdigit() or text[j] in "-:TZz+."):
                j += 1
            token = text[i:j].rstrip(".")
            stamp = parse_iso(token)
            if stamp is not None:
                off = hour_offset(stamp, day_start)
                if off is not None and -6.0 <= off <= 24.0:
                    found.append(off)
            i = max(j, i + 1)
        else:
            i += 1
    if not found:
        return 0
    shift = int(math.floor(min(found)))
    if shift < 1 or shift > 12:
        return 0
    return shift


def comparable_series(comp, align_unlock=True):
    """Log-ratio curve of one comparable, or None when it cannot be used."""
    day_start = parse_iso(comp.get("day_start"))
    if day_start is None:
        return None
    means = bucket_hours(comp.get("hourly"), day_start)
    if not means:
        return None

    if align_unlock:
        shift = unlock_shift_hours(comp, day_start)
        if shift:
            means = {h - shift: v for h, v in means.items() if 1 <= h - shift <= 24}

    base = means.get(1)
    if base is None or base <= 0:
        return None
    lr = {}
    log_base = math.log(base)
    for h in HOURS:
        v = means.get(h)
        if v is not None and v > 0:
            lr[h] = math.log(v) - log_base
    if len(lr) < 18:  # a curve too sparse to describe a launch day
        return None
    peak = max(lr.values())
    if peak > MAX_SPIKE_LOG:
        # The "first hour" was a pre-unlock trickle, not the launch hour, so
        # this curve's ratios describe a different phenomenon. Drop it.
        return None
    return {"p1": base, "lr": lr}


def collect_series(comparables, align_unlock=True):
    """All usable comparable curves, de-duplicated by (name, p1)."""
    out = []
    seen = set()
    for comp in comparables or ():
        if not isinstance(comp, dict):
            continue
        series = comparable_series(comp, align_unlock=align_unlock)
        if series is None:
            continue
        key = (str(comp.get("name") or ""), round(series["p1"], 6))
        if key in seen:
            continue
        seen.add(key)
        out.append(series)
    return out


# ----------------------------------------------------------------- aggregation
def hour_stats(series, h, floor=SPREAD_FLOOR, cap=SPREAD_CAP):
    """Median log-ratio and robust dispersion for hour h across comparables."""
    xs = [s["lr"][h] for s in series if h in s["lr"]]
    if len(xs) >= 3:
        spread = 1.4826 * mad(xs)
        if not math.isfinite(spread):
            spread = floor
        return median(xs), min(max(spread, floor), cap), len(xs)
    return 0.0, 0.15 * math.sqrt(h - 1), 0


def size_slopes(series, cap=BETA_CAP):
    """Per-hour slope of log-ratio on log10(hour-1 print), robustly fitted.

    One extra pass drops points whose residual exceeds 3 robust sigmas, which
    keeps a single freak launch from dictating the slope.
    """
    slopes = {}
    for h in HOURS:
        pairs = []
        for s in series:
            if h in s["lr"]:
                pairs.append((math.log10(max(s["p1"], 1.0)), s["lr"][h]))
        if len(pairs) < 8:
            slopes[h] = 0.0
            continue
        beta, mx, my = ols_line(pairs)
        resid = [y - (my + beta * (x - mx)) for x, y in pairs]
        sigma = max(1.4826 * mad(resid), 0.25)
        kept = [p for p, r in zip(pairs, resid) if abs(r) <= 3.0 * sigma]
        if len(kept) >= 8:
            beta, _, _ = ols_line(kept)
        if not math.isfinite(beta):
            beta = 0.0
        slopes[h] = min(max(beta, -cap), cap)
    return slopes


def forecast_rows(p1, series):
    """Quantile rows for hours 2..24 as [(hour, [q05, q25, q50, q75, q95])]."""
    if series:
        ref = median([math.log10(max(s["p1"], 1.0)) for s in series])
        delta = math.log10(max(p1, 1.0)) - ref
        delta = min(max(delta, -SIZE_CLIP), SIZE_CLIP)
        slopes = size_slopes(series)
    else:
        ref = delta = 0.0
        slopes = {h: 0.0 for h in HOURS}

    log_p = math.log(max(p1, MIN_Q))
    if not math.isfinite(log_p):
        log_p = 0.0
    rows = []
    for h in HOURS:
        centre, spread, _ = hour_stats(series, h) if series else (0.0, 0.15 * math.sqrt(h - 1), 0)
        if series:
            centre += SLOPE_SHRINK * slopes[h] * delta
        spread = max(spread * SPREAD_MULT, 1e-3)
        # A non-finite centre or spread would poison every level of this row;
        # fall back to the house baseline's shape rather than emit bad numbers.
        if not math.isfinite(centre) or not math.isfinite(spread):
            centre, spread = 0.0, 0.15 * math.sqrt(h - 1)

        # Rows must be non-decreasing across the five levels; hours are
        # independent of each other, so this guard resets for every hour.
        qs = []
        running = 0.0
        for z in Z:
            log_q = log_p + centre + z * spread
            log_q = min(max(log_q, math.log(MIN_Q)), math.log(MAX_Q))
            q = round(math.exp(log_q), 2)
            if q < MIN_Q:
                q = MIN_Q
            if q > MAX_Q:
                q = MAX_Q
            running = max(running, q)  # quantiles never decrease with tau
            qs.append(running)
        rows.append((h, qs))
    return rows


# ----------------------------------------------------------------------- I/O
def read_case(path):
    with open(path, "r", encoding="utf-8") as fh:
        return json.load(fh)


def build_payload(case):
    day_start = parse_iso(case.get("day_start"))
    checkpoint = parse_iso(case.get("checkpoint"))
    if checkpoint is None and day_start is not None:
        checkpoint = day_start + timedelta(hours=1)

    p1 = hour1_print(case, day_start, checkpoint)
    series = collect_series(case.get("comparables"))
    rows = forecast_rows(p1, series)

    return {"hours": [{"hour": h, "q": qs} for h, qs in rows]}


def write_json(path, payload):
    """Write compact, deterministic UTF-8 JSON atomically."""
    text = json.dumps(payload, ensure_ascii=False, allow_nan=False,
                      separators=(",", ":"), sort_keys=False)
    directory = os.path.dirname(os.path.abspath(path)) or "."
    os.makedirs(directory, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=directory, prefix=".tmp-", suffix=".out")
    os.fchmod(fd, 0o644)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            fh.write(text)
        os.replace(tmp, path)
    except BaseException:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise


def main(argv=None):
    in_dir = os.environ.get("MAKEFUN_INPUT_DIR") or "/input"
    out_dir = os.environ.get("MAKEFUN_OUTPUT_DIR") or "/output"

    try:
        names = sorted(n for n in os.listdir(in_dir) if n.endswith(".in"))
    except OSError as exc:
        print("cannot list %s: %s" % (in_dir, exc), file=sys.stderr)
        return 1

    os.makedirs(out_dir, exist_ok=True)
    failures = 0
    for name in names:
        src = os.path.join(in_dir, name)
        stem = name[:-3]
        dst = os.path.join(out_dir, stem + ".out")
        try:
            payload = build_payload(read_case(src))
            write_json(dst, payload)
        except Exception as exc:  # a bad case must not take the others down
            failures += 1
            print("case %s failed: %s" % (name, exc), file=sys.stderr)
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
