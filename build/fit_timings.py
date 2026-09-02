#!/usr/bin/env python3
"""Scale a recitation deck's presenter-note timings to fit the 80-minute slot.

Each presenter note opens with "N min" (sometimes "N min," or "N min."). A deck
whose timings sum past 80 will run out of time, and the TA discovers that at
minute 70. This rescales them proportionally to a target total, preserving the
relative emphasis the deck was written with, with a floor of 2 minutes.

Slides whose timing must not move -- a workshop block, say -- are pinned by
passing their current values with --pin.

    python build/fit_timings.py recitation-03-*.md --target 78
    python build/fit_timings.py recitation-13-*.md --target 78 --pin 30
"""
import argparse
import re
import sys

PAT = re.compile(r"^(\d+) min", re.M)


def rescale(text, target, pin):
    vals = [int(m.group(1)) for m in PAT.finditer(text)]
    if not vals:
        sys.exit("no timings found")
    pinned = sum(v for v in vals if v in pin)
    movable = [v for v in vals if v not in pin]
    if not movable:
        return text, sum(vals), sum(vals)
    budget = target - pinned
    scale = budget / sum(movable)

    out, exact = [], []
    for v in vals:
        if v in pin:
            out.append(v)
            exact.append(float(v))
        else:
            e = v * scale
            out.append(max(2, round(e)))
            exact.append(e)

    # Nudge the largest movable slides until the total lands on target.
    order = sorted(
        (i for i, v in enumerate(vals) if v not in pin),
        key=lambda i: -exact[i],
    )
    while sum(out) > target:
        for i in order:
            if out[i] > 2:
                out[i] -= 1
                if sum(out) <= target:
                    break
        else:
            break
    while sum(out) < target:
        for i in order:
            out[i] += 1
            if sum(out) >= target:
                break

    it = iter(out)
    new = PAT.sub(lambda m: f"{next(it)} min", text)
    return new, sum(vals), sum(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("files", nargs="+")
    ap.add_argument("--target", type=int, default=78)
    ap.add_argument("--pin", type=int, nargs="*", default=[])
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()

    for path in a.files:
        text = open(path).read()
        new, before, after = rescale(text, a.target, set(a.pin))
        if not a.dry_run:
            open(path, "w").write(new)
        print(f"{path}: {before} -> {after} min")


if __name__ == "__main__":
    main()
