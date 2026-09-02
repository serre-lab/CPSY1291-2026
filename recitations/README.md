# CPSY 1291 — Recitation materials (Fall 2026)

One 80-minute TA-led recitation per week, optional, for all fourteen weeks. Each
week ships a **handout** (prose, self-contained, with exercises and answers) and
a **Marp deck** (what the TA puts on the screen, with a timed presenter note on
every slide).

The two cover the same ground on purpose. A student who reads the handout has
lost nothing by skipping the session; a student who attends does not need to take
notes. See `RECITATION-PROGRAM.md` for the full plan, the scheduling
constraints, and what is deliberately excluded.

## The fourteen weeks

| # | Week | Serves | Handout | Deck |
|---|---|---|---|---|
| 1 | 1 | A1 | `bootcamp-notes` | `recitation-01-python-numpy` |
| 2 | 2 | A1 | `bootcamp-notes` | `recitation-02-vectors-to-pca` |
| 3 | 3 | A1 | `handout-03-matrices-fitting-knobs` | `recitation-03-matrices-fitting-knobs` |
| 4 | 4 | A2 | `handout-04-gradients-chain-rule` | `recitation-04-gradients-chain-rule` |
| 5 | 5 | A2 | `handout-05-pytorch-training-loop` | `recitation-05-pytorch-training-loop` |
| 6 | 6 | A3 | `handout-06-heldout-data-and-images` | `recitation-06-heldout-data-and-images` |
| 7 | 7 | midterm | `handout-07-midterm-review` | `recitation-07-midterm-review` |
| 8 | 8 | A4 | `handout-08-rsa-decoding-permutation` | `recitation-08-rsa-decoding-permutation` |
| 9 | 9 | A4 | `handout-09-hooks-gradients-saliency` | `recitation-09-hooks-gradients-saliency` |
| 10 | 10 | A5 | `handout-10-sequences-recurrence-dynamics` | `recitation-10-sequences-recurrence-dynamics` |
| 11 | 11 | A5 | `handout-11-attention-transformers-generative` | `recitation-11-attention-transformers-generative` |
| 12 | 12 | final | `handout-12-final-review` | `recitation-12-final-review` |
| 13 | 13 | project | `handout-13-project-clinic-i` | `recitation-13-project-clinic-i` |
| 14 | 14 | project | `handout-14-project-clinic-ii` | `recitation-14-project-clinic-ii` |

Weeks 1 and 2 share one handout: `bootcamp-notes.pdf` covers both sessions, and
predates the per-week convention.

**Two sessions have hard dates.** R07 must be held before the midterm (Thu 10/22),
and R12 must be held before Thanksgiving recess, since the final exam is Tue 12/1
— the first day back.

## Building

Handouts (from this directory; the shared preamble is `cpsy1291-handout.sty`):

    latexmk -lualatex handout-06-heldout-data-and-images.tex
    for f in handout-*.tex; do latexmk -lualatex "$f"; done

Decks (from `2026/slides/`):

    node _render_deck.js ../bootcamp/recitation-06-heldout-data-and-images.md

## Checks

Both should pass before anything is handed to the TA.

    node build/check_overflow_bootcamp.js          # nothing cropped by the 720px box

    for f in recitation-*.md; do \
      echo "$f $(grep -oE '^[0-9]+ min' $f | grep -oE '[0-9]+' | paste -sd+ - | bc)"; \
    done                                           # each deck should total ~78 min

`build/fit_timings.py` rescales a deck's timings proportionally if it drifts:

    python build/fit_timings.py recitation-06-*.md --target 78
    python build/fit_timings.py recitation-13-*.md --target 78 --pin 30   # keep the workshop block

## Scope, and how it was chosen

**R01–R05 were scoped from the actual API surface of Assignments 1 and 2**,
extracted from the notebooks rather than from a topic list. Everything in them is
used in an assignment. Re-derive it with:

    python build/nb_api_surface.py ../assignments/A2/A2_part*.ipynb

**R06–R12 were scoped from the lecture decks L09–L19**, because A3, A4 and A5 did
not exist when they were written. They are therefore provisional: once each
assignment is drafted, re-extract its API surface and cut whatever it does not
use. Do not let a handout keep material only because it is nice — R01 and R02
work because they are short and everything in them pays off within a fortnight.

Deliberately absent: lecture content (a recitation never re-teaches a lecture),
assignment solutions (every worked example uses different data), and anything
students do not need yet.
