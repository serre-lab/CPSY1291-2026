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
| 1 | 1 | A1 | `a1-programming-notes` | `recitation-01-python-numpy` |
| 2 | 2 | A1 | `linear-algebra-notes` | `recitation-02-vectors-to-pca` |
| 3 | 3 | A1 | `handout-03-matrices-fitting-knobs` | `recitation-03-matrices-fitting-knobs` |
| 4 | 4 | A2 | `handout-04-gradients-chain-rule` | `recitation-04-gradients-chain-rule` |
| 5 | 5 | A2 | `handout-05-pytorch-training-loop` | `recitation-05-pytorch-training-loop` |
| 6 | 6 | A3 | `handout-06-heldout-data-and-images` | `recitation-06-heldout-data-and-images` |
| 7 | 7 | Exam 1 | `handout-07-midterm-review` | `recitation-07-midterm-review` |
| 8 | 8 | A3 | `handout-08-rsa-decoding-permutation` | `recitation-08-rsa-decoding-permutation` |
| 9 | 9 | A3 | `handout-09-hooks-gradients-saliency` | `recitation-09-hooks-gradients-saliency` |
| 10 | 10 | A4 | `handout-10-sequences-recurrence-dynamics` | `recitation-10-sequences-recurrence-dynamics` |
| 11 | 11 | A4 | `handout-11-attention-transformers-generative` | `recitation-11-attention-transformers-generative` |
| 12 | 12 | Exam 2 | `handout-12-final-review` | `recitation-12-final-review` |
| 13 | 13 | project | `handout-13-project-clinic-i` | `recitation-13-project-clinic-i` |
| 14 | 14 | project | `handout-14-project-clinic-ii` | `recitation-14-project-clinic-ii` |

Weeks 1 and 2 are served by two standalone handouts rather than per-week ones:
`linear-algebra-notes.pdf` (vectors, distances, dissimilarity matrices, with worked
numbers) and `a1-programming-notes.pdf` (the Python/NumPy/SciPy/scikit-learn calls the
Assignment 1 notebooks use). Both are written as help for a student who found the
linear algebra fast, not as a record of what a session covered.

**Two sessions have hard dates.** R07 must be held before Exam 1 (Thu 10/22; L01–L08
and the four steps of backpropagation from L09), and R12 must be held before
Thanksgiving recess, since Exam 2 (L09–L19) is Tue 12/1 — the first day back.

Assignment windows: A1 to Tue 9/29; A2 to Sun 10/18; A3 Thu 10/22 to Thu 11/5
(L09–L12); A4 Tue 11/10 to Tue 11/24 (L15–L17). There is no fifth assignment.
The project ends with a poster session, report due, on Mon 12/21.

## Building

Handouts (from this directory; the shared preamble is `cpsy1291-handout.sty`):

    latexmk -lualatex handout-06-heldout-data-and-images.tex
    for f in handout-*.tex; do latexmk -lualatex "$f"; done

Decks (from the public repo root, as the Pages workflow does):

    npx @marp-team/marp-cli recitations/recitation-06-heldout-data-and-images.md \
      -o recitations/recitation-06-heldout-data-and-images.html \
      --html --theme-set recitations/theme/cpsy1291-recitation.css

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

**R06–R12 were scoped from the lecture decks L09–L19**, because A3 and A4 did
not exist when they were written. They are therefore provisional: once each
assignment is drafted, re-extract its API surface and cut whatever it does not
use. R07 and R10–R12 were re-mapped to the current lecture numbering on
2026-10-14; R11 was cut to A4's scope (attention, Lecture 17). Do not let a handout keep material only because it is nice — R01 and R02
work because they are short and everything in them pays off within a fortnight.

Deliberately absent: lecture content (a recitation never re-teaches a lecture),
assignment solutions (every worked example uses different data), and anything
students do not need yet.

## Worked-examples supplements (not tied to a week)

| Handout | Serves | What it is |
|---|---|---|
| `linear-algebra-notes` | L01–L02, A1 | vectors, dot products, distances, RDMs, with the numbers written out |
| `pca-statistics-notes` | L03, A1–A2 | mean, variance, covariance, eigenvectors, projection/reconstruction, SVD, standardization, whitening, on four points by hand; the `fit_pca` code |
