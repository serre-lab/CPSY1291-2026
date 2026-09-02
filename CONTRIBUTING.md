# Contributing

Recitation materials for CPSY 1291, Fall 2026 (Brown University).
Maintainers: Thomas Serre (@tserre), Peisen Zhou (@PPPayson).

- `recitation-NN-*.md` — one Marp deck per weekly 80-minute recitation.
- `handout-NN-*.tex/.pdf` — printed handout paired with a week, where one exists.
- `bootcamp-notes.tex/.pdf` — standalone Python/numpy notes for the early weeks.
- Every push to `main` rebuilds the decks and publishes to GitHub Pages automatically,
  and students may follow the published links at any time — finish an edit before pushing.

House rules for any change, by any contributor or tooling:

1. Match the existing decks' Marp formatting and voice: direct, no hype, no exclamation
   marks, no emoji. Content edits, not restyles.
2. Recitations are scoped from the current assignment. Every assignment reference
   (e.g. "A1 2g") must match the released student notebooks; if you cannot verify a
   reference, say so in the commit message rather than guessing.
3. Decks fit 80 minutes; keep the per-slide "~N min" timings honest and a presenter note
   on every slide.
4. Never include solutions to graded questions. Code skeletons appear only where the
   course has decided to hand them out.
5. Before pushing, rebuild what you touched
   (`npx @marp-team/marp-cli <deck>.md -o <deck>.html --html`) and run
   `node build/check_overflow_bootcamp.js` — zero clipped slides is a hard gate.
6. Handouts compile with lualatex, never pdflatex.
7. Structural changes (adding/removing a week, changing the handout scheme) are
   Thomas's call — propose first.

When a week's materials are settled, Thomas tags them (`r01-final`, ...); the tag is the
version of record for that week.
