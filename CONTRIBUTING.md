# Contributing

Public course materials for CPSY 1291, Fall 2026 (Brown University).
Maintainers: Thomas Serre (@tserre), Peisen Zhou (@PPPayson).

- `recitations/recitation-NN-*.md` — one Marp deck per weekly 80-minute recitation.
- `recitations/handout-NN-*.tex/.pdf` — printed handout paired with a week, where one exists.
- `recitations/bootcamp-notes.tex/.pdf` — standalone Python/numpy notes for the early weeks.
- Every push to `main` rebuilds the decks and publishes to GitHub Pages automatically,
  and students may follow the published links at any time — finish an edit before pushing.

House rules for any change, by any contributor or tooling:

1. **Never rename a published file.** Filenames are the published URLs, and Canvas links
   to them — a rename silently breaks the course site. Retitle content inside the file;
   the filename stays. (Applies to decks, handout PDFs, and notebooks.)
2. Match the existing decks' Marp formatting and voice: direct, no hype, no exclamation
   marks, no emoji. Content edits, not restyles.
3. Recitations are scoped from the current assignment. Every assignment reference
   (e.g. "A1 2g") must match the released student notebooks; if you cannot verify a
   reference, say so in the commit message rather than guessing.
4. Decks fit 80 minutes; keep the per-slide "~N min" timings honest and a presenter note
   on every slide.
5. Never include solutions to graded questions. Code skeletons appear only where the
   course has decided to hand them out.
6. Before pushing, rebuild what you touched
   (`npx @marp-team/marp-cli <deck>.md -o <deck>.html --html --theme-set recitations/theme/cpsy1291-recitation.css`
   — the same invocation CI uses) and run
   `node recitations/build/check_overflow_bootcamp.js` — zero clipped slides is a hard gate.
7. Handouts compile with lualatex, never pdflatex.
8. Structural changes (adding/removing a week, changing the handout scheme) are
   Thomas's call — propose first.

When a week's materials are settled, Thomas tags them (`r01-final`, ...); the tag is the
version of record for that week.
