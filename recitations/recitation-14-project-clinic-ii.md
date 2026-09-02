---
marp: true
theme: cpsy1291-recitation
paginate: true
math: katex
---

<!-- ===========================================================================
  CPSY 1291 — RECITATION 14: Project clinic II
  TA-led, 80 minutes. Optional. Week 14 — the last class meeting is Thu 12/10
  and presentations are Mon 12/21 at 9:00 AM.

  FORMAT: 40 minutes of teaching, then 40 minutes of workshop. In the
  workshop, each group puts ONE figure on the screen and the room applies the
  figure checklist to it. That is far more useful than more slides, and it is
  the last chance anyone gets to catch an unlabeled axis.

  TONE: some groups will arrive with a result that did not come out. The
  section on that is not a consolation prize — take it seriously and early,
  because those groups need to hear it before they redesign their talk in a
  panic.

  MATERIAL: handout-14-project-clinic-ii.pdf, including the night-before
  checklist.
============================================================================ -->

<div class="eyebrow">Recitation 14</div>

# Project clinic II

## Figures, the report, and the talk

<!--
2 min. Frame: you have results, or you have discovered you will not get the ones
you wanted. Both are fine — the syllabus grades the question, the method, and
the honesty of the interpretation, not whether the model worked.

Say the plan: 40 minutes on figures, report and talk, then 40 minutes where each
group puts one figure up and the room takes it apart.
-->

---

# The one idea

<mark>The claim comes first. Everything else is evidence for it.</mark>

Write your one-sentence result **before** you make a single figure. Then every
figure either supports that sentence or comes out.

Projects that assemble figures first and look for a story afterwards produce
talks that are a **tour of what was done**, not an argument about what is true.

<!--
3 min. Say the last line slowly — the tour-of-what-was-done talk is the single
most common shape of a mediocre project presentation, and naming it now is the
best chance of preventing it.

Ask the room: can each group say its result in one sentence right now? Some
cannot, and knowing that today is much better than knowing it on the 20th.
-->

---

# One figure, one claim

**Description:** "Accuracy by condition and model."

**Claim:** "The texture model loses more accuracy under shape distortion than the
shape model does."

<mark>The claim tells you what to plot — and what to leave out.</mark>

If a figure needs two sentences, it is two figures. If it needs none, it is not a figure.

<!--
3 min. Have the room convert one or two more descriptions into claims out loud.
It is a quick exercise and it changes how they look at their own panels.

The "leave out" half matters as much as the "what to plot" half. Most student
figures are overloaded because nothing was ever deliberately excluded.
-->

---

# The figure checklist

| every figure | why |
|---|---|
| axis labels **with units** | an unlabeled axis is uninterpretable |
| **error bars**, captioned SD or SE | they answer different questions |
| a **chance line** where one exists | the reader cannot infer it |
| $n$ stated — of **what** | runs? stimuli? subjects? |
| readable at slide size | shrink it and check |
| distinguishable without color | ~8% of men are color-blind |

<!--
3 min. This is the slide to photograph, and it is what the workshop will apply.

Row 2 is the one that comes back for the fourth time this term: label them.
Standard deviation says how much a single run varies; standard error says how
well you know the mean. An unlabeled bar asks the reader to trust you instead of
checking you.
-->

---

# Three plots almost always worth making

**The distribution, not just the mean** — plot the individual points over the bars. With ten runs there is no reason not to.

**The thing that would falsify you** — per-item differences, not two averages. Winning on 51% of items by a hair is a different claim from winning on 90%.

**The failure cases** — <mark>the most informative figure in most projects, and almost always missing.</mark>

<!--
4 min. The third one needs encouragement: nobody wants to put their failures on
a slide, and it is exactly what makes a talk credible.

Say it plainly — a presenter who shows what their model got wrong is read as in
command of their work. One who shows only successes is read as not having
looked.
-->

---

# Two things not to do

**Do not truncate a $y$-axis** to make a difference look big.

<mark>If the difference needs a truncated axis, the difference is small — and saying so is better than being caught.</mark>

**Do not use a rainbow colormap** for continuous data. It creates boundaries that
are not in the data. `viridis` is the default for a reason.

<!--
3 min. Both are quick. The truncated axis is the one that gets noticed in
question time, and being caught doing it costs more credibility than the small
effect ever would have.
-->

---

# The report: five sections, five jobs

| section | its one job |
|---|---|
| **Question** | what you asked, and why it matters |
| **Method** | enough that someone could redo it |
| **Results** | what happened — no interpretation |
| **Interpretation** | what it means, and what it does not |
| **Limitations** | what would change your mind |

<mark>Keeping Results and Interpretation apart lets a reader disagree with your reading while accepting your data. That is how science is supposed to work.</mark>

<!--
3 min. The Results/Interpretation split is the structural point. Most student
reports interleave them, and the effect is that a reader who disagrees with one
sentence cannot tell what else to keep.
-->

---

# Limitations is a *strength* section

✗ "More data would help." — a truism, not a limitation.

✓ "We used one architecture, so we cannot separate an effect of depth from an
effect of this particular network."

<mark>That sentence names the confound, bounds the claim, and says which experiment would settle it.</mark>

<!--
3 min. Students write limitations defensively, as damage control. Written well
it is the section that demonstrates they understand their own work.

Ask each group to draft their strongest limitation sentence during the workshop —
it is often the fastest improvement available to a nearly-finished project.
-->

---

# The talk: seven slides

| slide | job |
|---|---|
| 1 | the question, one sentence, and why anyone should care |
| 2 | what you did, **in one picture** — not bullets |
| 3–5 | the result: one figure per slide, one claim per figure |
| 6 | the honest caveat: what would change your mind |
| 7 | what you would do next |

<!--
3 min. Slide 2 is the one people get wrong. Your method is a pipeline, and a
pipeline is a picture — boxes and arrows, data to number.

Three bullets describing the same pipeline take longer to say and are harder to
follow. Draw it. Say that you will not accept "we did not have time" — a boxes-
and-arrows diagram is ten minutes of work.
-->

---

# Rehearsal is not optional

1. **Out loud, with a timer, once.** Silent reading takes about half as long as speaking — a deck that feels right unspoken is roughly twice too long.
2. **Decide who says what.** Silent handovers are the commonest thing to go wrong in a group talk. Write them into the notes.
3. **Cut a *result*, not the caveat and not the question.**

<!--
3 min. Point 1 surprises people every year. Point 3 is the one to argue for:
a talk that covers less and lands beats one that covers everything at speed, and
the caveat slide is what makes the rest believable.
-->

---

# Questions you will be asked

- *What is chance here?* — have the number.
- *How many seeds / subjects / stimuli?* — have the $n$, and say what it is the $n$ **of**.
- *What does a trivial baseline get?*
- *Would this hold for another dataset or architecture?*
- *Why this metric?*

<mark>"We don't know" — said once, with a reason — reads as command of the material. A confident answer to something you did not investigate does not.</mark>

<!--
4 min. Run this as a drill: pick a group, ask them one of these about their own
project, right now. It is friendlier in this room than on the 21st.

The highlighted line is worth repeating. The room can always tell the difference
between an honest "we don't know" and an improvised answer.
-->

---

# If the result did not come out

1. **Say what you expected and what you got** — both, plainly, early.
2. **Show a positive control** — a case where your method *does* detect a known effect.
3. **Bound it** — how large an effect *could* you have detected?
4. **Say what you would do differently.**

<mark>Step 2 turns "our analysis found nothing" into "there is nothing there, and here is the evidence that we could have found it." Those are very different claims.</mark>

<!--
4 min. Take this seriously and do not rush it — some groups in the room are in
exactly this position and need to hear it before they panic-redesign their talk.

Step 3 in plain terms: a null result with a stated sensitivity is informative. A
null without one is silence. That distinction is worth marks and is worth
knowing for life after this course.
-->

---

# WORKSHOP — one figure, on the screen

Each group: put up **one figure**.

The room applies the checklist:
axis labels · error bars (which?) · chance line · $n$ of what · readable at size · color-safe

**And: what claim does it make?**

<!--
40 min, and this is the session.

Go group by group. Let the ROOM find the problems — they are much better at
spotting a missing chance line in someone else's figure than in their own, and
having found it once they will check their own.

Be encouraging and specific. Every figure will have at least one thing on the
list; that is expected, and finding it now is the entire point.

Close by pointing at the night-before checklist in handout-14.pdf, especially
item 10: cut the slide you are keeping only because the work was hard. It
catches something in almost every project.
-->
