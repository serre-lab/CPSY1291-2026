---
marp: true
theme: cpsy1291
paginate: true
math: katex
---

<!-- ===========================================================================
  CPSY 1291 — RECITATION 13: Project clinic I
  TA-led, 80 minutes. Optional. Week 13 — the final exam is Tue 12/1 and
  project work begins Thu 12/3, so hold this at the end of week 13.
  Presentations are Mon 12/21.

  FORMAT: half teaching, half workshop. Slides 2-13 are about 48 minutes.
  Then break the room into project groups and have each group write its
  question on the sentence-test template and read it out. That last 30 minutes
  is where the session earns its place — a vague question said out loud in
  front of peers gets fixed on the spot.

  DO NOT: approve or reject project ideas. You are helping them SCOPE, not
  grading. If a group's idea is too large, ask what they would cut, rather
  than telling them.

  MATERIAL: handout-13-project-clinic-i.pdf, including the ten questions and
  the day-by-day schedule.
============================================================================ -->

<div class="eyebrow">Recitation 13 · Week 13</div>

# Project clinic I

## A question you can answer, and a baseline by Friday

<!--
2 min. Set the frame: two and a half weeks, roughly 31 hours each.

Say the risk plainly. It is NOT that the model fails — a negative result clearly
analyzed is a perfectly good project and the syllabus says so. The risk is
spending twelve days on setup and having nothing to say on the thirteenth.

Say the plan for today: about 48 minutes on scoping and de-risking, then 30 minutes
where each group writes its own question down and reads it out.
-->

---

# The one rule

<mark>Get to a bad answer fast.</mark>

A pipeline that runs end to end on **ten items** and produces a **wrong number**
is worth more on day three than a beautiful half-finished component.

It tells you what the real obstacles are. Everything after that is improvement —
and improvement is schedulable in a way that discovery is not.

<!--
3 min. This is the single most valuable thing in the session. Say it, then
defend it, because it runs against the instinct to build things properly.

The argument: on day three you do not know what will be hard. The end-to-end run
tells you. A component built beautifully in isolation might be solving a problem
you were about to abandon.
-->

---

# The three shapes, and what each must have by day 3

| shape | the risk | by day 3 |
|---|---|---|
| **extend a paper** | the original does not reproduce | the original result, on a subset |
| **improve a model** | no measurable axis | a **baseline number** on your metric |
| **compare with data** | the data never arrives | the data loaded and **plotted** |

<mark>Every one of these is a de-risking step, not a result. If it fails you still have eleven days to change course.</mark>

<!--
5 min. Go through each row and name the failure it prevents.

"Extend a paper" is the most dangerous of the three, because reproducing the
original can quietly consume the whole two weeks. The subset requirement is what
keeps that bounded — reproduce it approximately, on 200 items, and move on.
-->

---

# The sentence test

<mark>"We measure ______ on ______, comparing ______ against ______, and we expect ______."</mark>

Every blank must be **concrete**.

If any blank is vague, the project is not scoped — and no amount of coding fixes that.

<!--
4 min. Have someone volunteer a project idea and fill the template on the board
in front of the room. Do it once publicly before asking groups to do it
themselves later.

Expect the first attempt to fail on blanks 4 and 5. That is normal and is
exactly what the exercise is for.
-->

---

# Two failure patterns to check for

**No comparison.**
✗ "We will see whether a CNN represents shape."
✓ "We compare a CNN's accuracy on shape-preserving vs texture-preserving distortions."

**No expectation.**
If you cannot say what you expect, you cannot tell an **interesting result** from
a **bug**.

<mark>Write the expectation down before you run anything. Being wrong is a finding; being wrong without a record is confusion.</mark>

<!--
4 min. The first pattern kills more projects. A question with one condition has
no possible result — whatever number comes out, there is nothing to compare it
to.

The second is subtler and matters most at 2am on day eleven, when a surprising
number appears and the group has no way to judge whether to celebrate or debug.
-->

---

# What fits in 31 hours

| fits | does not fit |
|---|---|
| fine-tuning a pretrained model | training a large model from scratch |
| a few thousand images | ImageNet |
| 2–4 conditions | a factorial sweep |
| one dataset, carefully | three datasets, carelessly |
| a laptop or Colab free tier | anything needing a reserved GPU |

<!--
4 min. Read the right column as a list of things people have actually attempted
in a two-week project.

The last row is worth dwelling on: Colab's free tier restricts GPU access and
guarantees nothing. A project that requires a GPU to produce its main result is
a project that might produce no result.
-->

---

# Assume the network is slow

<mark>Downloads in this course have run at kilobytes per second.</mark>

- start every download on **day one**, in the background
- build against a small local subset while it runs

A project blocked on a download at day ten is the most avoidable failure there is.

<!--
3 min. This is not hypothetical — it is what happened while the assignments were
being built. Say so; a concrete anecdote lands better than advice.

Corollary: check the size of what you are downloading BEFORE you start it. A
40 GB dataset at 75 KB/s is six days.
-->

---

# Load it, then *look* at it

```python
print(X.shape, X.dtype)
print(np.isnan(X).sum(), np.isinf(X).sum())
print(X.min(), X.max(), X.mean())
print(np.unique(y, return_counts=True))     # class balance
plt.imshow(X[0]); plt.show()                # LOOK at one item
```

<mark>Five lines. Each of the things they catch has cost somebody a week.</mark>

<!--
4 min. Enumerate what they catch: transposed dimensions; images in [0, 255]
where the model expects [0, 1]; a class that occurs eleven times; dead units
that will produce nan correlations; files that decoded into noise.

The last line is the one people skip and the one that catches the worst errors.
Looking at one item takes two seconds and there is no substitute for it.
-->

---

# Write a loader, once

```python
def load_data(root, subset=None):
    """Return X, y, meta. The ONLY place that touches the filesystem."""
    ...
```

Everything downstream calls this.

<mark>When you discover on day nine that the labels were off by one, you fix it in one place — rather than hunting six notebooks that each loaded the data slightly differently.</mark>

<!--
3 min. This is the highest-leverage thirty minutes of engineering in a
two-week project, and it feels like a detour on day two.

Group-specific version: three people each loading the data their own way
guarantees three subtly different datasets, and the disagreement surfaces as an
inexplicable result rather than as an error.
-->

---

# The skeleton, on ten items

```python
X, y, meta = load_data(ROOT, subset=10)     # 1. ten items
model      = get_model()                    # 2. pretrained, off the shelf
preds      = predict(model, X)              # 3. no batching, no speed
score      = evaluate(preds, y)             # 4. one number
plot(score)                                 # 5. one figure
```

**Why ten?** It runs in seconds, you can verify every number by hand, and scaling
10 → 10,000 is a *boring* change.

<mark>Scaling from "nothing runs" to "something runs" is not boring, and it is the part with unknown cost.</mark>

<!--
4 min. Walk the five lines. Every one is deliberately stupid, and that is the
design.

When the skeleton works: raise subset, start the full run, and go and have
lunch. That is a good day three.
-->

---

# Two baselines you always need

**Chance.** With imbalanced classes it is the majority proportion, **not** $1/k$.
Compute it; do not assume it.

**A trivial model.** Raw pixels instead of network features. Logistic regression
instead of a fine-tuned transformer.

<mark>If the dumb thing does nearly as well, that is the most important fact about your problem — and you want to learn it on day three, not in your last slide.</mark>

<!--
4 min. The trivial baseline is what separates a project that has a result from
one that has a number.

Give an example: a decoder gets 78% on your neural data. Impressive? Raw
luminance gets 74%. Now you know what the project is actually about, and you
have eleven days to do something about it.
-->

---

# Four habits, each under a minute

1. **One seed, set once, at the top** — and remember: reproducible ≠ meaningful. Headline numbers get several seeds.
2. **Save results, not just figures** — a `.csv` or `.npz` as you go.
3. **Never edit a figure by hand** — you will need to regenerate it.
4. **One notebook per question, not per person** — plus a shared `utils.py`.

<!--
4 min. Habit 4 is group-project-specific and prevents the most common lost day:
three people editing three copies of the same notebook.

Habit 3 sounds pedantic until day twelve, when a reviewer asks for a change and
the figure cannot be rebuilt.
-->

---

# The deadline that matters is day 9

| by | you have |
|---|---|
| day 3 | data loaded and looked at; skeleton on 10 items; chance + trivial baseline |
| day 6 | the real comparison, run once, at full size, one figure |
| **day 9** | **the comparison across seeds, with error bars; one control** |
| day 12 | every figure final; the result in one sentence |
| day 14 | slides built, rehearsed once, out loud, with a timer |

<!--
4 min. Say why day 9 and not day 14: if the main comparison is not running by
day 9 there is no time to add error bars, and a result without error bars cannot
be defended.

That is the single most common way a technically fine project loses marks —
not a wrong analysis, an undefendable one.
-->

---

# WORKSHOP — your turn

In your groups, write your question on the template:

<mark>"We measure ______ on ______, comparing ______ against ______, and we expect ______."</mark>

Then: **what is your day-3 de-risking step?**

Be ready to read both out.

<!--
30 min, and this is the point of the session.

Give groups 10 minutes to write, then go round the room. For each: read the
sentence, then ask the room which blank is vaguest. Peers are much better at
spotting this than the group that wrote it.

Your job is to help them SCOPE, not to approve or reject. If an idea is too
large, ask what they would cut — do not tell them.

Close by pointing at the ten questions in handout-13.pdf, and at next week's
session: figures, the report, and the talk.
-->
