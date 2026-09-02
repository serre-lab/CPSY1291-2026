---
marp: true
theme: cpsy1291
paginate: true
math: katex
---

<!-- ===========================================================================
  CPSY 1291 — BOOTCAMP RECITATION 1 of 2: Python and NumPy
  TA-led, 80 minutes. Optional session; students who already program
  comfortably in Python should skip it.

  HOW TO RUN THIS DECK: every slide's presenter note gives a timing, what to
  say, and (where relevant) the answer to the exercise on the slide. Slides
  marked EXERCISE are meant to be done live — put the prompt up, let them try
  for 2-3 minutes, then take answers from the room before revealing.

  SCOPE: reconciled against the FINAL Assignment 1 (2026-09-01). Everything
  here is required by A1 parts 1-2, which students are starting this week:
  axis/keepdims and the broadcasting error (1b, 2c), boolean arrays (1f, 2i,
  3e), and fancy indexing (1e, 1f, 2g). Nothing else is here. Do NOT add
  gradients, the chain rule, or how networks are trained — those arrive with
  Assignment 2 and its own recitation. Concepts (Shepard, psychological space)
  are carried by L1/L2, not here (Recitation 2 keeps only pointers).

  MATERIAL: the companion notes (bootcamp-notes.pdf) cover the same ground in
  prose. Tell students at the start that they do not need to take notes.
============================================================================ -->

<div class="eyebrow">Bootcamp · Recitation 1 of 2</div>

# Python and NumPy

## The habits the assignments assume

<!--
2 min. Introduce yourself, say office hours, and set expectations:
this session and the next are OPTIONAL, and they exist so that nobody loses a
weekend to a shape error. Anyone comfortable with numpy broadcasting can leave
now with no penalty — say that out loud, it keeps the room self-selected and
the pace honest.

Say what this is NOT: not a Python course, not comprehensive. It is the
specific set of habits Assignment 1 rewards. A1 parts 1 and 2 are out;
everything in this session appears in them, under its real section label.
-->

---

# Who this is for

- You know how to program, but **not in Python** — you are coming from Java, Racket, or MATLAB
- Or: you know Python, but have never used **NumPy** in anger
- Or: you have used NumPy but `axis=` and broadcasting still feel like guessing

<mark>If none of these is you, this session is safe to skip.</mark>

The companion notes cover the same material in prose — you do not need to take notes.

<!--
2 min. The third bullet catches more people than they expect, including strong
CS students. Encourage staying for the broadcasting and indexing sections even
if the early part is familiar — those are the parts that actually cost people
time in A1.
-->

---

# One rule, and everything else follows

<mark>Before you write a line of code, be able to say in one sentence what it is supposed to compute.</mark>

If you cannot say it, the thing you are stuck on is the **idea**, not the syntax — and no amount of bracket-fiddling will fix it.

This is also why the assignments ask you to predict what a result should look like *before* you compute it.

<!--
3 min. This is the most important slide in the deck; do not rush it.

Make it concrete with a story: a student spends an hour debugging a distance
function, and the actual problem is that they never decided whether the
distance should ignore brightness. No error message will ever tell you that.

Connect it to how they will really work: outside this course they will use AI
tools to write code, and those tools are only useful if you can state what you
want and recognize a wrong answer. That judgment is what we are building here.
Thomas is explicit about this — assignments are their own work, but the skill
being trained is deciding WHAT to compute, not typing it.
-->

---

# A list of numbers is a point

Write down $p$ numbers about something — pixel values, firing rates, ratings —
and you can treat that list as a **point** in a $D$-dimensional space.

Two things are "similar" when their points are **close together**.

<mark>The whole first assignment is that sentence, made precise.</mark>

<!--
2 min. This is the bridge from "programming session" to "why we are here", and
it is the one conceptual slide in an otherwise practical deck. The psychology
behind it — Shepard, psychological space — is lecture material and Recitation
2's opening; the notebooks also carry glossary cells for the vocabulary.

Draw it: two axes on the board, a few dots. Ask what "close" means, and take
two or three answers — you want someone to say "depends what you mean by
distance". That is exactly the assignment's first question, so tell them they
have just written 1b themselves.
-->

---

# Lists and arrays are not the same thing

```python
xs = [3, 1, 4]              # list: anything, any type, grows
import numpy as np
X = np.array([[1., 2.], [3., 4.]])   # array: numbers, one type, has a SHAPE
```

| | `list` | `np.array` |
|---|---|---|
| holds | anything | one numeric type |
| arithmetic | `xs * 2` **repeats** it | `X * 2` **doubles every entry** |
| speed on 4,096 numbers | slow | fast |

<!--
4 min. The `xs * 2` row is the demo to actually run live — it surprises people
from every background, and it is a real source of silent bugs.

MATLAB refugees: reassure them arrays behave much like MATLAB matrices, with
the crucial difference that indexing starts at 0 and the last index is n-1.
Java refugees: no type declarations, no compile step, and indentation is
syntax.
-->

---

# Shape is your unit test

```python
X.shape      # (120, 4096)  -> 120 stimuli, 4096 features each
X.dtype      # float64
X[0].shape   # (4096,)      -> one stimulus's vector
```

<mark>Print shapes constantly. Most numerical bugs are shape bugs, and they confess immediately if you look.</mark>

**Course convention:** data is always **stimuli × features** — one row per image, one column per pixel or unit.

<!--
5 min. Have them type along. Then break something on purpose: transpose a
matrix and show the error message, so the first time they meet
"could not be broadcast together with shapes ..." it is in a room with help
rather than at 2am.

The stimuli-by-features convention is assumed by every assignment and by the
lectures. Say it twice. A1's own arrays follow it: 120 images x 4,096 late-layer
units, and it is the shape every function in 1b takes.
-->

---

# Slicing

```python
X[0]        # first row
X[:, 3]     # column 3, across all stimuli
X[:5]       # first five rows
X[::24]     # every 24th row
X[-1]       # last row
```

`X[::24]` looks exotic but you will need it: one of our datasets is **51 objects photographed from 24 angles**, stored in order — `F[::24]` is exactly what 3c asks for.

<!--
3 min. Slices are VIEWS, not copies — modifying a slice modifies the original.
Mention it once; do not dwell.

The X[::24] example lands better if you ask first: "if the data goes object 1
from 24 angles, then object 2 from 24 angles, how would you get one picture of
each object?" Let them derive the stride. In A1 3c they will use F[::24] to get
51 images at a single viewpoint.
-->

---

# `axis=` — the rule that ends the guessing

<mark>The axis you name is the axis that disappears.</mark>

```python
X.shape                 # (120, 4096)
X.mean(axis=0).shape    # (4096,)  -- averaged DOWN the rows
X.mean(axis=1).shape    # (120,)   -- averaged ACROSS the columns
```

`keepdims=True` keeps it as a length-1 dimension instead of removing it:

```python
X.mean(axis=1, keepdims=True).shape   # (120, 1)
```

<!--
6 min. THE central slide of the session. Do not move on until the rule has been
said back to you by someone in the room.

Draw the (120, 4096) grid on the board and physically cross out the axis being
collapsed. The mnemonic is reliable and students keep it. "axis=0 collapses
images, leaving one number per unit" — that sentence is 2c's dead-unit check
(variance across the 120 images, per unit) in a nutshell.

keepdims will look pointless right now. Say only: "it exists so subtraction
lines up — we will see why in three slides." Then deliver on that.
-->

---

# EXERCISE 1

`X` has shape `(120, 4096)`.

1. What shape is `X.sum(axis=0)`?
2. What shape is `X.max(axis=1)`?
3. Which of the two is "the average stimulus"?

<!--
4 min. Answers: (1) (4096,)  (2) (120,)  (3) X.mean(axis=0) — averaging down
the rows gives one number per feature, which is the average stimulus.

Question 3 is the one that separates rule-following from understanding. If the
room struggles, go back to the crossed-out grid.
-->

---

# Broadcasting, and the error it throws

When shapes differ, NumPy stretches size-1 dimensions to match.

```python
X - X.mean(axis=0)                   # (120,4096) - (4096,)  -> centers each COLUMN
X - X.mean(axis=1, keepdims=True)    # (120,4096) - (120,1)  -> centers each ROW
X - X.mean(axis=1)                   # (120,4096) - (120,)   -> ERROR
```

```
ValueError: operands could not be broadcast together with shapes (120,4096) (120,)
```

**Rule:** compare shapes from the right. Each pair must be equal, or one must be 1 — and `(120,)` lines up against `4096`, not against `120`.

<!--
6 min. Here is the promised payoff for keepdims, and the single most common
error in A1 part 1 — show the traceback live, on purpose, so the first time
they meet it is in this room.

Make the difference between the two working lines concrete: centering each
column removes "this pixel is bright on average"; centering each row removes
"this image is bright overall". They lead to different answers to the same
question, and in 1b the row version is what turns cosine distance into
correlation distance.

Walk the error: (120,4096) vs (120,) aligned from the RIGHT pairs 4096 with
120. Neither is 1, so NumPy refuses. keepdims=True makes it (120,1), which
stretches. That is all keepdims is for.
-->

---

# EXERCISE 2

You want to subtract each **stimulus's own mean** from its feature vector.

Which is correct, and what happens if you use the other?

```python
A = X - X.mean(axis=0)
B = X - X.mean(axis=1, keepdims=True)
```

<!--
4 min. Answer: B. A removes each FEATURE's mean across stimuli — a different,
also useful operation (PCA does exactly A, as part 3 will point out).

Then ask what `X - X.mean(axis=1)` (without keepdims) does: the error from the
previous slide. Having just seen it, someone will say so — good, that means it
stuck. B is one line of correlation_dist in 1b.
-->

---

# 1b hands you the hard idiom — read it

The notebook gives you the Euclidean shape-move; you supply the other two:

```python
sq = (X ** 2).sum(axis=1)                       # (n,)   given in 1b
D2 = sq[:, None] + sq[None, :] - 2 * (X @ X.T)  # (n, n) given in 1b

n = X / np.linalg.norm(X, axis=1, keepdims=True)   # each row to unit length
C = 1 - n @ n.T                                    # cosine distance, all pairs
```

Correlation distance: **center each row first** (`X - X.mean(axis=1, keepdims=True)`), then the cosine lines.

<mark>One more thing: `np.sqrt(np.maximum(D2, 0))` — floating point leaves tiny negatives where the true value is 0, and `sqrt` of those is `nan`.</mark>

<!--
5 min. This is 1b's HINT block, read slowly. The notebook is explicit that the
Euclidean idiom is given because the definitions are the point, not the idiom —
so do not re-derive it, just decode it: sq[:, None] is (n,1), sq[None, :] is
(1,n), and broadcasting turns their sum into every pairwise combination.

The cosine skeleton is the two lines on the slide: normalize rows (keepdims
again, doing real work), then 1 - n @ n.T. Correlation is the same two lines
after row-centering — the notebook says "one line different" and this is what
it means.

The maximum(D2, 0) clip is not paranoia; they WILL hit sqrt of -1e-13 -> nan.
Say it now so they recognize it later.
-->

---

# Masks, counting, sorting

```python
mask = (taxon == 2)         # boolean, one per stimulus
X[mask]                     # just those rows
mask.sum()                  # how many (True counts as 1)
mask.mean()                 # what FRACTION — a boolean's mean is a rate

np.unique(taxon, return_counts=True)    # what values, how many of each
np.argsort(v)[:5]           # indices of the 5 SMALLEST
np.argsort(v)[::-1][:5]     # indices of the 5 LARGEST
```

<!--
4 min. argsort returns INDICES, not values — say it, then show it, because the
confusion is universal.

mask.mean() is the quiet star: 1f asks for the PERCENTAGE of 200,000 triples
that violate the triangle inequality, and that is one .mean() on a boolean
array. Flag it here, deliver it three slides on.

Motivate argsort: "the five images people rated most similar to this one" is
an argsort on a row of the similarity matrix — 2a makes them do exactly that,
and it is how they check the data was loaded correctly.
-->

---

# Booleans: `&` `|` `~`, and the other error

```python
(taxon == 2) | (taxon == 5)     # parentheses required — | binds tighter than ==
np.isin(cat, [0, 2])            # True where cat is 0 OR 2, in one call
np.argmax(cum >= 0.85)          # FIRST index where a condition holds
```

Write `and` / `or` between arrays and you get:

```
ValueError: The truth value of an array with more than one element is
ambiguous. Use a.any() or a.all()
```

<mark>Element-wise logic on arrays is `&` `|` `~`, always with parentheses.</mark>

<!--
6 min. Every one of these lines is lifted from A1. np.isin(cat, [0, 2]) is 3e's
animate indicator — Animal is category 0 and Face is category 2, so that one
call builds the binary variable the whole animacy claim rests on. The
boolean-argmax is "how many components reach 85% of the variance" (3b) —
argmax of a boolean returns the first True because True > False. (The notebook
also offers np.searchsorted for that; both are fine, this one generalizes to
any condition.)

Show the and/or error live: (taxon == 2) and (taxon == 5). Read the message
aloud — Python is asking "do you mean ANY of these, or ALL?", because a
120-element boolean is not one truth value. any()/all() answer that question
when you really do want one truth value; &/| when you want 120 of them.

2i's contingency table is a nested loop with a boolean & — mention it as the
place parenthesized & shows up next.
-->

---

# EXERCISE 3

`S` is a 120×120 **similarity** matrix. Write one line that gives the index of
the stimulus most similar to stimulus 7.

<!--
4 min. Answer: np.argsort(S[7])[-2]

The trap is [-1], which returns stimulus 7 itself — nothing is more similar to
something than itself. Let the room fall into it; do not warn first. It is a
memorable lesson about checking that an answer is sensible, and 2a says it in
so many words: "an image is always maximally similar to itself, so exclude the
query image from its own neighbor list."

Follow-up if quick: how would you get the five most similar, excluding itself?
Answer: np.argsort(S[7])[::-1][1:6]
-->

---

# Fancy indexing: index with arrays, not numbers

```python
D = np.arange(16).reshape(4, 4)
a = np.array([0, 2, 3])
b = np.array([1, 0, 3])
D[a, b]        # array([ 1,  8, 15])  — the pairs (0,1), (2,0), (3,3)
```

Two index arrays pull one element **per pair** — 3 pairs here, 200,000 in 1f:

```python
t = rng.integers(0, len(X), (200_000, 3))
a, b, c = t.T
viol = D[a, c] > D[a, b] + D[b, c] + 1e-9
viol.mean()                       # the fraction 1f asks for
```

<!--
6 min. Work the 4x4 on the board until D[a, b] is obvious: element-wise pairs,
NOT a submatrix. Without this idiom students write a Python loop over 200,000
triples, watch nothing happen for minutes, and conclude Colab hung — that is
the failure mode this slide exists to prevent.

The bottom block is 1f's own skeleton (the notebook gives the sampling lines).
Two things to say about it: compute each full distance matrix ONCE and index
into it — recomputing distances inside a loop takes minutes rather than
seconds, and the notebook warns exactly that; and the + 1e-9 tolerance is not
decoration — 1d showed your functions agree with SciPy to ~10 decimal places,
and a bare > counts every rounding disagreement as a violation.

And there is the promised boolean .mean(): the violation rate in one call.
-->

---

# Two more moves you will need this week

**The upper triangle, in fixed pair order** — every pair once, no diagonal:

```python
iu = np.triu_indices(120, k=1)    # a PAIR of index arrays
D[iu]                             # 7,140 values — same fancy indexing as D[a, b]
```

**Reordering rows *and* columns** — 1e sorts its matrices by category:

```python
order = np.argsort(taxon)
D[np.ix_(order, order)]           # the reordered 120x120 grid
D[order, order]                   # NOT that — paired indexing: 120 diagonal entries
```

<!--
5 min. D[iu] is the same move as the last slide — iu is literally a pair of
index arrays — and it matters because the assignment leans on it repeatedly:
1e's provided check correlates upper triangles, 2g pairs s_ij = S_human[iu]
with MDS distances, and 2j/2k reuse the same pair order. Say the phrase "same
pair order" now; it is why two D[iu] vectors can be correlated at all.

np.ix_ is 1e verbatim: order = np.argsort(taxon), then np.ix_(order, order).
The contrast on the slide is the lesson — D[order, order] silently applies the
PAIRED rule from the previous slide and hands back 120 numbers, no error. If a
student's "sorted" RDM comes out as a vector, this is why. np.ix_ builds the
open grid: every row in order against every column in order.
-->

---

# Getting data in

```python
d = np.load(f'{DATA}/a1_activations_behav_trained.npz', allow_pickle=True)
d.files                     # what's inside
A = d['late']               # index it like a dictionary
```

Then, before computing **anything**:

```python
print(A.shape, A.dtype, A.min(), A.max())
np.isnan(A).sum()
plt.imshow(S); plt.colorbar()
```

<!--
3 min. An .npz is a zip of named arrays — index by name, not position. The
loading cells in A1 are PROVIDED, so nobody has to write this; the point is
reading them and knowing what came back.

The second block is the habit to instill: look at the data before trusting a
number computed from it. 1a makes them do this explicitly (and its output —
most activations are exactly zero, because ReLU — answers a question shapes
alone cannot), and part 3's recordings have real NaNs waiting for exactly the
np.isnan check.
-->

---

# When something breaks

1. **Shape.** Print it. Right dimensions, right way round?
2. **Type and range.** `dtype`, `min()`, `max()`
3. **Missing.** `np.isnan(X).sum()` — one `nan` spreads everywhere
4. **Degenerate.** Anything with zero variance breaks a correlation
5. **A case you can check by hand.** Try a 2×2
6. **Plot it.** A wrong matrix usually looks wrong

<!--
2 min. Walk down the list in order — the discipline is the point, and the first
two catch most problems. Item 4 is 2c verbatim: the untrained network has
hundreds of zero-variance units, and 2c asks what they do to a correlation.

Tell them to bring a failing cell to office hours having already done steps 1-3,
and that doing so usually means they no longer need office hours.
-->

---

# Where this lands in Assignment 1

| You just learned | Where it shows up |
|---|---|
| shapes, `axis=`, `keepdims` | every cell; the three distance functions (1b) |
| the broadcasting error | centering rows vs columns (1b, 2c) |
| boolean masks, `.mean()`, `np.isin` | violation rates (1f), clusters vs taxon (2i), animacy (3e) |
| `D[a, b]`, `D[iu]`, `np.ix_` | 200,000 triples (1f), sorted RDMs (1e), pairing matrices (2g–2k) |
| `argsort` | nearest neighbors (2a) |
| `np.load`, sanity checks | the first section of every part |

**Next session:** curve fitting, R², the library calls, and the skeletons for parts 2–3.

<!--
2 min. Closing the loop matters — students sit through a methods session much
more willingly once they can see the assignment on the other side of it. Read
the table out; every row carries a real section label they can go find.

The concepts — psychological space, Shepard's law, why any of this is science —
are what L1/L2 carry this week — Recitation 2 keeps them to pointers; the notebooks
also start with glossary cells for the vocabulary. Point them at
bootcamp-notes.pdf for the prose version, and take questions with whatever time
is left.
-->
