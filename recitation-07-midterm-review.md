---
marp: true
theme: cpsy1291
paginate: true
math: katex
---

<!-- ===========================================================================
  CPSY 1291 — RECITATION 7: Midterm review (Lectures 1-8)
  TA-led, 80 minutes. Optional. Week 7 — MUST be held before Thu 10/22.

  FORMAT: this is a PROBLEM SESSION, not a lecture. The slides are prompts.
  Put a problem up, give the room 3-4 minutes, take answers from students
  before revealing. Resist the urge to talk through the summary slides — they
  exist so the room can check itself, and the handout carries the full version.

  BUDGET: about 15 min on the compressed summary (slides 2-5), about 55 min on
  problems, and 8 min of open questions at the end. If you fall behind, cut
  problems 5 and 6 — not the open-question block, which is what the students
  who came actually came for.

  DO NOT: predict what is on the exam, or rank topics by likelihood. Say you
  do not know. Point at the learning objectives in the syllabus instead.

  MATERIAL: handout-07-midterm-review.pdf has the full compressed summary of
  L01-L08 and fourteen problems with worked answers. Tell them at the start
  that it exists so nobody transcribes the board.
============================================================================ -->

<div class="eyebrow">Recitation 7 · Week 7</div>

# Midterm review

## Lectures 1–8, as problems

<!--
2 min. Say the format immediately: this is a problem session. You will put
questions up and the room answers them. Fifteen minutes of summary, then an
hour of problems.

Say the thing about the handout: it has the full compressed summary and fourteen
worked problems, so nobody needs to transcribe anything.

If asked what is on the exam: you do not know. Point at the syllabus learning
objectives. Do not speculate — it is unfair to whoever is not in the room.
-->

---

# What the exam can ask

Every lecture in this course does three things:

1. introduces an **object** — a distance, a matrix, a rule
2. says **what it is for**
3. says **when it misleads you**

<mark>Questions come from all three. The third is the one people prepare least and are asked about most.</mark>

<!--
3 min. This is the most useful framing you can give them, so give it first.

Illustrate with one example they know: PCA. The object is the eigendecomposition
of the covariance. What it is for is finding directions of greatest variance.
When it misleads you: it is linear, it is scale-sensitive, and without centering
PC1 is the mean. All three are examinable, and only the first is what students
revise.
-->

---

# L01–L04 in one slide

- **L01** — response matrix; dot product; three distances (Euclidean / cosine / correlation) and what each is blind to; three metric axioms and real violations of all three
- **L02** — RDMs; metric vs non-metric MDS; Shepard's exponential law; clustering always returns a tree; Spearman on the upper triangle
- **L03** — curse of dimensionality; centering; covariance; eigenvectors; projection and reconstruction; eigenfaces; PCA is linear and scale-sensitive
- **L04** — perplexity; the crowding problem; three ways to misread a $t$-SNE map

<!--
5 min. Read the bullets, do not expand them. This slide is a checklist for the
room to audit itself against — anyone who cannot expand a bullet knows what to
revise tonight.

Ask for a show of hands on which bullet is least familiar, and spend two minutes
on whatever wins. Do not spend six.
-->

---

# L05–L08 in one slide

- **L05** — conditions are manifolds; effective dimensionality is *estimated*; mixed selectivity buys separability and pays in abstraction; the metric zoo disagrees with itself
- **L06** — $a = g(\boldsymbol{w}\cdot\boldsymbol{x}+b)$; a neuron measures a direction; two causes of a strong response; tuning curves; one unit cannot do XOR
- **L07** — Hebb → PC1 but diverges; Oja adds one decay term; perceptron vs delta rule; Rescorla–Wagner and blocking
- **L08** — nonlinearity is non-negotiable; XOR by re-representation; universal approximation says *possible*, not *findable*; backprop is the chain rule, not an optimizer

<!--
5 min. Same treatment. The two most misunderstood items on this slide are
"universal approximation says possible, not findable" and "backprop is not an
optimizer" — flag both explicitly, because both are natural exam questions and
both are usually answered wrong.
-->

---

# PROBLEM 1

A unit's activity across three stimuli is $(2, 4, 6)$. Another's is $(1, 2, 3)$.

1. Cosine distance?
2. Correlation distance?
3. What does the pair of answers tell you?

<!--
5 min. Answers: both are 0. Cosine because the vectors point in exactly the same
direction. Correlation because after centering each row both become proportional
to (-1, 0, 1).

Part 3 is the real question: these units carry the same information about the
stimuli and differ only in gain. Neither distance can see that; Euclidean can.
Which one you want is a modelling decision.

Follow-up if the room is fast: what if the second unit were (3, 2, 1)?
-->

---

# PROBLEM 2

$d(A,B) = 0.2$, $d(B,C) = 0.2$, $d(A,C) = 0.9$.

1. Which axiom fails?
2. Name a distance used in this course that permits it.

<!--
4 min. Answers: the triangle inequality, since 0.9 > 0.4. Cosine distance and
correlation distance both permit it.

Push: is that a defect? No — and this is worth landing. Human similarity
judgments violate the triangle inequality too, so a distance that permits it may
be the better model of the behaviour. A "violation" is only a defect if you
needed the axiom.
-->

---

# PROBLEM 3

You forget to center your images before running PCA.

Describe PC1.

<!--
4 min. Answer: it points approximately at the MEAN image, because the uncentered
second-moment matrix is dominated by it. You have spent your first component
describing the average face, and the structure moves to PC2 onward.

Connect it forward: this is the same reason Hebb's rule needs centered data —
Problem 6. Both are "the second moment is not the covariance unless you center."
That one sentence covers two lectures.
-->

---

# PROBLEM 4

Someone reports that their MDS map's horizontal axis is **animacy**.

1. What is the methodological objection?
2. What evidence would answer it?

<!--
5 min. Answers: (1) MDS axes carry no inherent meaning — the solution can be
rotated or reflected freely, so "the horizontal axis" is not a property of the
data at all. (2) Independent evidence: correlate the coordinate with an external
animacy rating, or show a classifier trained on that coordinate alone predicts
animacy on held-out stimuli.

The general lesson is worth naming: naming an axis is a claim that requires
evidence outside the plot. Assignment 1 Part 3j is exactly this.
-->

---

# PROBLEM 5

Hebb's rule is run on data with a large positive mean.

1. What does the weight converge toward?
2. What single preprocessing step fixes it?

<!--
5 min. Answers: toward the mean DIRECTION of the data, not the direction of
greatest variance. Hebbian dynamics follow the second-moment matrix, which
equals the covariance only after centering. Fix: subtract the mean.

Ask them to connect it to Problem 3. Same fact, two lectures apart.
-->

---

# PROBLEM 6

Write Oja's rule.

What does the extra term, relative to Hebb, actually accomplish — and what does it *not* change?

<!--
5 min. Answer: dw = eta * a * (x - a*w). The -eta a^2 w term is a decay
proportional to the unit's own activity: it grows exactly when the weight grows,
driving ||w|| toward 1.

What it does NOT change: the DIRECTION of convergence. Hebb already finds PC1's
direction; Oja fixes the magnitude. Students often say Oja "makes it find PC1",
which is half wrong and is a good exam trap.
-->

---

# PROBLEM 7

The perceptron is trained twice on the same separable data, differing **only in
the order of the examples**, and gives two different boundaries.

1. Why?
2. Would the delta rule do the same?

<!--
6 min. Answers: (1) the perceptron updates only on mistakes and halts as soon as
there are none, so it stops at whichever separating boundary it reaches first —
and which one that is depends on the order the mistakes arrived in. (2) No: the
delta rule minimizes a squared error with a single minimum, and converges to the
same answer regardless of order.

Land the general point: "it converged" and "it converged to a unique answer" are
different claims. Assignment 2 Part 3c measures exactly this spread.
-->

---

# PROBLEM 8

Rescorla–Wagner. Cue A predicts reward for 50 trials. Then A **and** B together,
still rewarded, for 50 more.

1. What happens to $V_B$?
2. What is this called?

<!--
5 min. Answers: V_B stays near zero. The error term is lambda - (V_A + V_B), and
V_A already accounts for the reward, so there is no error left to drive learning
about B. This is BLOCKING.

Why it mattered historically: it shows animals learn from prediction ERROR, not
from mere co-occurrence. B co-occurs with reward on every one of 50 trials and
is still not learned. That result is why R-W is a landmark rather than a
footnote.

Follow-up: what if phase 1 is removed? Both cues learn, sharing the association.
-->

---

# PROBLEM 9

A 2–2–1 network fails to learn XOR from some initializations.

1. Give one concrete mechanism.
2. Give a diagnostic you could run on a failed network.

<!--
6 min. Answers: (1) a hidden unit whose pre-activation is negative for all four
inputs — its ReLU gradient is zero, it never updates, and the network is
effectively 2-1-1, which cannot solve XOR. (2) Run the four inputs through the
trained network and count, per hidden unit, how many produce a non-zero output.
Zero for all four means dead.

Also accept: both hidden units converging to the same function, or a saddle.

This is Assignment 2 Part 4e. If anyone has done it, let them describe what they
found rather than describing it yourself.
-->

---

# PROBLEM 10

Which of these is false, and why?

1. Universal approximation guarantees a wide enough one-hidden-layer network can approximate any continuous function on a bounded domain.
2. Therefore gradient descent will find such a network.
3. Backpropagation is an optimization algorithm.

<!--
6 min. Answers: 1 is true as stated. 2 is FALSE — existence is not findability,
and the theorem says nothing about training. 3 is FALSE — backprop computes the
gradient; gradient descent (or Adam) does the optimizing. Backprop supplies the
derivative, nothing more.

Both false statements are things students write on exams because both sound like
reasonable paraphrases of lecture. Say that out loud.
-->

---

# Six things to be able to state cold

1. the three distances and what each is **blind to**
2. the three metric axioms, and one **real** violation of each
3. what an eigenvector of the covariance is — in a sentence, no algebra
4. why Hebb needs centering, and what Oja's extra term does
5. why the perceptron's answer depends on order and the delta rule's does not
6. why a stack of linear layers is a linear layer

<!--
4 min. Read them out. This is the "night before" list and it is in the handout.

Each of these has appeared in a lecture, an assignment, AND a recitation — which
is the reason they are on the list, and worth saying, because it tells them the
list is not arbitrary.
-->

---

# Open questions

Bring what you are stuck on.

<!--
8 min, and protect it. This is what the people who came actually came for.

If nobody speaks, prime the room with a question of your own: "who can tell me
why we center the data before PCA?" — then let the discussion go where it wants.

Close by saying that Assignment 3 is due Tue 10/27 and that recitation 8 covers
comparing representations — RSA, decoding, and permutation tests.
-->
