---
marp: true
theme: cpsy1291-recitation
paginate: true
math: katex
---

<!-- ===========================================================================
  CPSY 1291 — RECITATION 7: Exam 1 review (Lectures 1-8, and backprop from 9)
  TA-led, 80 minutes. Optional. Week 7 — MUST be held before Thu 10/22.

  SCOPE: Exam 1 covers L01-L08 plus the four steps of backpropagation from
  L09 (forward pass, output error, error sent backward, gradient = error x
  input). Nothing else from L09 (softmax, the training loop, autograd,
  transfer learning) and nothing from L10 onward.

  FORMAT: this is a PROBLEM SESSION, not a lecture. The slides are prompts.
  Put a problem up, give the room 3-4 minutes, take answers from students
  before revealing. Resist the urge to talk through the summary slides — they
  exist so the room can check itself, and the handout carries the full version.

  BUDGET: about 15 min on the framing and the three summary slides (slides
  2-5), about 49 min on problems, 4 min on the cold list, and 7 min of open
  questions at the end. If you fall behind, cut problems 5 and 6 — not the
  backprop problem and not the open-question block, which is what the
  students who came actually came for.

  DO NOT: predict what is on the exam, or rank topics by likelihood. Say you
  do not know. Point at the learning objectives in the syllabus instead.

  MATERIAL: handout-07-midterm-review.pdf has the full compressed summary of
  L01-L08, the four backprop steps, and sixteen problems with worked answers.
  Tell them at the start that it exists so nobody transcribes the board.
============================================================================ -->

<div class="eyebrow">Recitation 7</div>

# Exam 1 review

## Lectures 1–8 and the four steps of backpropagation, as problems

<!--
2 min. Say the format immediately: this is a problem session. You will put
questions up and the room answers them. Fifteen minutes of summary, then an
hour of problems.

Say the scope once, precisely: Lectures 1-8, plus the four steps of
backpropagation from Lecture 9. Nothing after that.

Say the thing about the handout: it has the full compressed summary and sixteen
worked problems, so nobody needs to transcribe anything.

If asked what is on the exam: you do not know. Point at the syllabus learning
objectives. Do not speculate — it is unfair to whoever is not in the room.
-->

---

# What the exam can ask

Every lecture in this course does three things:

1. introduces an **object** — a distance, a matrix, a score, a rule
2. says **what it is for**
3. says **when it misleads you**

<mark>Questions come from all three. The third is the one people prepare least and are asked about most.</mark>

<!--
3 min. This is the most useful framing you can give them, so give it first.

Illustrate with one example they know: PCA. The object is the eigendecomposition
of the covariance. What it is for is finding directions of greatest variance.
When it misleads you: it is linear, it depends on the units of each feature, and
the direction with most variance need not be the one that separates the groups.
All three are examinable, and only the first is what students revise.
-->

---

# L01–L04 in one slide

- **L01** — representation as a vector; the response matrix $\mathbf{X}$ ($N$ images × $D$ units); single trials are noisy, so average and check split-half reliability; the RDM
- **L02** — Euclidean / cosine / correlation and what each ignores; which are metrics; MDS, Stress-1, three warnings; non-metric MDS; Shepard's law; RSA = Spearman on upper triangles
- **L03** — centering; covariance; eigenvectors as directions $\mathbf{C}$ does not turn; eigenvalue = variance; scores and reconstruction; scree plot; scale changes PCA; most variance ≠ most useful
- **L04** — ambient / embedding / intrinsic dimension; Isomap; $t$-SNE's $P$, $Q$ and perplexity; UMAP; sizes and gaps in a map are not measurements

<!--
5 min. Read the bullets, do not expand them. This slide is a checklist for the
room to audit itself against — anyone who cannot expand a bullet knows what to
revise tonight.

Ask for a show of hands on which bullet is least familiar, and spend two minutes
on whatever wins. Do not spend six.
-->

---

# L05–L08 in one slide

- **L05** — rank vs **participation ratio**; a spectrum with no elbow; z-score or not; a linear readout is a hyperplane; majority-class baseline; XOR is not linearly separable
- **L06** — **flexible** (shattering) vs **abstract** (CCGP); nonlinear mixed selectivity raises the rank; random labels measure capacity; RSA vs CKA
- **L07** — above chance? (permutation) · would it hold? (cross-validation, leakage, bootstrap) · how high could it be? (noise ceiling) · change the measure, the winner changes
- **L08** — $a = g(\mathbf{w}^\top\mathbf{x}+b)$; weights as boundary and as template; perceptron, delta rule, logistic regression: **η × error × input**; gradient descent

<!--
5 min. Same treatment. The two most misunderstood items on this slide are
"PR counts how evenly the variance is spread, not how much information there
is" and "high shattering and high CCGP pull against each other" — flag both
explicitly, because both are natural exam questions and both are usually
answered wrong.
-->

---

# From Lecture 9: backpropagation in four steps

1. **Forward** — run the input through; keep every $z$ and every $a$
2. **Output error** — $\delta_{\text{out}} = \hat y - y$
3. **Backward** — $\delta_j = g'(z_j)\sum_k w_{kj}\,\delta_k$; a ReLU that was off passes nothing
4. **Gradients** — $\partial\ell/\partial w = \delta \times \text{input}$

<mark>This is the only part of Lecture 9 on Exam 1. The 2–2–1 XOR example from the lecture is the model to practise on.</mark>

<!--
4 min. Say the scope again: the four steps, on a network small enough to do by
hand. Not the softmax, not the training loop, not autograd.

Remind them that with a sigmoid output and the cross-entropy, the output error
is simply prediction minus target: the sigmoid's slope cancels the loss's
denominator, as in logistic regression. Problem 10 is this calculation on a new
input.
-->

---

# PROBLEM 1

A unit's activity across three stimuli is $(2, 4, 6)$. Another's is $(1, 2, 3)$.

1. Cosine distance?
2. Correlation distance?
3. What does the pair of answers tell you?

<!--
4 min. Answers: both are 0. Cosine because the vectors point in exactly the same
direction. Correlation because after centering each row both become proportional
to (-1, 0, 1).

Part 3 is the real question: these units differ only in gain. Cosine discards
gain, correlation discards gain and offset, so neither can tell them apart;
Euclidean keeps both and can. Which one you want is a modelling decision:
choose by what should not count.

Follow-up if the room is fast: what if the second unit were (3, 2, 1)?
-->

---

# PROBLEM 2

$d(A,B) = 0.2$, $d(B,C) = 0.2$, $d(A,C) = 0.9$.

1. Which requirement of a distance fails?
2. Name a measure from this course that permits it.

<!--
4 min. Answers: the triangle inequality, since 0.9 > 0.4. Cosine distance and
correlation distance both permit it; they also fail identity (a brighter copy
of an image sits at cosine distance 0 from it). Euclidean distance satisfies
all four requirements.

Push: is that a defect? Only if you needed the requirement. MDS assumes the
matrix is close to a table of distances between points, which is why it
matters which measure built the RDM.
-->

---

# PROBLEM 3

Someone reports that their MDS map's horizontal axis is **animacy**.

1. What is the methodological objection?
2. What evidence would answer it?

<!--
4 min. Answers: (1) the MDS solution is not unique — shift, rotate or reflect
it and every distance, and the stress, is unchanged — so "the horizontal axis"
is not a property of the data. Only relative positions mean anything. (2)
Independent evidence: correlate a direction in the map with an external
animacy rating, or train a readout for animacy and test it on held-out stimuli.

The lecture's line is worth quoting: if a story about your map depends on which
way is "up", it is a story about the plot.
-->

---

# PROBLEM 4

Four units. The eigenvalues of their covariance are $(4, 4, 0, 0)$.

1. Rank? Participation ratio?
2. Same questions for $(7, 1, 0, 0)$.
3. What does the difference tell you?

<!--
5 min. Answers: (1) rank 2; PR = (4+4)^2 / (16+16) = 64/32 = 2. (2) rank 2;
PR = 8^2 / (49+1) = 64/50 = 1.28.

(3) Both populations use two directions, but in the second one direction
carries almost all the variance. PR counts how evenly the variance is spread:
if k directions share it equally, PR = k. It does not measure how much
information the population carries.

Follow-up: why not just report the rank of a real recording? Because noise
makes every real recording full rank (480 IT neurons: rank 480).
-->

---

# PROBLEM 5

Two $t$-SNE runs on the same responses, differing only in `random_state`. One
shows two tight islands far apart; the other, three looser groups.

1. Is either wrong?
2. What can you claim, and how would you check it?

<!--
5 min. Answers: (1) neither; the random start is set by the seed, and another
seed gives another map. (2) Claim only what holds across seeds and across a
perplexity sweep. Sizes and gaps in a t-SNE map are not distances in the data,
so "far apart" and "tight" are not findings.

To check a grouping, go back to the response space: are an image's nearest
neighbors there the same images? Does a readout trained on some images separate
the groups on others?

Land the lecture's distinction: a map can be stable (same neighbors run to run)
and still unfaithful (different neighbors from the original space).
-->

---

# PROBLEM 6

Four stimuli sit at the corners of a square, in two units.

1. Of the 16 ways to split them in two, how many can a linear readout separate?
2. Which fail?
3. What kind of third unit fixes it?

<!--
5 min. Answers: (1) 14. (2) The two XOR splits, which put opposite corners
together. (3) A unit whose responses are NOT a weighted sum of the other two:
the product x1 x2, or an AND unit. It raises the rank to 3 and all 16 splits
become separable. Adding x1 + x2 does nothing: the rank stays 2.

Connect it to the recordings: this is why nonlinear mixed selectivity makes a
population high-dimensional, and why a high shattering dimensionality means
almost any grouping is readable.
-->

---

# PROBLEM 7

A readout on **480 IT neurons**, trained on **102 images** with **shuffled**
labels, scores **1.00** on its training images.

1. How is that possible?
2. What number do you report instead, and against what?

<!--
5 min. Answers: (1) more neurons than training images: in that many dimensions
the points are nearly perpendicular and any labelling can be fit. Getting the
training images right is memorizing them. (2) Accuracy on held-out test images,
read against a baseline: the shuffled-label score on test images (0.56 in the
lecture, chance) and the majority-class baseline.

The same holds for deep networks, which memorize ImageNet with shuffled labels
(Zhang et al. 2017).
-->

---

# PROBLEM 8

Model A scores **0.30** against IT; its noise ceiling is **0.35**.
Model B scores **0.45**; its ceiling is **0.90**.

1. Which is closer to its ceiling?
2. What two other references does the lecture ask for?

<!--
5 min. Answers: (1) A, at 0.30 / 0.35 ≈ 0.86 of its ceiling; B is at 0.50. A
raw score cannot be read without its ceiling. (2) The shuffled floor (a
permutation test: shuffle one system's image order 1,000 times and beat the
95th percentile) and the same area in another brain.

Then ask: how is the ceiling obtained? Split-half reliability: how well two
halves of the repeats agree with each other.
-->

---

# PROBLEM 9

Delta rule. $\mathbf{w} = (0.5, -1)$, $b = 0$, $\mathbf{x} = (2, 1)$, $y = 1$, $\eta = 0.1$.

1. $\hat y$? The update $\Delta\mathbf{w}$?
2. $\hat y$ after the update?

<!--
5 min. Answers: (1) ŷ = 0.5·2 − 1·1 = 0. Error y − ŷ = 1. Δw = η (y − ŷ) x =
0.1 · 1 · (2, 1) = (0.2, 0.1), so w becomes (0.7, −0.9). (2) ŷ = 1.4 − 0.9 =
0.5: halfway to the target, and the squared-error loss falls from 0.5 to 0.125.

Then the conceptual half: the perceptron uses the same update with a
thresholded output, so it changes w only on mistakes; logistic regression uses
it with σ(z). One recipe: learning rate × error × input.
-->

---

# PROBLEM 10

The lecture's XOR network, same starting weights: hidden $\mathbf{w}_1 = (0.50, -0.40)$,
$\mathbf{w}_2 = (0.30, 0.80)$, output weights $(0.70, -0.60)$, biases 0, ReLU hidden
units, sigmoid output, cross-entropy.

Input $(1, 1)$, target $0$ (XOR of 1 and 1). Run the four steps.

<!--
7 min. Answers, two decimals:
Step 1. z1 = 0.50 − 0.40 = 0.10, so a1 = 0.10. z2 = 0.30 + 0.80 = 1.10, so a2 = 1.10.
Output z = 0.70·0.10 − 0.60·1.10 = −0.59, ŷ = σ(−0.59) = 0.36. Loss −log(1 − 0.36) = 0.44.
Step 2. δ_out = ŷ − y = 0.36: the output was too high.
Step 3. δ1 = 0.36 × 0.70 × 1 = 0.25 (unit 1 excites the output: decrease it).
δ2 = 0.36 × (−0.60) × 1 = −0.21 (unit 2 inhibits the output: increase it).
Step 4. Output weights: 0.36 × 0.10 = 0.04 and 0.36 × 1.10 = 0.39. Unit 1's
weights: 0.25 × (1, 1) = (0.25, 0.25). Unit 2's weights: −0.21 × (1, 1) = (−0.21, −0.21).
Biases: 0.36 (output), 0.25 and −0.21 (hidden).

The point to land: each hidden error is the output error sent back through that unit's
outgoing weight, so its sign follows the sign of the weight. One step with η = 1 lowers ŷ to 0.11.
-->

---

# Six things to be able to state cold

1. the three dissimilarity measures, what each **ignores**, and which are metrics
2. what an eigenvector of the covariance is — in a sentence, no algebra
3. ambient, embedding and intrinsic dimension, and what a $t$-SNE map **cannot** show
4. what shattering measures and what CCGP measures
5. the three questions before trusting a score: chance, held-out, ceiling
6. one update for three rules, and the **four steps** of backpropagation

<!--
4 min. Read them out. This is the "night before" list and it is in the handout.

Each of these has appeared in a lecture and an assignment or a recitation —
which is the reason they are on the list, and worth saying, because it tells
them the list is not arbitrary.
-->

---

# Open questions

Bring what you are stuck on.

<!--
7 min, and protect it. This is what the people who came actually came for.

If nobody speaks, prime the room with a question of your own: "who can tell me
why a readout needs held-out images?" — then let the discussion go where it
wants.

Close by saying that Assignment 3 is released Thu 10/22, the day of the exam,
and is due Thu 11/5; and that recitation 8 covers comparing representations —
RSA, decoding, and permutation tests.
-->
