---
marp: true
theme: cpsy1291-recitation
paginate: true
math: katex
---

<!-- ===========================================================================
  CPSY 1291 — RECITATION 12: Exam 2 review (Lectures 9-19)
  TA-led, 80 minutes. Optional. Week 12.

  *** SCHEDULING: this session MUST be held before Thanksgiving recess
  (Wed 11/25). Exam 2 is Tue 12/1, the first day back. There is no
  session between recess and the exam. Announce this in Recitation 11 and
  again at the top of this one — students will otherwise assume the usual
  rhythm and miss it. ***

  SCOPE: Exam 2 focuses on L09-L19. Earlier material appears only as tools.

  FORMAT: a problem session, like Recitation 7. Slides are prompts. Put a
  problem up, give the room 3-4 minutes, take answers before revealing.

  BUDGET: ~16 min on the framing and the two summary slides, ~50 min on
  problems, 4 min on the cold list, 7 min open. Protect the open block.

  DO NOT: predict what is on the exam. Point at the syllabus objectives.

  MATERIAL: handout-12-final-review.pdf — the full compressed summary of
  L09-L19 and seventeen problems with worked answers.
============================================================================ -->

<div class="eyebrow">Recitation 12</div>

# Exam 2 review

## Lectures 9–19, as problems

<!--
2 min. FIRST WORDS: Exam 2 is Tuesday 12/1, the first day back from recess.
This session is before recess because there is no session after it. Say it
twice.

Then the format: problem session, like the Exam 1 review. Fifteen minutes of
summary, fifty minutes of problems, then open questions.

If asked what is on the exam: you do not know. Point at the syllabus objectives.
-->

---

# The shape of the second half

Lectures 1–8 built **machinery** for describing a representation.

Lectures 9–19 learn representations, and keep asking:

<mark>Does this model tell us anything about the brain — and how would we know?</mark>

Almost every lecture presents a method for answering that, then presents a
reason the method is not enough. **Both halves are examinable, and the second
half is where the marks are.**

<!--
4 min. This is the most useful framing you can give, and it is genuinely how
the second half is organized.

Give one example: L12 gives linear predictivity and RSA, then shows that past a
point recognition accuracy stops helping, that many very different networks tie,
and that the scores disagree with each other. A student who revised only the
method has revised half the lecture.
-->

---

# L09–L14 in one slide

- **L09** — hidden layers re-represent; linear layers collapse into one; backprop's four steps; softmax; the training loop; autograd; transfer
- **L10** — minibatches, schedules, Adam; tuning curves and ablation; three splits; bias–variance; regularization; **shortcuts**; **random labels**; double descent
- **L11** — tied linear autoencoder → PCA *subspace*; sparse coding → V1-like filters; simple → complex cells; convolution and pooling; ResNets, BatchNorm; **texture bias**
- **L12** — linear predictivity over each site's reliability; RSA vs predictivity; many networks tie; the scores **disagree**; reverse predictivity and control; decoding
- **L13** — saliency, Grad-CAM, occlusion; interventions test the map, **sanity checks** test the method
- **L14** — ELBO = reconstruction − KL; reparameterization; GAN minimax; diffusion predicts the **noise**

<!--
5 min. A checklist for the room to audit itself against. Read the bullets; do
not expand them.

Ask which bullet is least familiar and spend two minutes on the winner. The
usual answers are "reverse predictivity" and "sanity checks test the method".
-->

---

# L15–L19 in one slide

- **L15** — a state with shared weights; unrolling; BPTT; leak; evidence to a bound; vanishing gradients as a **product**; clipping vs gating
- **L16** — fixed points; point vs **line attractors**; Jacobian eigenvalues; Hopfield energy, about $0.138D$ memories; modern Hopfield = one attention step
- **L17** — $\mathrm{softmax}(QK^\top/\sqrt{d_k})V$; multi-head; causal mask; next-symbol loss against a bigram baseline; a map shows **where** a head reads, not why
- **L18** — next token + transformer + scale; scaling laws; in-context learning; emergence and the ruler; surprisal; contrastive vs masked SSL; predictive coding
- **L19** — world models; three limits; **better AI is not a better brain model**

<!--
5 min. Same treatment.

Flag two items as commonly missed: "the training objective of diffusion is a
plain squared error on the noise" and "an attention map shows where a head
reads, not why". Both are natural exam questions and both are usually answered
from vibes.
-->

---

# PROBLEM 1

A network reaches **100% training accuracy on randomly shuffled labels**.

Why does this undermine capacity-based explanations of generalization?

<!--
5 min. Answer: if the network can fit arbitrary labels, its capacity is
sufficient to memorize the training set outright — so capacity cannot be what
stops it memorizing when the labels are real. The same network memorizes or
generalizes, depending on the labels.

Whatever explains generalization must involve the data, the optimizer, or the
architecture. Not the parameter count alone.

Push: does this mean capacity is irrelevant? No — it means capacity alone is not
the explanation. Double descent is the constructive follow-up: which fit
training finds affects the test error.
-->

---

# PROBLEM 2

Sparse coding on whitened natural image patches produces **Gabor-like filters**.

1. Why is that evidence about V1, rather than a coincidence?
2. Where does the match fail?

<!--
5 min. Answers: (1) the filters were not fitted to neural data, and nothing
about edges was built in. They fall out of an objective — describe each patch
with a few atoms — applied to natural images alone, and they come out
localized, oriented and bandpass, like V1 simple cells.

(2) Measured on Ringach's plot: the width matches (median 0.33 for atoms, 0.32
for cells), the length does not (0.87 against 0.41) — the atoms are longer along
their stripes than most real cells. The objective predicts V1 to first order;
measuring shows where it falls short.
-->

---

# PROBLEM 3

A classifier reaches **94%** on held-out test data and **41%** on X-rays from
a different hospital.

1. Name the phenomenon.
2. How would you catch it with the tools of L13?

<!--
5 min. Answers: (1) shortcut learning — a cue that predicts the label in the
training images but not in the world, such as a hospital's marker in the corner
(DeGrave et al. 2021). (2) An attribution map suggests the cue (the map lights up
the marker, not the lungs); an intervention confirms it (edit that region and
watch the decision), with a control (the same edit to a same-size region
elsewhere).

Map, then intervention, then control. A map alone is a hypothesis.
-->

---

# PROBLEM 4

A ResNet stage predicts an IT site with held-out $R^2 = 0.40$. The site's
split-half reliability is $0.80$.

1. What score does the lecture report, and what would 1 mean?
2. Why report RSA as well?

<!--
5 min. Answers: (1) 0.40 / 0.80 = 0.50. A score of 1 means the model predicts
the site as well as the site predicts itself: the noise ceiling, one per site.
Linear predictivity is the median over sites.

(2) They assume different things. Predictivity fits a weight per unit, so
reweighting units barely changes it; RSA fits nothing and compares geometry as
it is. They can disagree (best stage: block group 3 for predictivity, 4 for
RSA), so report both, each against its own ceiling.
-->

---

# PROBLEM 5

You randomize a network's weights, from the top layer down, and its
attribution map **barely changes**.

What follows?

<!--
5 min. Answer: the map cannot be explaining what the network LEARNED; it is
driven by the image and the architecture. In the lecture, gradient and Grad-CAM
change and pass; guided backpropagation and guided Grad-CAM still show the bird
and fail.

Ask: is the method therefore worthless? It is worthless FOR THAT CLAIM. The
general lesson: a method needs a check that could have failed, and weight
randomization is that check. Sanity checks test the method; interventions test
the map.
-->

---

# PROBLEM 6

An RNN trains on sequences of length 20 and fails at length 300.

1. Name the mechanism.
2. Why does gradient clipping not fix it?

<!--
5 min. Answers: (1) vanishing gradients — BPTT sends the error back through a
product of Jacobians, diag(g′) W_hh, one per step; with spectral radius below 1
the long product shrinks, and tanh′ ≤ 1 makes it worse. (2) Clipping rescales
gradients that are too LARGE; it cannot create gradient.

The fix is gating: the LSTM's cell is updated by addition, so with the forget
gate near 1 the error flows back many steps without shrinking.
-->

---

# PROBLEM 7

A trained RNN has a fixed point with one Jacobian eigenvalue at **1.00** and the
rest near **0.3**.

1. What computation does this support?
2. Give a task that would need it.

<!--
5 min. Answers: (1) a line attractor — a slow direction along which an offset
neither decays nor grows, with everything else contracting onto it. With no
input the state stays put; input moves it along the line, so its position is
the sum of the input so far. (2) Accumulating evidence for a decision (Mante et
al.'s trained network), or holding a graded value in working memory: eye
position, head direction on a ring.

Contrast with all |λ| below 1, a point attractor — a stored memory or a
committed choice. Some above 1 and some below: a saddle.
-->

---

# PROBLEM 8

1. Why is the $\sqrt{d_k}$ in $\mathrm{softmax}(QK^\top/\sqrt{d_k})V$ necessary?
2. What exactly fails without it?

<!--
5 min. Answers: (1) a dot product of two random d_k-dimensional vectors has a
spread that grows like sqrt(d_k), so without scaling the softmax argument
becomes large. (2) The softmax puts almost all weight on one position and its
gradient becomes close to zero, so training fails for large d_k while appearing
fine for small d_k.

Connect it to L10's dead units and L15's vanishing gradients: three lectures,
one phenomenon — a unit or a nonlinearity with no slope passes no gradient.
-->

---

# PROBLEM 9

Which is false, and why?

1. Diffusion training minimizes a squared error between the predicted and the added noise.
2. Diffusion is slow to sample because it removes all the noise in one step.
3. In a VAE, the KL term is what makes reconstructions accurate.

<!--
5 min. Answers: 1 is TRUE — no adversary, no KL to balance. 2 is FALSE — it is
slow because it takes many small denoising steps, one network pass each; one big
step is what a VAE or GAN decoder attempts. 3 is FALSE — the reconstruction term
does that; the KL keeps each input's code distribution close to the prior, which
fills the holes in the latent space so that samples decode to sensible images.

Statement 3 is the one that catches people. Ask the room what the
reparameterization trick is for: it moves the randomness outside the network, so
gradients reach the encoder.
-->

---

# PROBLEM 10

State the closing question of Lecture 19, and the evidence that makes it worrying.

<!--
6 min. Answer: does better AI mean better models of biology? The worry: past a
point, more accurate networks do not predict IT better (Brain-Score), and
alignment with brains and with human strategies can FALL as accuracy rises.

The proposed responses: compare strategies, not just outputs (ClickMe); align
training to human strategy (harmonization); rethink the data diet; bring back
recurrence and feedback.

This is the course's closing argument. Let the room discuss it rather than
dictating an answer.
-->

---

# Eight things to state cold

1. the four steps of backprop, and why linear layers collapse into one
2. the three splits, and what **random labels** rule out
3. what a tied linear autoencoder finds, and what sparse coding of natural images finds
4. how a predictivity score is normalized, and why one score is not enough
5. map, intervention, control — and what a **sanity check** tests
6. the ELBO's two terms, and what a diffusion network predicts
7. vanishing gradients as a **product**, why gating not clipping, and a **line attractor**
8. the attention equation, the causal mask, and what an attention map does **not** show

<!--
4 min. Read them out. This is the "night before" list from the handout.

Each has appeared in a lecture and in an assignment or a recitation — which is
why it is on the list, and worth saying so the list does not look arbitrary.
-->

---

# Open questions

Bring what you are stuck on.

**Exam 2 is Tue 12/1** — the first day back.

<!--
7 min, and protect it.

Close by repeating the date one more time, and by saying what happens after:
project work starts Thu 12/3, and the last two recitations are project clinics —
scoping and data first, then figures and the poster.
-->
