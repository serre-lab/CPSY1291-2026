---
marp: true
theme: cpsy1291-recitation
paginate: true
math: katex
---

<!-- ===========================================================================
  CPSY 1291 — RECITATION 12: Final exam review (Lectures 9-19)
  TA-led, 80 minutes. Optional. Week 12.

  *** SCHEDULING: this session MUST be held before Thanksgiving recess
  (Wed 11/25). The final exam is Tue 12/1, the first day back. There is no
  session between recess and the exam. Announce this in Recitation 11 and
  again at the top of this one — students will otherwise assume the usual
  rhythm and miss it. ***

  FORMAT: a problem session, like Recitation 7. Slides are prompts. Put a
  problem up, give the room 3-4 minutes, take answers before revealing.

  BUDGET: ~16 min on the two summary slides, ~55 min on problems, 7 min open.
  Protect the open block.

  DO NOT: predict what is on the exam. Point at the syllabus objectives.

  MATERIAL: handout-12-final-review.pdf — the full compressed summary of
  L09-L19 and sixteen problems with worked answers.
============================================================================ -->

<div class="eyebrow">Recitation 12</div>

# Final exam review

## Lectures 9–19, as problems

<!--
2 min. FIRST WORDS: the exam is Tuesday 12/1, the first day back from recess.
This session is before recess because there is no session after it. Say it
twice.

Then the format: problem session, like the midterm review. Fifteen minutes of
summary, an hour of problems, ten minutes of open questions.

If asked what is on the exam: you do not know. Point at the syllabus objectives.
-->

---

# The shape of the second half

Lectures 1–8 built **machinery**.

Lectures 9–19 spend it on one repeated question:

<mark>Does this model tell us anything about the brain — and how would we know?</mark>

Almost every lecture presents a method for answering that, then presents a
reason the method is not enough. **Both halves are examinable, and the second
half is where the marks are.**

<!--
4 min. This is the most useful framing you can give, and it is genuinely how
the second half is organized.

Give one example: L12 gives RSA and model-brain scores, then shows the metrics
disagree, that scores can be engineered, and that at scale everything scores
alike. A student who revised only the method has revised half the lecture.
-->

---

# L09–L13 in one slide

- **L09** — three splits; the classical U; **random labels**; double descent; grokking; compositional generalization; **shortcut learning**
- **L10** — linear tied autoencoder → PCA *subspace*; sparse coding → Gabors → V1; efficient coding as a **normative** explanation; SAEs
- **L11** — ventral hierarchy; Gabors in V1; simple/complex → conv/pool; weight sharing; CNNs predict IT — and show **texture bias**
- **L12** — RDMs, RSA, noise ceiling, permutation; the mapping hypothesis; the metrics **disagree**; decoding ≠ used
- **L13** — attribution vs synthesis; saliency is fragile; polysemantic units; meaning lives in **directions**

<!--
5 min. A checklist for the room to audit itself against. Read the bullets; do
not expand them.

Ask which bullet is least familiar and spend two minutes on the winner. The
usual answers are "normative explanation" and "the metrics disagree".
-->

---

# L14–L19 in one slide

- **L14** — state carried forward; unrolling; BPTT as a **product** of Jacobians; clipping vs gating
- **L15** — fixed points; point vs **line attractors**; Hopfield energy, $0.14n$ capacity; recurrence solves what feedforward cannot
- **L16** — $\mathrm{softmax}(QK^\top/\sqrt{d_k})V$; every position sees every other; multi-head is a reshape; maps show **routing**, not importance
- **L17** — teacher forcing; sampling knobs; scaling laws and Chinchilla; contrastive vs masked self-supervision
- **L18** — reparameterization + ELBO; GAN minimax; diffusion is closed-form noising + an MSE
- **L19** — world models; three limits; **does better AI mean better models of biology?**

<!--
5 min. Same treatment.

Flag two items as commonly-missed: "the training objective of diffusion is a
plain MSE" and "attention maps show routing, not importance". Both are natural
exam questions and both are usually answered from vibes.
-->

---

# PROBLEM 1

A network reaches **100% training accuracy on randomly shuffled labels**.

Why does this undermine capacity-based explanations of generalization?

<!--
5 min. Answer: if the network can fit arbitrary labels, its capacity is
sufficient to memorize the training set outright — so capacity cannot be what
stops it memorizing when the labels are real.

Whatever explains generalization must involve the data, the optimizer, or the
architecture's inductive bias. Not the parameter count.

Push: does this mean capacity is irrelevant? No — it means capacity alone is not
the explanation. Double descent is the constructive follow-up.
-->

---

# PROBLEM 2

Sparse coding on natural image patches produces **Gabor-like filters** that match
V1 simple cells.

Why is that evidence about V1, rather than a coincidence?

<!--
5 min. Answer: the filters were not fitted to neural data. They fall out of an
objective — reconstruct natural images with few active units — applied to natural
images alone, and they then match measured receptive fields quantitatively.

That makes it a NORMATIVE explanation: the biology is what you would expect if
V1 were solving that problem. Contrast with a model fitted to neural responses,
which can match without explaining anything.

This is the cleanest example of normative explanation in the course. Make sure
they can say the words.
-->

---

# PROBLEM 3

A classifier reaches **94%** on held-out test data and **41%** on photographs from
a different hospital.

1. Name the phenomenon.
2. Name one method from L13 that could have caught it in advance.

<!--
5 min. Answers: (1) shortcut learning — the model used a feature correlated with
the label in the training hospital (scanner, marking, preprocessing) rather than
the pathology. (2) Attribution: a saliency or Grad-CAM map showing the model
attending to a region that cannot carry the diagnosis.

Note the honest caveat: attribution CAN catch this and is not guaranteed to. It
is the clearest scientific use of the method and still not a proof.
-->

---

# PROBLEM 4

A model scores **0.42** by RSA against IT.

1. What two numbers do you need before this means anything?
2. How is each obtained?

<!--
5 min. Answers: (1) the noise ceiling and a null distribution. (2) The ceiling
from a split-half correlation of the neural RDMs; the null from permuting
stimulus LABELS and reindexing rows and columns together.

Follow-up worth asking: why not permute the entries of the upper triangle? Because
those entries are not exchangeable — moving one stimulus moves a whole row and
column. Permuting entries gives a null far too narrow, and everything comes out
significant.
-->

---

# PROBLEM 5

You randomize a network's weights and its **saliency map is unchanged**.

What follows?

<!--
5 min. Answer: the map is determined by the input image and the architecture,
not by anything the model learned. Any claim it supported about the trained
model's strategy is unsupported — the figure is evidence about edges.

Ask: is the method therefore worthless? No — it is worthless FOR THAT CLAIM. The
general lesson is that a method needs a control that could have failed, and
weight randomization is that control.
-->

---

# PROBLEM 6

An RNN trains on sequences of length 20 and fails at length 300.

1. Name the mechanism.
2. Why does gradient clipping not fix it?

<!--
5 min. Answers: (1) vanishing gradients — BPTT over 300 steps multiplies 300
Jacobians and the product decays exponentially. (2) Clipping bounds gradients
from ABOVE; it does nothing for gradients that are too small.

The fix is gating, which routes the cell state through an addition rather than a
repeated multiplication, giving the gradient a path back that is not a long
product.
-->

---

# PROBLEM 7

A trained RNN has a fixed point with one Jacobian eigenvalue at **1.00** and the
rest near **0.3**.

1. What computation does this support?
2. Give a task that would need it.

<!--
5 min. Answers: (1) a line attractor — a direction that neither decays nor grows,
with everything else contracting onto it. (2) Integrating evidence in a
perceptual decision, or holding a continuous value: eye position, a remembered
angle, a running total.

Contrast with all eigenvalues below 1, which is a point attractor — a discrete
memory or a committed decision.
-->

---

# PROBLEM 8

1. Why is the $\sqrt{d_k}$ in $\mathrm{softmax}(QK^\top/\sqrt{d_k})V$ necessary?
2. What exactly fails without it?

<!--
5 min. Answers: (1) dot products of d_k-dimensional vectors grow like sqrt(d_k),
so without scaling the softmax argument becomes large. (2) The softmax saturates
to nearly one-hot and its gradient nearly vanishes, so training fails for large
d_k while appearing fine for small d_k.

Connect to L09's dead ReLUs and L14's vanishing gradients: three lectures, one
phenomenon — a saturated nonlinearity has no gradient.
-->

---

# PROBLEM 9

Which is false, and why?

1. Diffusion training minimizes a mean squared error between predicted and actual noise.
2. Diffusion is complicated because its loss is complicated.
3. A VAE's KL term going to zero means training is going well.

<!--
5 min. Answers: 1 is TRUE. 2 is FALSE — the loss is an MSE; the complexity is in
the sampling loop and the denoiser architecture. 3 is FALSE — it is posterior
collapse: the encoder matched the prior, the latent carries no information, and
the decoder reconstructs on its own. Samples can still look fine.

Statement 3 is the one that catches people, because "the loss term went down"
sounds like good news everywhere else.
-->

---

# PROBLEM 10

State the closing question of Lecture 19, and the evidence that makes it worrying.

<!--
6 min. Answer: does better AI mean better models of biology? The worry: as
benchmark performance has risen, brain-predictivity has not risen with it, and
at scale many quite different models score alike — so the standard scores no
longer discriminate between them.

The proposed response is to compare STRATEGIES rather than outputs: ClickMe,
harmonization, rethinking data diets, better objectives.

This is the course's closing argument, and it is very likely examinable in some
form. Let the room discuss it rather than dictating an answer.
-->

---

# Eight things to state cold

1. the three splits, and what **random labels** rules out
2. why sparse coding → V1 is a **normative** explanation
3. selectivity from convolution, tolerance from pooling — and each cortical analogue
4. the noise ceiling and the stimulus permutation, and what each is for
5. why a saliency map surviving weight randomization is worthless
6. vanishing gradients as a **product**, and why gating not clipping
7. what a **line attractor** is, and one task that needs one
8. what an attention map does and does **not** show

<!--
4 min. Read them out. This is the "night before" list from the handout.

Each has appeared in a lecture, a recitation, and either an assignment or a
paper discussed in class — which is why it is on the list, and worth saying so
the list does not look arbitrary.
-->

---

# Open questions

Bring what you are stuck on.

**The exam is Tue 12/1** — the first day back.

<!--
7 min, and protect it.

Close by repeating the date one more time, and by saying what happens after:
project work starts Thu 12/3, and the last two recitations are project clinics —
scoping and data first, then figures and the talk.
-->
