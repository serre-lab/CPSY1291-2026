---
marp: true
theme: cpsy1291-recitation
paginate: true
math: katex
---

<!-- ===========================================================================
  CPSY 1291 — RECITATION 11: Attention and the transformer
  TA-led, 80 minutes. Optional. Week 11 — hold this after Lecture 17.
  Assignment 4 is due Tue 11/24. This is the LAST content session of the term.

  SCOPE: Assignment 4 covers Lectures 15-17. Recitation 10 did recurrence and
  dynamics (L15-L16); this session finishes the assignment's scope with
  Lecture 17: the attention head, multi-head attention, the causal mask, the
  next-character loss, the bigram baseline, generation, and the bridge from
  the Hopfield network to attention. Generative models (L14) and LLMs (L18)
  are not on A4; they are on Exam 2, and next week's review covers them.

  BUDGET: attention ~39 min, training and generation ~18 min, Hopfield to
  attention 7 min, exercise 8 min, close 3 min. Attention is the part
  everything else depends on — protect it.

  IF YOU RUN LONG: shorten the bigram slide and point at the handout. Do NOT
  cut the three-details slide on attention (sqrt(d_k), dim=-1, -inf masking),
  which is where the assignment's bugs will come from.

  MATERIAL: handout-11-attention-transformers-generative.pdf, seven exercises.
============================================================================ -->

<div class="eyebrow">Recitation 11</div>

# Attention and the transformer

## What Assignment 4 needs from Lecture 17

<!--
2 min. Say the scope straight away: this is the last part of Assignment 4,
which covers Lectures 15-17 and is due next Tuesday. Recitation 10 did
recurrence and dynamics; today is attention.

Say also what is NOT here: generative models and LLMs. Both are on Exam 2, and
next week's review session covers them.
-->

---

# The one idea

<mark>Compute a set of weights saying how much each position contributes. Take a weighted average of what the positions offer.</mark>

- **attention** — the weights come from how well a query matches each key
- **a modern Hopfield update** — the same step, with the cue as the query and the stored patterns as keys and values

The differences are in *what* is averaged and *what sets* the weights.

<!--
5 min. This framing ties the two halves of Assignment 4 together: the Hopfield
network from last week and the attention head from this one are one operation.
The end of the session makes that exact.

Do not oversell it before the equations are up. Say it, then move on.
-->

---

# Tokens are a matrix

$T$ tokens, each a $d$-dimensional vector → a matrix $\boldsymbol{X}$ of shape $(T, d)$.

<mark>Notice what has been given up: a matrix's rows have no order unless you put it there.</mark>

That is why positional embeddings exist. An RNN got order for free and paid for
it with a sequential loop.

<!--
5 min. The trade is the point: transformers gave up built-in order to gain
parallelism, then bought order back with a learned table of positions added to
each symbol's vector.

Ask the room: what else did they gain? Every position can reach every other in
one step, so there is no long product of Jacobians — the vanishing-gradient
problem from last week does not arise. The price: T × T scores, so doubling the
sequence length quadruples the cost.
-->

---

# Q, K, V are three views of the *same* input

```python
Q = X @ Wq     # what each position is looking for
K = X @ Wk     # what each position offers
V = X @ Wv     # what each position would contribute
```

All three are linear projections of the same $\boldsymbol{X}$ — which is what
makes it ***self***-attention.

<!--
5 min. The retrieval metaphor: a query is matched against keys, and the matching
keys' VALUES are returned. Say it once and let the code carry the rest.

The word "self" is doing work. In cross-attention Q comes from one sequence and
K, V from another — same operation, different sources — which is how a decoder
looks at an encoder.
-->

---

# Attention, step by step

$$\mathrm{Attention}(Q,K,V)=\mathrm{softmax}\!\left(\frac{QK^\top}{\sqrt{d_k}}\right)V,\qquad \text{masked entries set to } -\infty \text{ before the softmax}$$


| step | shape | meaning |
|---|---|---|
| $QK^\top$ | $(T,T)$ | every position against every position |
| after softmax | $(T,T)$ | each **row** is a distribution |
| $AV$ | $(T,d_v)$ | each position: a weighted average of values |

<!--
6 min. Walk the shapes. The (T, T) matrix is the object everyone plots; the
final (T, d_v) is what the model actually passes on.

Have the room tell you what each shape means before you reveal the table. Once
they can narrate the three shapes they understand attention.
-->

---

# Three details that are not decoration

- **the $\sqrt{d_k}$** — dot products grow like $\sqrt{d_k}$; a softmax of large numbers saturates and has almost no gradient
- **`dim=-1`** — the softmax runs across **keys**. The wrong axis produces nonsense that runs perfectly
- **the mask is $-\infty$, before the softmax** — zeroing afterwards leaves rows that no longer sum to 1

<!--
8 min. The highest-value slide in the deck and the source of the bugs they will
actually hit. Spend the time.

The dim=-1 case is the nastiest: the matrix is (T, T), so BOTH axes are valid
and nothing errors. Normalizing down the columns gives weights that do not sum
to one, the model trains badly, and there is no diagnostic at all.
-->

---

# Multi-head: split, attend, concatenate

```python
Q = Q.view(N, T, H, d // H).transpose(1, 2)     # (N, H, T, d/H)
# ... attention exactly as before, batched over H ...
out = out.transpose(1, 2).reshape(N, T, d)      # concatenate back
```

Split $d$ into $H$ groups, run the same operation on each, concatenate; $\mathbf{W}_O$ then mixes the heads back to width $d$.

<mark>The `transpose(1, 2)` puts heads next to the batch axis, so the matmuls treat them as independent problems.</mark>

<!--
5 min. Demystify it: there is no new attention mathematics in multi-head
attention. Each head has its own three projections of width d/H, so the cost is
that of one wide head, and the model gets several independent weight patterns
per layer.

The transpose is the only subtle part and it is worth pointing at twice —
before and after.
-->

---

# Reading an attention map, carefully

- There are $H$ heads per layer and many layers. **"The" attention map is one choice among many.**
- A map shows **where** a head reads, not **why** the output is what it is.
- A map is a hypothesis. **Change the input, or switch the head off**, to test it.

<mark>The same discipline as the attribution maps of Recitation 9: map, then intervention.</mark>

<!--
6 min. Recitation 9's whole discipline applies again, and saying so explicitly
helps — this is the third time this term that a compelling picture has turned
out to answer a different question than the one being asked of it.

Single heads in trained language models do track relations people can name
(verb to object, pronoun to referent). That is a finding about where they read.
If a student is planning an attention-map figure for their project, this slide
is for them.
-->

---

# One forward pass, $T$ training examples

- The target at position $t$ is the symbol at $t+1$; the loss is the cross-entropy, averaged over positions
- The **causal mask** keeps position $t$ from reading the symbol it must predict
- So one pass over a sequence of length $T$ gives $T$ predictions, each with its own target

<mark>Feeding the true text as input, not the model's own guesses, is teacher forcing — Recitation 10's mismatch, at every position at once.</mark>

<!--
7 min. The mask matters because training is parallel: the whole sequence goes in
at once. Without it, position t could read the word it must predict, the loss
would fall to nearly zero, and generation would fail.

Ask: what does the first row of the masked attention matrix look like? It can
only read itself, so its single weight is 1.
-->

---

# The bigram baseline

Count, in the training text, how often each symbol follows each other one:

$$P(\text{next}=b \mid \text{current}=a) = \frac{\text{count}(a\to b)}{\text{count}(a)}$$

Score it with **the same cross-entropy, on the same held-out text**.

<mark>If the transformer's held-out loss is not lower than the bigram's, it has not used more than one symbol of context.</mark>

<!--
6 min. Quick check for the room, on the lecture's toy text "that hat ate the
tea": what is P(next = a | current = h)? h is followed by a twice and by e once,
so 2/3.

Then the point: a loss means something only next to a baseline computed on the
same data, the rule from the baselines lecture. The gap between the two losses is
what the longer context is worth.
-->

---

# Generation is a loop

```python
logits = model(idx)[:, -1, :] / temperature
probs = torch.softmax(logits, dim=-1)
nxt   = torch.multinomial(probs, 1)         # SAMPLE, not argmax
idx   = torch.cat([idx, nxt], dim=1)
```

| temperature | result |
|---|---|
| $\to 0$ | greedy: repetitive, often loops |
| $=1$ | samples the model's own distribution |
| $>1$ | flatter, more surprising, less coherent |

<!--
5 min. Land the consequence: two very different outputs from the SAME weights
differ only in this one number.

So any claim about "what the model does" has to state it. If the text grows
longer than the context, keep only the last T symbols before the next pass.
-->

---

# The Hopfield update is one attention step

$$\mathbf{s}\;\leftarrow\;\mathbf{X}^{\top}\,\mathrm{softmax}\big(\beta\,\mathbf{X}\,\mathbf{s}\big)
\qquad\qquad
\mathrm{softmax}\!\left(\frac{\mathbf{q}\,\mathbf{K}^{\top}}{\sqrt{d_k}}\right)\mathbf{V}$$

- the cue $\mathbf{s}$ is the query; the stored patterns $\mathbf{X}$ are both keys and values
- no learned projections: $\mathbf{W}_K=\mathbf{W}_V=\mathbf{I}$
- $\beta$ plays the role of $1/\sqrt{d_k}$

<!--
7 min. Read the left side in three steps: X s gives one match score per stored
pattern; the softmax turns the scores into weights; X transpose times the
weights is a weighted average of the stored patterns. The right side is the
same three steps with the cue renamed query.

This is the bridge in the second half of Assignment 4, and it closes the loop on
the first slide. Ask what a large beta does: it sharpens the weights, so recall
picks out one stored pattern.
-->

---

# EXERCISE

1. You delete the $\sqrt{d_k}$. Training works at $d_k=8$, fails at $d_k=512$. Why?
2. You mask by multiplying the attention weights by 0 **after** the softmax. What is wrong?
3. Your transformer's held-out loss equals the bigram table's. What has it learned?

<!--
8 min. Answers:

(1) Dot products of d_k-dimensional vectors grow like sqrt(d_k). At 512 the
unscaled scores are large, the softmax saturates to nearly one-hot, and its
gradient is nearly zero. At 8 the scores are small enough that it hardly matters.

(2) The softmax has already given weight to the masked positions, so the rows no
longer sum to one, and the unmasked positions get less than their share. Mask
with -inf before the softmax.

(3) Nothing that one symbol of context does not already give. Its attention is
not using the rest of the left context — check the mask, the positions, and
whether it trained at all.
-->

---

# Where this lands

- **Assignment 4** — the attention head, the causal mask, the next-character loss against a bigram baseline, generation, and Hopfield to attention
- **Exam 2 (Tue 12/1)** — L09–L19, and next week's session is the review
- **Your project** — attention maps are laptop-scale, and they need the same checks as any map

<mark>Next week: the Exam 2 review. Held before Thanksgiving, because the exam is the first day back.</mark>

<!--
3 min. Point at handout-11.pdf.

Say the week-12 timing loudly and more than once — the review session is BEFORE
recess and the exam is the Tuesday after. Anyone who assumes the usual rhythm
will miss it.
-->
