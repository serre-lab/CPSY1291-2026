---
marp: true
theme: cpsy1291
paginate: true
math: katex
---

<!-- ===========================================================================
  CPSY 1291 — RECITATION 11: Attention, transformers, and generative models
  TA-led, 80 minutes. Optional. Week 11 — hold this after Lecture 18.
  Assignment 5 is due Tue 11/24. This is the LAST content session of the term.

  WHY THREE LECTURES IN ONE SESSION: A5 covers L15-L18, it is due Tue 11/24,
  Thanksgiving recess starts the next day, and the final exam is the first day
  back. Week 12's session has to be the final review. There is no other slot.

  BUDGET: attention ~42 min, generation ~10 min, VAE + diffusion ~26 min. 5 min
  close. Attention is the part everything else depends on — protect it.

  IF YOU RUN LONG: cut the diffusion slides and point at the handout. Do NOT
  cut the three-details slide on attention (sqrt(d_k), dim=-1, -inf masking),
  which is where the assignment's bugs will come from.

  MATERIAL: handout-11-attention-transformers-generative.pdf — the longest of
  the term, seven exercises. Tell them to read it in three passes, not one.
============================================================================ -->

<div class="eyebrow">Recitation 11 · Week 11</div>

# Attention, transformers, generative models

## Three lectures of mechanics, one session

<!--
2 min. Be straight about why this session is dense: A5 covers L15-L18, it is due
next Tuesday, Thanksgiving follows immediately, and the final is the first day
back. Week 12 has to be the exam review.

Tell them the handout is the longest of the term and is meant to be read in
three passes rather than one.
-->

---

# The one idea — and it covers all three lectures

<mark>Compute a set of weights saying how much each thing contributes. Take a weighted average.</mark>

- **attention** — over positions in a sequence
- **a VAE** — over a distribution of latent codes
- **diffusion** — over time steps of a noising process

The differences are in *what* is averaged and *what sets* the weights.

<!--
5 min. This framing does real work — it lets them see three lectures as
variations rather than three separate things to memorize before the final.

Do not oversell it. The models differ in important ways. But the shape is
genuinely shared, and noticing that is worth five minutes.
-->

---

# Tokens are a matrix

$T$ tokens, each a $d$-dimensional vector → a matrix $\boldsymbol{X}$ of shape $(T, d)$.

<mark>Notice what has been given up: a matrix's rows have no order unless you put it there.</mark>

That is why positional encodings exist. An RNN got order for free and paid for
it with a sequential loop.

<!--
5 min. The trade is the point: transformers gave up built-in order to gain
parallelism, then bought order back with positional encodings.

Ask the room: what else did they gain? Every position can see every other in one
step, so there is no long product of Jacobians — the vanishing-gradient problem
from last week simply does not arise.
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

# Attention, in four lines

```python
scores = Q @ K.transpose(-2, -1) / math.sqrt(d_k)   # (T, T)
scores = scores.masked_fill(mask == 0, float('-inf'))
A = torch.softmax(scores, dim=-1)                   # rows sum to 1
out = A @ V                                         # (T, d_v)
```

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

# Multi-head is only a reshape

```python
Q = Q.view(N, T, H, d // H).transpose(1, 2)     # (N, H, T, d/H)
# ... attention exactly as before, batched over H ...
out = out.transpose(1, 2).reshape(N, T, d)      # concatenate back
```

Split $d$ into $H$ groups, run the same operation on each, concatenate.

<mark>The `transpose(1, 2)` puts heads next to the batch axis, so the matmuls treat them as independent problems.</mark>

<!--
5 min. Demystify it: there is no new mathematics in multi-head attention, only
bookkeeping that lets different heads attend to different things.

The transpose is the only subtle part and it is worth pointing at twice —
before and after.
-->

---

# Reading an attention map, carefully

- There are $H$ heads per layer and many layers. **"The" attention map is one choice among dozens, and they disagree.**
- Attention weight is **not** importance — a position contributes its *value* vector, which may be near zero.
- The **residual stream** carries information around the block entirely.

<mark>An attention map tells you where information was <em>routed</em>, not what the model <em>used</em>.</mark>

<!--
6 min. Recitation 9's whole discipline applies again, and saying so explicitly
helps — this is the third time this term that a compelling picture has turned
out to answer a different question than the one being asked of it.

All three objections have caused published overclaims. If a student is planning
an attention-map figure for their project, this slide is for them.
-->

---

# Generation is a loop

```python
logits = model(idx)[:, -1, :] / temperature
if top_k: logits[logits < topk_threshold] = float('-inf')
probs = torch.softmax(logits, dim=-1)
nxt   = torch.multinomial(probs, 1)         # SAMPLE, not argmax
idx   = torch.cat([idx, nxt], dim=1)
```

| temperature | result |
|---|---|
| $\to 0$ | greedy: repetitive, often degenerate |
| $=1$ | samples the model's own distribution |
| $>1$ | flatter, more surprising, less coherent |

<!--
5 min. Land the consequence: two very different outputs from the SAME weights
differ only in these two numbers.

So any claim about "what the model does" has to state them — and a surprising
fraction of published prompting results are partly results about a temperature
setting. That is worth their skepticism.
-->

---

# Surprisal — the measurement you can do on a laptop

```python
logp = torch.log_softmax(model(idx).logits, dim=-1)
surprisal = -logp[0, :-1].gather(1, idx[0, 1:, None]) / math.log(2)   # bits
```

The negative log probability the model gave to the token that actually occurred.

<mark>It correlates with human reading times — which is the whole basis of using language models as models of language processing.</mark>

<!--
5 min. This is the most project-friendly thing in the session: a real
psycholinguistic measurement, computable on a CPU, with a genuine literature
behind it.

Say that explicitly for anyone still choosing a final project topic.
-->

---

# VAE: the trick that makes sampling differentiable

You cannot backpropagate through "draw a sample". So rewrite the sample:

$$\boldsymbol{z} = \boldsymbol{\mu} + \boldsymbol{\sigma}\odot\boldsymbol{\varepsilon}, \qquad \boldsymbol{\varepsilon}\sim\mathcal{N}(0,\boldsymbol{I})$$

```python
mu, logvar = encoder(x).chunk(2, dim=1)
z = mu + torch.exp(0.5 * logvar) * torch.randn_like(mu)
```

<mark>The randomness now sits in $\boldsymbol{\varepsilon}$, which needs no gradient. $\boldsymbol{\mu}$ and $\boldsymbol{\sigma}$ are on the differentiable path.</mark>

<!--
6 min. The trick is genuinely clever and worth admiring for a moment.

Note why the network predicts LOGVAR rather than variance: a variance must be
positive, and exp enforces that for free, so the network's output is
unconstrained. Small design decisions like this are everywhere and are usually
unexplained in papers.
-->

---

# The two terms pull against each other

```python
recon = F.mse_loss(xhat, x, reduction='sum')
kl    = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
loss  = recon + beta * kl
```

- **reconstruction** wants codes that are distinct and informative
- **KL** wants them close to $\mathcal{N}(0, I)$, so the space is smooth and samplable

`beta` sets the exchange rate — and the character of the model follows from it.

<!--
5 min. Then give the failure mode with a name: if the KL dominates, the encoder
collapses to the prior, mu -> 0 and sigma -> 1, and the latent carries NOTHING.

Samples still look fine, because the decoder learned the data on its own. This
is posterior collapse. The diagnostic is one line: check whether the KL term
goes to zero.
-->

---

# Diffusion: one closed form, one MSE

$$\boldsymbol{x}_t = \sqrt{\bar\alpha_t}\,\boldsymbol{x}_0 + \sqrt{1-\bar\alpha_t}\,\boldsymbol{\varepsilon}$$

```python
t   = torch.randint(0, T, (x0.shape[0],))
eps = torch.randn_like(x0)
xt  = abar[t].sqrt()[:,None]*x0 + (1-abar[t]).sqrt()[:,None]*eps
loss = F.mse_loss(model(xt, t), eps)      # predict the noise you added
```

<mark>The training objective is a mean squared error. That is all it is.</mark>

<!--
6 min. The closed form is why training is cheap: pick a random t, jump straight
there, never simulate the forward process.

Then answer the obvious question — where did the complexity go? Into the
SAMPLING loop, which runs the denoiser backwards many times, and into the
denoiser's architecture. Not into the loss. That is also why diffusion models
are so much easier to train than GANs.
-->

---

# EXERCISE

1. You delete the $\sqrt{d_k}$. Training works at $d_k=8$, fails at $d_k=512$. Why?
2. Your VAE's KL drops to near zero in a few epochs and reconstructions look good. What happened?

<!--
6 min. Answers:

(1) Dot products of d_k-dimensional vectors grow like sqrt(d_k). At 512 the
unscaled scores are large, the softmax saturates to nearly one-hot, and its
gradient is nearly zero. At 8 the scores are small enough that it hardly matters.

(2) Posterior collapse. The encoder matched the prior, z carries no information,
and the decoder reconstructs on its own. Not useful as a REPRESENTATION even
though samples look fine. Lower beta, or weaken the decoder.
-->

---

# Where this lands

- **Assignment 5** — attention shapes, a generation loop, and one generative objective
- **The final (Tue 12/1)** — L09–L19, and next week's session is the review
- **Your project** — surprisal, attention maps, and latent spaces are all laptop-scale

<mark>Next week: the final exam review. Held before Thanksgiving, because the exam is the first day back.</mark>

<!--
3 min. Point at handout-11.pdf and repeat the three-passes advice.

Say the week-12 timing loudly and more than once — the review session is BEFORE
recess and the exam is the Tuesday after. Anyone who assumes the usual rhythm
will miss it.
-->
