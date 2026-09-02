---
marp: true
theme: cpsy1291-recitation
paginate: true
math: katex
---

<!-- ===========================================================================
  CPSY 1291 — RECITATION 10: Sequences, recurrence, and dynamics
  TA-led, 80 minutes. Optional. Week 10 — hold this after Lecture 15.
  Assignment 5 is due Tue 11/24.

  SCOPE: two lectures, one object. L14 treats the RNN as a sequence machine,
  L15 treats the same equations as a dynamical system. Teach them as one thing
  seen twice — that framing is the point of the session.

  THE DEMO: if you can run code live, the highest-value five minutes is
  plotting gradient norm against time step on a log axis for a 200-step
  sequence. Vanishing gradients stop being received wisdom the moment they
  see the straight line.

  IF YOU RUN LONG: shorten the Hopfield block. Do not cut fixed points — the
  assignment and Lecture 15 both turn on them.

  MATERIAL: handout-10-sequences-recurrence-dynamics.pdf, six exercises.
============================================================================ -->

<div class="eyebrow">Recitation 10</div>

# Sequences, recurrence, dynamics

## Two lectures, one object

<!--
2 min. Say the frame: Lecture 14 gave you a machine that processes sequences.
Lecture 15 gave you a dynamical system with fixed points. They are the same
equations, and today is about seeing that.

The second view is the one that lets you reverse-engineer a trained network
rather than just evaluate it, which is why it is worth the trouble.
-->

---

# The one idea

<mark>A recurrent network is a map $\boldsymbol{h} \mapsto f(\boldsymbol{h}, \boldsymbol{x})$, applied over and over.</mark>

Everything follows:

- training is hard because you differentiate a **long composition** of that map
- memory exists because the map has **states it does not leave**

<!--
4 min. Read it and move. Both halves get their own slides later; this is the
sentence they should carry out of the room.
-->

---

# Shapes, before anything else

| object | shape |
|---|---|
| a batch of sequences | $(N, T, D)$ with `batch_first=True` |
| PyTorch's **default** | $(T, N, D)$ |
| hidden state | $(\text{layers}, N, H)$ — batch in the **middle**, always |

<mark>Always pass `batch_first=True`. And note the hidden state ignores it.</mark>

<!--
6 min. That inconsistency is the single most common source of silent bugs in
recurrent code, and it is worth a minute of indignation.

Why it is silent: (T, N, D) and (N, T, D) hold the same number of elements, so
nothing errors when T and N are compatible. You simply train on transposed data,
and the model learns nothing about time while reporting a perfectly normal loss.
-->

---

# A cell, by hand

$$\boldsymbol{h}_t = \tanh(\boldsymbol{W}_{hh}\boldsymbol{h}_{t-1} + \boldsymbol{W}_{xh}\boldsymbol{x}_t + \boldsymbol{b})$$

```python
h = torch.zeros(N, d_hid)
for t in range(T):
    h = cell(X[:, t], h)          # the SAME weights at every step
```

Two things in that loop: **weight sharing across time**, and a graph $T$ layers deep.

<!--
6 min. Weight sharing across time is the temporal analogue of a convolution's
weight sharing across space — say it in those words, because the parallel is
exact and it is what lets one network handle any sequence length.

The second: unrolling. An RNN run for 50 steps IS a 50-layer feedforward network
whose layers happen to be identical. Everything about training deep networks
applies, only worse.
-->

---

# The library version, and which output you want

```python
rnn = nn.LSTM(input_size=D, hidden_size=H, batch_first=True)
out, (h_n, c_n) = rnn(X)      # out: (N, T, H)   h_n: (1, N, H)
```

- **classifying a whole sequence** → `h_n`, or a pooled `out`
- **labeling every step** → `out`

<!--
5 min. Straightforward, but students routinely take out[:, -1] when they meant
h_n and vice versa, and with a single layer those are the same thing — so the
bug only appears when they add a layer.

Say that: it works until it doesn't, which is the worst kind of bug.
-->

---

# Why gradients vanish

The loss at step $T$, differentiated w.r.t. a weight at step $1$, needs the chain
rule through every step between.

<mark>That is a <em>product</em> of $T$ Jacobians. Products of numbers below 1 go to zero; above 1, they explode.</mark>

That is the entire phenomenon. There is nothing else to it.

<!--
6 min. Deflate the mystery — students arrive thinking vanishing gradients are
deep and they are arithmetic.

Connect back to Recitation 4: the chain rule as a pipeline, multiplying local
slopes. Here the pipeline is 200 stages long and the slopes are all similar, so
the product is an exponential.
-->

---

# See it for yourself

```python
loss.backward()
for t, h in enumerate(hidden_states):
    print(t, h.grad.norm().item())     # needs h.retain_grad() first
```

Plot norm against $t$, log $y$-axis. A clean exponential decay backwards in time.

<mark>Worth seeing once — it turns received wisdom into an observation.</mark>

<!--
6 min. Run this live if at all possible. The straight line on a log axis is the
memorable thing.

retain_grad() is needed because PyTorch discards gradients for non-leaf tensors
by default. Mention it or the code silently gives None.
-->

---

# What to do about each

| problem | fix |
|---|---|
| **exploding** | `clip_grad_norm_(model.parameters(), 1.0)` — one line, reliable |
| **vanishing** | gating: LSTM or GRU. **Clipping does not help** |

Gating works because the cell state passes through an **addition**, not a
repeated multiplication — so the gradient has a path back that is not a long product.

<!--
6 min. The clipping asymmetry is worth stating clearly: clipping bounds
gradients from ABOVE. It does nothing whatsoever for gradients that are too
small, and students try it anyway.

The gating mechanism in one sentence is the highlighted line. The three gates are
bookkeeping around that idea, and Lecture 14 already gave them.
-->

---

# Teacher forcing, and the mismatch it creates

Training a generator: feed back the model's own output, or the true previous token?

Feeding the truth — **teacher forcing** — trains far faster and more stably.

<mark>At generation time the model sees its own outputs, including its own mistakes, which it never saw during training. Errors compound.</mark>

<!--
5 min. Worth knowing before Lecture 17, where this same trick trains every
language model in existence.

Ask the room what could be done about it. Answers exist (scheduled sampling,
and in practice: enormous amounts of data), and none is clean. It is a real
unsolved seam in how these models are trained.
-->

---

# Now: the same equations, as a dynamical system

Hold the input constant. The update becomes a map from $\boldsymbol{h}$ to
$\boldsymbol{h}$.

**Iterate it. Ask where it goes.**

<!--
3 min. Divider, and a shift of gears. Say explicitly that nothing has changed
about the network — only the question being asked of it.
-->

---

# Finding fixed points, numerically

A fixed point: $\boldsymbol{h}^* = f(\boldsymbol{h}^*)$.

```python
h = torch.randn(200, H, requires_grad=True)     # 200 starts at once
opt = torch.optim.Adam([h], lr=0.01)
for _ in range(2000):
    opt.zero_grad()
    q = 0.5 * ((cell(x_const, h) - h) ** 2).sum(1)
    q.sum().backward(); opt.step()
found = h[q.detach() < 1e-8]
```

<mark>This is how a trained RNN gets reverse-engineered — and autograd does all the work.</mark>

<!--
5 min. The trick is worth admiring: you are optimizing the STATE, not the
weights, with the weights frozen. Same optimizer, different variable — the third
time this course has done that (weights, inputs, now states).

Many starting points matters. A network can have several fixed points, each with
its own basin, and one start finds one of them.
-->

---

# Stable, unstable, or a line?

Linearize: take the Jacobian at the fixed point, look at its eigenvalues.

| | meaning |
|---|---|
| all $\|\lambda\| < 1$ | **stable** — nearby states fall in. A memory |
| some $\|\lambda\| > 1$ | **unstable** in those directions. A saddle |
| one $\|\lambda\| \approx 1$ | a **line attractor** — the state drifts freely along it |

<!--
5 min. The third row is the neuroscience. A direction that neither decays nor
grows is a way to hold a CONTINUOUS quantity — evidence accumulated so far, an
eye position, a value in working memory.

Draw it: a valley floor that is flat along one direction and steep across it.
Once they see the picture, "line attractor" is self-explanatory.
-->

---

# EXERCISE 1

A trained RNN has a fixed point whose Jacobian has one eigenvalue at $0.999$ and
all the rest below $0.6$.

What is the network doing?

<!--
6 min. Answer: it has a line attractor. One direction along which the state
neither decays nor grows, with every other direction contracting onto it — an
integrator, or a working-memory store.

Push: what task would produce this? Any task requiring the network to accumulate
evidence over time and hold the running total. That is exactly the decision-
making result from Lecture 15.
-->

---

# A Hopfield network in fifteen lines

```python
def store(P):                       # P: (n_patterns, n) of +/-1
    W = P.T @ P / len(P)            # Hebbian outer product
    np.fill_diagonal(W, 0)
    return W

def recall(W, s, steps=20):
    for _ in range(steps):
        for i in rng.permutation(len(s)):        # ASYNCHRONOUS
            s[i] = 1 if W[i] @ s >= 0 else -1
    return s
```

<mark>Memories <em>are</em> the attractors. Recall is falling downhill.</mark>

<!--
5 min. Three facts to state and, ideally, to run:

(1) Recall monotonically DECREASES the energy -0.5 s W s, which is why it
converges rather than cycling — and asynchronous updating is required for that
guarantee. Synchronous updates can oscillate.

(2) Capacity is about 0.14n patterns for n units. Store more and recall returns
blends.

(3) The NEGATION of every stored pattern is also a fixed point, straight out of
the symmetry of the update rule. Students find this surprising and it is a
one-line consequence.
-->

---

# EXERCISE 2

A Hopfield network with **100 units** is given **30 patterns** to store.

What happens on recall, and why?

<!--
5 min. Answer: capacity is about 0.14 * 100 = 14, so 30 is roughly double it.
Recall converges to spurious states — blends and mixtures — rather than to the
stored patterns, and it degrades further the more you store.

Worth noting: this is a graceful, gradual failure, not a cliff, and modern
"dense associative memory" variants push the capacity far higher. Lecture 15
mentions the connection to energy-based models and to attention.
-->

---

# Where this lands

- **Assignment 5** — RNN shapes, gradient behavior over long sequences, and a Hopfield network
- **Lecture 16** — attention exists because these two walls are real
- **The final** — L14 and L15 are both examinable

Next week: **attention, transformers, and generative models** — the last content session, and the A5 clinic.

<!--
3 min. Point at handout-10.pdf.

Say next week's shape now, because it is unusual: it covers three lectures in
one session because Assignment 5 is due the following Tuesday and Thanksgiving
follows immediately. It will be dense and the handout is the longest of the term.
-->
