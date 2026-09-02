---
marp: true
theme: cpsy1291-recitation
paginate: true
math: katex
---

<!-- ===========================================================================
  CPSY 1291 — RECITATION 4: One neuron, and the calculus of learning
  TA-led, 80 minutes. Optional. Week 4 — hold this AFTER Lecture 6
  (the model neuron). Assignment 2 is due Tue 10/13.

  HOW TO RUN THIS DECK: every slide's presenter note gives a timing, what to
  say, and (where relevant) the answer to the exercise on the slide.

  SCOPE: Part A is the geometry of one unit (|w||x|cos theta, angles,
  gratings, tuning curves). Part B is the calculus every learning rule in the
  course is built from. Do NOT teach backpropagation as an algorithm — L08
  does that. Teach the chain rule as a pipeline and let the lecture name it.

  AUDIENCE SPLIT: roughly half the room has had calculus. Say at the top that
  Part B is revision for them, and offer them the option to arrive late rather
  than have them sit bored — a self-selected room keeps the pace honest.

  MATERIAL: handout-04-gradients-chain-rule.pdf covers the same ground in prose,
  with six exercises and answers.
============================================================================ -->

<div class="eyebrow">Recitation 4</div>

# One neuron, and the calculus of learning

## Angles, gratings, gradients, and the chain rule

<!--
2 min. Set the split immediately: Part A (about 30 min) is the geometry of a
single unit and is new to almost everyone. Part B (about 45 min) is derivatives
and the chain rule, and is revision if you have had calculus.

Tell the calculus-havers they may skip Part B, and mean it. Then say the one
thing they should not skip: the delta-rule derivation, because it turns three
memorized rules into one.
-->

---

# Part A · The geometry of one unit

<!--
2 min. Divider. Nothing to say beyond naming the section.
-->

---

# The dot product, written two ways

$$\boldsymbol{w}\cdot\boldsymbol{x} = \sum_i w_i x_i
\qquad = \qquad
\|\boldsymbol{w}\|\,\|\boldsymbol{x}\|\,\cos\theta$$

The first says **how to compute it**. The second says **what it means** — and it
splits the response into two separate causes:

- $\|\boldsymbol{x}\|$ — how **strong** the input is
- $\cos\theta$ — how well its **pattern** matches what the unit prefers

<!--
5 min. The two-causes framing is the point of the slide, so say it as a
scientific problem: a unit fires hard. Was that because the stimulus matched its
preference, or because the stimulus was simply intense? Those are different
claims, and the raw response cannot tell them apart.

This is the setup for the next slide, which is the fix.
-->

---

# Normalize, and only the pattern survives

```python
xn  = x / np.linalg.norm(x)               # unit length: intensity removed
cos = (w @ xn) / np.linalg.norm(w)
theta = np.degrees(np.arccos(np.clip(cos, -1, 1)))
```

Fix $\|\boldsymbol{x}\| = 1$ and the response becomes $\|\boldsymbol{w}\|\cos\theta$ — a **pure** measure of pattern match.

<mark>The `np.clip` is not paranoia: floating point returns 1.0000000000000002, and `arccos` of that is `nan`.</mark>

<!--
5 min. The clip point earns its place — they WILL hit it, and a nan appearing
halfway through a histogram is very hard to trace backwards.

Ask the room what normalizing corresponds to physically. Answer: contrast
normalization, which real visual cortex also does. It is not a numerical
convenience, it is a model of something.
-->

---

# The best stimulus for a linear unit is its own weights

Among inputs of fixed length, $\|\boldsymbol{w}\|\|\boldsymbol{x}\|\cos\theta$ is largest when $\cos\theta = 1$.

That is: when $\boldsymbol{x}$ points **along** $\boldsymbol{w}$.

<mark>This is why plotting a unit's weights as an image tells you what it is looking for — and why a first layer can be called "Gabor-like" at all.</mark>

<!--
4 min. Short and important. It licenses the entire practice of visualizing
filters, which they will do in the assignment and see again in the vision
lecture and the explainability lecture.

Flag the caveat honestly: this argument holds for a LINEAR unit. Add a
nonlinearity and "the optimal stimulus" becomes an optimization problem — which
is exactly what feature visualization is, and it arrives in Lecture 13.
-->

---

# Angles: radians, and the modulo-180 trap

NumPy trig takes **radians**. Use `np.deg2rad` / `np.rad2deg` and never convert by hand.

Orientation is defined **modulo 180°** — a bar at 10° and a bar at 190° are the same bar.

```python
d = abs(a - b) % 180
diff = min(d, 180 - d)
```

<!--
4 min. Give the failure concretely: a unit preferring 175 and a unit preferring
5 are 10 degrees apart, but subtract naively and you report 170 — you have
turned the two most similar units in the population into the two most different.

This appears in the assignment when they build a histogram of preferred
orientations.
-->

---

# A grating in four lines

```python
def grating(size=11, theta=0.0, freq=2.0, phase=0.0):
    y, x = np.mgrid[0:size, 0:size] / size
    proj = x * np.cos(theta) + y * np.sin(theta)
    return np.sin(2 * np.pi * freq * proj + phase)
```

The middle line is the whole idea: **project each pixel onto the direction $\theta$**, then make brightness a sine of that projection.

Everything perpendicular to $\theta$ shares a projection — which is why the stripes come out parallel.

<!--
4 min. Draw the projection on the board: a grid of pixels, an arrow at theta,
and the perpendicular lines of constant projection. Once they see that picture
the code is obvious; without it, the code is four magic lines.

Say why gratings and not photographs: they are the stimulus Hubel and Wiesel's
successors used to characterize V1, so measuring a network with them puts the
network and the cortex on the same axes. That comparison is the assignment.
-->

---

# Tuning curves, and the nuisance parameter

A **tuning curve** is response vs. a stimulus parameter. Sweep, record, plot.

**Phase is a nuisance.** A unit that likes vertical stripes may respond hard to
one phase and weakly to its opposite, purely because light and dark swapped.

<mark>Average over several phases at each orientation — that is what makes the curve about orientation rather than about where the stripes landed.</mark>

**HWHM**: half the distance between the two half-maximum points. Macaque V1 sits near 23°.

<!--
4 min. The nuisance-parameter idea generalizes far beyond this course and is
worth naming as such: a variable you do not care about but which affects your
measurement, so you average it out by design.

Draw a tuning curve and mark peak, baseline, halfway, and the two crossings.
Students routinely compute HWHM from the peak to zero instead of to the
baseline, and get the wrong number with no error message.
-->

---

# Part B · The calculus of learning

<!--
2 min. Divider. Repeat the offer: revision for anyone who has had calculus,
except for the delta-rule slide.
-->

---

# A derivative answers one question

<mark>If I nudge this input a little, how much does the output move, and in which direction?</mark>

Everything else — gradients, backprop, the whole training loop — is that same
question asked about more variables at once.

| $f$ | $f'$ |
|---|---|
| $x^2$ | $2x$ |
| $e^x$ | $e^x$ |
| $\sigma(x)$ | $\sigma(x)(1-\sigma(x))$ |
| $\mathrm{ReLU}(x)$ | $1$ if $x>0$, else $0$ |

<!--
5 min. Read the framing line and move; the table is reference, not a lecture.

One remark worth making: the sigmoid and tanh derivatives are written in terms
of the function's OWN OUTPUT. That is why a network can compute them almost for
free during backprop — it already has the output stored.
-->

---

# The chain rule is a pipeline

$$z = \boldsymbol{w}\cdot\boldsymbol{x} + b, \quad a = \mathrm{ReLU}(z), \quad L = \tfrac12 (a-y)^2$$

$$\frac{\partial L}{\partial w_i} = \underbrace{(a-y)}_{\partial L/\partial a}\cdot\underbrace{\mathbb{1}[z>0]}_{\partial a/\partial z}\cdot\underbrace{x_i}_{\partial z/\partial w_i}$$

Each factor is **local** — one stage, and quantities that stage already has.

<mark>That locality <em>is</em> backpropagation. Nothing else is going on.</mark>

<!--
5 min. The most important slide in Part B. Walk backwards through the three
factors on the board, naming each stage as you pass it.

Do not call this "backpropagation" as an algorithm — Lecture 8 does that, and
pre-empting it costs the lecture its punchline. Here it is just the chain rule
applied to a three-stage pipeline.
-->

---

# Where dead ReLUs come from

If $z \le 0$, the middle factor is **0**, so the whole product is **0**.

The weight does not move — no matter how wrong the answer was.

<mark>A unit that outputs 0 for every training input can never recover.</mark>

You will meet exactly this when you look inside the networks that fail on XOR.

<!--
4 min. This lands a concrete, checkable prediction and it is the reason the
assignment asks them to look inside the failures rather than just count them.

If someone asks how to fix it: smaller initial weights, a smaller learning rate,
or leaky ReLU. Say those exist and move on — the diagnosis is the lesson here,
not the cure.
-->

---

# The gradient points uphill

$$\nabla_{\boldsymbol{w}} f = \Big(\tfrac{\partial f}{\partial w_1}, \dots, \tfrac{\partial f}{\partial w_p}\Big)$$

Direction of steepest **increase**; its length says how steep.

To decrease a loss, step in $-\nabla$.

<mark>Every minus sign in every learning rule this term is that one sentence.</mark>

<!--
4 min. Use the hillside metaphor and then immediately puncture it: in two
dimensions you can see the hill, in 4,096 you cannot, and the gradient is the
only local information you have. That is why the step size matters so much.
-->

---

# EXERCISE 1 — derive the delta rule

Linear neuron $\hat{y} = \boldsymbol{w}\cdot\boldsymbol{x}$, squared error $L = \tfrac12 (y-\hat{y})^2$.

1. Compute $\partial L / \partial \boldsymbol{w}$.
2. Write the gradient-descent update.

<!--
4 min. Give them the full time; this is the exercise that pays.

Answer: dL/dw = (yhat - y) x, so w <- w + eta (y - yhat) x.

Then land it: that IS the delta rule, and Rescorla-Wagner, and LMS in signal
processing. Three literatures, three names, one derivation — "move the weights
in proportion to the error, along the input."

Say out loud: if you can reproduce those two lines you never have to memorize
any of the three, including on the midterm.
-->

---

# The learning rate is the whole difficulty

| $\eta$ | what you see |
|---|---|
| too small | loss falls slowly and smoothly; you run out of patience |
| about right | loss falls fast, then flattens |
| too large | loss oscillates, or hits `nan` in a few steps |

```python
plt.semilogy(losses)      # a log y-axis, always
```

<mark>Flat on a linear axis is often a clean straight descent on a log axis.</mark>

<!--
5 min. The semilogy habit decides whether they conclude "it is not learning" or
"it needs more epochs", and those lead to opposite actions.

Mention the classical decaying schedule eta_t = eta_0/(1+t): shrinks fast enough
to settle, slowly enough to still travel any finite distance. Say that this
condition has a name (Robbins-Monro) and that the fashionable schedulers later
in the term do NOT satisfy it — a thread to pick up in Lecture 17.
-->

---

# Check a gradient you derived by hand

```python
def numerical_grad(f, w, eps=1e-5):
    g = np.zeros_like(w, dtype=float)
    for i in range(w.size):
        wp, wm = w.copy(), w.copy()
        wp[i] += eps; wm[i] -= eps
        g[i] = (f(wp) - f(wm)) / (2 * eps)
    return g
```

Two-sided, `eps=1e-5`. Should agree to about $10^{-7}$.

Slow — run it **once** on a tiny example, confirm, then never again.

<!--
5 min. The most useful debugging tool in this half of the course. Make them
photograph it.

Give the failure dictionary: exactly a factor of 2 = a missing 1/2 in the loss.
A sign flip = you wrote (y - yhat) where the derivative gives (yhat - y).
Agreement everywhere except a few entries = a ReLU sitting exactly at its kink.
-->

---

# EXERCISE 2

Your analytic gradient is **exactly twice** your numerical gradient, everywhere.

What is the bug?

<!--
3 min. Answer: a missing factor of 1/2 in the loss — one of you differentiated
(y - yhat)^2 and the other 1/2 (y - yhat)^2.

Follow-up worth asking: does this bug break training? Not really — it is
equivalent to doubling the learning rate. Which is a good way to make the point
that a wrong gradient can still descend, and that "the loss went down" is not
evidence your derivation was right.
-->

---

# Why it can fail once there is a hidden layer

One linear neuron + squared error = a **bowl**. One minimum, found from anywhere.

Add a hidden layer and you get:

- **local minima** — a valley that is not the deepest valley
- **saddle points** — uphill one way, downhill another, near-zero gradient between
- **dead units** — gradient exactly zero, permanently

<!--
5 min. Draw the bowl, then draw a bumpy landscape next to it. The visual
contrast is the whole slide.

Be careful not to overstate local minima: in very high dimensions saddles are
the more common obstacle, and modern practice suggests bad local minima are
rarer than the 1990s feared. Say that honestly rather than repeating the
textbook story.
-->

---

# The consequence you have to live with

<mark>The same network, trained twice from different random initializations, can give different answers — and one of them can simply fail.</mark>

Not a bug to fix.

It means one training run is **one observation**, and a claim about an
architecture needs several seeds.

Same discipline as $t$-SNE two weeks ago.

<!--
4 min. Close the loop with Recitation 3 explicitly — same rule, different
method, and that recurrence is the point.

The assignment makes them run 20 seeds per hidden width for exactly this
reason. Say that the table they will produce is meaningless with one seed, and
that this is a fact about science, not about PyTorch.
-->

---

# Where this lands in Assignment 2

- **Part 1** — $\|\boldsymbol{w}\|\|\boldsymbol{x}\|\cos\theta$, normalized patches, gratings, tuning curves, HWHM
- **Part 2–3** — the delta rule you just derived, and its learning rate
- **Part 4** — dead units, seeds, and why one run is not a result

Next week: the same ideas in **PyTorch** — tensors, autograd, and a training loop.

<!--
2 min. Point at handout-04.pdf for the prose version and its six exercises with
answers, and at office hours.

Preview Recitation 5 honestly: you will not have to write a training loop in the
assignment, but you will have to read one, and next week is where reading one
stops being frightening.
-->
