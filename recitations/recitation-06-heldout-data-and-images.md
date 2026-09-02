---
marp: true
theme: cpsy1291-recitation
paginate: true
math: katex
---

<!-- ===========================================================================
  CPSY 1291 — RECITATION 6: Held-out data, and images through a network
  TA-led, 80 minutes. Optional. Week 6 — hold this AFTER Lecture 10.
  Assignment 3 is due Tue 10/27.

  WHY THIS SESSION CARRIES TWO TOPICS: the midterm is Thu 10/22 and
  Recitation 7 is given over to it entirely, so there is no session between
  Lecture 11 (the CNN lecture, Tue 10/20) and the A3 deadline. Part B
  therefore delivers the convolution arithmetic a few days before the lecture
  that motivates it. Do not apologize for this on the slide; just teach it.

  TIMING: Part A ~46 min, Part B ~32 min. If you run long, cut the hooks slide
  — Recitation 9 covers hooks properly — but do NOT cut the output-size slide.

  MATERIAL: handout-06-heldout-data-and-images.pdf covers the same ground in
  prose, with six exercises and answers.
============================================================================ -->

<div class="eyebrow">Recitation 6</div>

# Held-out data, and images

## Two topics, because the midterm eats next week

<!--
2 min. Say the plan: Part A is the discipline of held-out data — the most
transferable 45 minutes of the term, and it applies to a psychology experiment
as directly as to a network. Part B is the arithmetic of convolutions, arriving
a few days before Lecture 11 so that the assignment is not blocked.

Say plainly that next week's session is the midterm review and nothing else.
-->

---

# The one idea

<mark>A number computed on data your model has already seen is not a measurement of your model. It is a measurement of your model's memory.</mark>

Everything in Part A exists to keep those two apart.

<!--
2 min. Read it, pause, move. Do not elaborate — the rest of the session is the
elaboration.
-->

---

# Part A · Held-out data

<!--
2 min. Divider.
-->

---

# Three splits, three jobs

| split | used for | seen by the model |
|---|---|---|
| **training** | fitting the parameters | constantly |
| **validation** | hyperparameters, early stopping | indirectly, many times |
| **test** | one final estimate | **once**, at the very end |

<mark>Validation data is not seen by the optimizer — but it <em>is</em> seen by you.</mark>

<!--
5 min. The highlighted line is the subtle part and the reason there are three
splits and not two.

Make it concrete: every time you look at validation accuracy and change
something, a little information leaks from the validation set into your model.
Do that twenty times and validation accuracy is optimistic too. That is why the
test set exists and why it is opened once.
-->

---

# The cardinal rule

<mark>Tune anything against the test set and you no longer have an estimate of generalization. You have a training score with extra steps.</mark>

The most common methodological error in student projects, in papers, and in industry.

```python
Xtr, Xte, ytr, yte = train_test_split(
    X, y, test_size=0.2, random_state=0, stratify=y)
```

<!--
4 min. Give the concrete version: "I tried eight hidden-layer sizes and reported
the best test accuracy" is the same error, and it sounds completely reasonable
when you say it out loud. That is why it survives peer review.

stratify=y keeps class proportions equal across the split. Without it, a small
or imbalanced dataset can produce a test set missing a class entirely.
-->

---

# When data is small: cross-validation

```python
scores = cross_val_score(clf, X, y, cv=5)
print(f"{scores.mean():.3f} +/- {scores.std():.3f}")
```

<mark>Report the spread, not just the mean. $0.72 \pm 0.02$ and $0.75 \pm 0.09$ are not distinguishable.</mark>

**The leak hiding here:** any preprocessing fitted on the data — scaling, PCA,
feature selection — must be fitted **inside** each fold.

<!--
5 min. The leak is the important half. Fit PCA on all your data and then
cross-validate, and every fold's "held-out" data has already shaped the
features. The scores go up. They are wrong, and nothing reports it.

Give them the fix by name: sklearn.pipeline.Pipeline does it correctly by
construction, and using it is easier than doing it wrong.
-->

---

# The diagnostic plot of the course

| what you see | what it means |
|---|---|
| both high, both flat | **underfitting** — too little capacity, or $\eta$ too small |
| train falls, val follows | healthy; keep going |
| train → 0, val turns up | **overfitting** — stop at the val minimum |
| train high, val *lower* | **a bug** — leakage, or a mismatched split |

<!--
5 min. Draw all four shapes on the board. They are visually distinct and that
is the point — this plot is a diagnosis, not decoration.

Row 4 deserves emphasis: students who see it conclude their model generalizes
unusually well. It never does. Validation above training means dropout is on in
one regime and off in the other, or the split is mismatched, or there is a leak.
-->

---

# Overfitting is really about too few examples

A model with a million parameters is not overfitting because a million is large.

It overfits because it has **more freedom than the data constrains**.

<mark>Whenever you want to say "the model is too big", check whether you mean "the dataset is too small".</mark>

<!--
4 min. This reframing is what makes the next slide (double descent) survivable
— if capacity alone caused overfitting, double descent would be impossible.

It is also the honest framing for neuroscience: the brain has far more synapses
than it has training examples in any classical sense, and that puzzle is a real
one, not a rhetorical one.
-->

---

# Double descent, plotted honestly

```python
plt.plot(widths, test_err, 'o-')
plt.xscale('log')             # you cannot see it on a linear axis
plt.axvline(w_interp, ls=':')
```

Past the **interpolation threshold** — where the model fits the training data
exactly — test error frequently falls again, often below the bottom of the U.

<mark>Two ways to plot it so that you see nothing: a linear $x$-axis, and widths sampled too coarsely to bracket the threshold.</mark>

<!--
5 min. The plotting advice is not housekeeping — it decides whether the effect
is visible at all, and a student who plots it linearly concludes the phenomenon
is not real.

Say what it does NOT claim: overfitting is not imaginary, and this does not
license training without a validation set. It says the classical bargain is
incomplete, not wrong.
-->

---

# Enough probability for an error bar

Draw $n$ independent samples with standard deviation $\sigma$, take the mean.

$$\text{that mean has standard deviation } \sigma/\sqrt{n}$$

```python
acc = np.array([run(seed) for seed in range(10)])
print(f"{acc.mean():.3f} +/- {acc.std(ddof=1)/np.sqrt(len(acc)):.3f}")
```

<mark>Quadrupling your seeds only halves your error bar.</mark>

<!--
4 min. The sqrt(n) is the whole slide. The practical consequence: if a
difference is not visible with 10 seeds, 40 will rarely rescue it — so the
answer to a marginal result is a better experiment, not more runs of the same
one.
-->

---

# SD or SE? They answer different questions

- **standard deviation** — how much does a *single run* vary?
  *"How reliable is this architecture?"*
- **standard error** — how well do I know the *average*?
  *"Is this mean different from that one?"*

<mark>Label your error bars. An unlabeled error bar is uninterpretable — and about half of them in the literature are.</mark>

<!--
4 min. This is a lab skill they will use for years. Say that.

Also: which n is it? Ten training runs give n = 10 RUNS, not n = 10,000 test
images. Ask what you would have to redo to get another independent number —
that is your n. Getting this wrong shrinks error bars by a factor of 30 and is
extremely common.
-->

---

# EXERCISE 1

You tune the number of hidden units by trying eight values, keep the one with
the best **test** accuracy, and report that accuracy.

What have you actually measured?

<!--
4 min. Answer: the best of eight noisy numbers, which is biased upward. You
used the test set as a validation set, so the number is a training score in
disguise.

The correct procedure: choose the width on validation, then evaluate that one
model once on test. Ask the room how much bias this introduces — nobody knows,
which is exactly the problem.
-->

---

# Part B · Images through a network

<!--
2 min. Divider. Note that Lecture 11 supplies the biology and the architecture;
this is the arithmetic only.
-->

---

# An image is a tensor

| object | shape |
|---|---|
| one color image | $(3, H, W)$ — channels **first** |
| a batch | $(N, 3, H, W)$ — "NCHW" |
| a conv layer's weights | $(C_\text{out}, C_\text{in}, k, k)$ |

Pretrained models expect their training normalization: ImageNet mean
$(0.485, 0.456, 0.406)$, sd $(0.229, 0.224, 0.225)$.

<!--
4 min. Read the weight shape aloud as "C_out filters, each with C_in channels,
each k by k". That reading is what makes AlexNet's (64, 3, 11, 11) legible.

The normalization: skip it and the model still produces numbers — they are just
not comparable to anything published. Silent, again.
-->

---

# Convolution is template matching, repeated

A small filter slides over the image. At every position: a **dot product**
between the filter and the patch underneath.

Each output pixel answers *"how well does this pattern match the image here?"*

<mark>Everything from Recitation 4 applies unchanged — including that the optimal patch is the filter itself.</mark>

<!--
4 min. Connect it back explicitly: the response at each location is
||w|| ||x|| cos(theta), so the whole geometry of the model neuron carries over,
one location at a time.

That is also why first-layer filters can be displayed as images and recognized
as edge detectors — a fact they will use in the assignment.
-->

---

# Weight sharing, and the number that justifies it

One layer, $224\times224\times3$ image → 64 outputs:

| | parameters |
|---|---|
| fully connected | $\approx 9{,}600{,}000$ |
| convolutional, $11\times11$, 64 filters | $23{,}232$ |

<mark>Four hundred times fewer — and a built-in assumption that a feature worth detecting somewhere is worth detecting everywhere.</mark>

<!--
4 min. The assumption is a claim about vision, not a computational trick. It is
approximately true, which is why it works, and where it is false — faces are not
equally likely everywhere in a photograph — networks have to relearn position
sensitivity later.

Worth noting: the brain does something similar but not identical. Retinotopic
maps repeat similar tuning across position, but the weights are not literally
shared.
-->

---

# The output-size formula — and why not to use it

$$H_\text{out} = \left\lfloor\frac{H - k + 2p}{s}\right\rfloor + 1$$

```python
conv = nn.Conv2d(3, 64, kernel_size=3, padding=1)
conv(torch.zeros(1, 3, 32, 32)).shape      # (1, 64, 32, 32)
```

<mark>Build the layer, push zeros through it, print the shape. One line, never wrong.</mark>

<!--
4 min. Give the formula, then tell them not to trust their head with it. The
zeros trick is the actual professional habit and it is worth naming as such.

Two settings worth memorizing anyway: p = (k-1)/2 with s = 1 leaves the size
unchanged, and stride 2 or a 2x2 max-pool halves it.
-->

---

# One block, and what it maps onto

```python
nn.Sequential(
    nn.Conv2d(3, 32, 3, padding=1),   # template matching   — selectivity
    nn.ReLU(),                        # keep matches, drop mismatches
    nn.MaxPool2d(2),                  # tolerate small shifts — invariance
)
```

<mark>Selectivity from the convolution, tolerance from the pooling — simple cells and complex cells.</mark>

Stack blocks and receptive fields grow: early units see edges, deep units see objects.

<!--
4 min. This is the bridge to Lecture 11 and it is worth stating even though the
lecture will state it again — hearing it twice from two directions is how it
sticks.

If time allows, ask: why does the receptive field grow? Because each unit sees a
patch of the layer below, and each of THOSE saw a patch of the layer below that.
Composition, not magic.
-->

---

# EXERCISE 2

`nn.Conv2d(3, 16, kernel_size=5, padding=2, stride=1)` applied to a
$(8, 3, 64, 64)$ batch.

1. Output shape?
2. Number of parameters?

<!--
4 min. Answers: H_out = (64 - 5 + 4)/1 + 1 = 64, so (8, 16, 64, 64).
Parameters: 16 * 3 * 5 * 5 + 16 = 1,216.

Then ask the harder follow-up: how many MULTIPLICATIONS does this layer do per
image? 16 * 3 * 5 * 5 * 64 * 64 ≈ 4.9 million. Parameters and compute are
different quantities, and conv layers have few of the first and many of the
second. That distinction explains a lot about why vision models are slow.
-->

---

# Getting activations out of the middle

```python
with torch.no_grad():
    feats = net.features[:5](x)       # simplest: run a prefix
```

When the layer is not a prefix, register a **forward hook**:

```python
h = net.features[8].register_forward_hook(grab('conv4'))
net(x); h.remove()                    # always remove it
```

`.detach()` or you keep the graph. `.remove()` or it fires forever.

<!--
4 min. Keep it brief — Recitation 9 does hooks properly. The point today is
that both routes exist and that the prefix route covers most cases.

If you are running short on time, this is the slide to cut.
-->

---

# Where this lands

- **Assignment 3** — train/validation curves, seeds and error bars, images through a pretrained network
- **The midterm** — the three splits and the four curve shapes are examinable
- **Your final project, and every project after it**

Next week: **the midterm review**, and nothing else.

<!--
2 min. Point at handout-06.pdf. Say which parts are examinable, since the
midterm is nine days away and this is the last content session before it.

Announce next week's format now: worked problems from L01-L08, bring questions,
no new material.
-->
