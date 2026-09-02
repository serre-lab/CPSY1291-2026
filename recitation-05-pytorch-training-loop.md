---
marp: true
theme: cpsy1291
paginate: true
math: katex
---

<!-- ===========================================================================
  CPSY 1291 — RECITATION 5: PyTorch, tensors, autograd, and the training loop
  TA-led, 80 minutes. Optional. Week 5 — hold this AFTER Lecture 8
  (MLPs and backpropagation). Assignment 2 is due Tue 10/13.

  HOW TO RUN THIS DECK: presenter notes carry timing, what to say, and the
  answers to the exercises.

  SCOPE: reading and debugging a training loop, not writing one from scratch.
  The assignment PROVIDES its training loops; the skill being taught is
  knowing which lines matter and how to diagnose one that fails.

  RUN THE CODE LIVE if the room has a projector and a Colab tab. The single
  most valuable five minutes of this session is deleting opt.zero_grad() in
  front of them and watching the loss explode.

  MATERIAL: handout-05-pytorch-training-loop.pdf covers the same ground in
  prose, with a seven-item debugging checklist and six exercises.
============================================================================ -->

<div class="eyebrow">Recitation 5 · Week 5</div>

# PyTorch

## Tensors, autograd, and the five lines that train everything

<!--
2 min. Frame the session honestly: you will not have to WRITE a training loop
in this course. You will have to read one, change one, and fix one. That is
what today is.

If you can run code live, say so now and promise the zero_grad demo — it gives
the room a reason to stay awake through the tensor mechanics.
-->

---

# The one idea

<mark>PyTorch is NumPy that remembers what you did to it.</mark>

Every operation on a tensor is **recorded**. When you finally produce one
number — the loss — the library replays the recording backwards and hands you
the derivative with respect to every parameter that took part.

That replay is `loss.backward()`. It is the only thing PyTorch does that NumPy cannot.

<!--
4 min. Land this properly; everything else follows from it.

Connect back to last week explicitly: they computed dL/dw by hand and checked
it with a two-sided difference. backward() is that same quantity, obtained by
bookkeeping instead of by algebra, for every parameter at once.
-->

---

# Tensors are arrays, with two gotchas

```python
x = torch.tensor(X_np, dtype=torch.float32)   # note the cast
x.numpy()                                     # and back
x.mean(dim=0)                                 # 'axis' is spelled 'dim'
```

**NumPy defaults to `float64`. PyTorch wants `float32`.**

<mark>`RuntimeError: expected scalar type Float but found Double` = an uncast NumPy array. Never a bug in your model.</mark>

<!--
5 min. Put the error message on the board in full — recognizing it by sight
saves each of them twenty minutes.

Second gotcha, worth stating here: classification TARGETS must be long
(integers), shape (N,), as class indices — not one-hot, not floats. That error
message is much less readable, so warn them in advance.
-->

---

# Batch first, always

```python
x.shape          # (784,)     one image — layers reject this
x[None].shape    # (1, 784)   a batch of one
```

| data | shape |
|---|---|
| tabular | $(N, D)$ |
| images | $(N, C, H, W)$ |
| sequences | $(N, T, D)$ |

<mark>Images are channels-<em>first</em> in PyTorch and channels-<em>last</em> in matplotlib: `img.permute(1, 2, 0)` before `imshow`.</mark>

<!--
4 min. The permute point produces a confusing error and comes up the moment
they try to look at a filter, which is Part 1a of the assignment. Say it now.

The (N, C, H, W) convention has a name worth mentioning — NCHW — because they
will see it in documentation and in error messages.
-->

---

# Autograd, in three lines

```python
w = torch.tensor([2.0], requires_grad=True)
loss = (w ** 2).sum()      # a graph is being recorded
loss.backward()            # replay it backwards
w.grad                     # tensor([4.])   — d(w^2)/dw = 2w
```

<!--
4 min. Run this live if you can. Seeing 4.0 appear where they know the answer
is 2w is what makes autograd stop being magic.

Then perturb it: change to w**3 and ask the room what .grad will print before
you run it. 3w^2 = 12.
-->

---

# Three facts that explain all autograd confusion

1. **Gradients accumulate** — `.grad` is *added* to, not replaced. Hence `opt.zero_grad()` every step.
2. **`backward()` needs a scalar** — if it complains, you forgot a `.mean()` or `.sum()`.
3. **The graph is freed after backward** — calling it twice means you built the loss outside the loop.

<!--
7 min. Fact 1 is the demo. Delete opt.zero_grad() from a working loop in front
of them: the loss falls for a few epochs and then explodes. Then put it back.

If asked WHY accumulation is the default rather than a bug: it lets you sum
gradients over several batches before stepping, which is how people train
models too big to fit a batch in memory. It is a feature that costs beginners
one line of attention.
-->

---

# `torch.no_grad()`

```python
model.eval()
with torch.no_grad():
    acc = (model(Xte).argmax(1) == yte).float().mean()
```

Nothing is recorded inside the block: **faster, and far less memory**.

Use it whenever you are evaluating rather than training — validation accuracy,
feature extraction, plotting.

<!--
4 min. The symptom worth naming: a script that trains fine and then runs out of
memory during evaluation. Almost always a missing no_grad, because the graph is
being kept for every evaluation batch.

Feature extraction from a pretrained network is the case they will hit in the
assignment.
-->

---

# `nn.Linear` is the layer from lecture

`nn.Linear(in, out)` computes $\boldsymbol{x}\boldsymbol{W}^{\!\top} + \boldsymbol{b}$ — a whole layer of last week's units.

```python
model = nn.Sequential(
    nn.Linear(64, 16),
    nn.ReLU(),               # between the linear layers, not after the last
    nn.Linear(16, 10),
)
sum(p.numel() for p in model.parameters())
```

<mark>`.weight` has shape `(out, in)` — the transpose of what you expect.</mark>

<!--
5 min. The (out, in) convention confuses everyone once. Say it now so that when
they inspect AlexNet's first layer and get (64, 3, 11, 11) they read it as
"64 filters" rather than "64 inputs".

Ask the room what happens if you delete the ReLU. Answer: the stack collapses
to a single linear map — more parameters, no more power. That is Lecture 8's
point and it is worth hearing twice.
-->

---

# The trap: never touch your logits

The last layer emits **logits** — raw, unbounded scores. Not probabilities.

`nn.CrossEntropyLoss` applies `log_softmax` **internally**.

- softmax before it → applied twice → trains badly, silently
- **ReLU** before it → every negative logit clamped to 0 → the model can no longer say *"this class is unlikely"*

<mark>Both are silent. Both are common in code you will find online.</mark>

<!--
7 min. Spend the time — this is the single highest-value slide in the deck,
because it is invisible and because they will meet it in copied code.

Make the ReLU case vivid: a classifier works by accumulating evidence for AND
against each class. ReLU destroys all the evidence against. The model still
trains, still improves, and still ends up much worse, with no error anywhere.

Give the rule to write down: if the loss has "CrossEntropy" in its name, the
last thing in your model is a Linear layer. Full stop.
-->

---

# The training loop, complete

```python
for epoch in range(200):
    opt.zero_grad()             # 1. forget last step's gradients
    out  = model(Xtr)           # 2. forward
    loss = lossfn(out, ytr)     # 3. how wrong are we?
    loss.backward()             # 4. gradients w.r.t. every parameter
    opt.step()                  # 5. one step downhill
```

<mark>Steps 1–4 only <em>compute</em>. Nothing moves until `opt.step()`.</mark>

<!--
6 min. Read it as five jobs, not eleven lines. Have someone in the room name
each job before you reveal the comments.

The highlighted line matters for debugging: if weights are not changing, the
question is whether step() is being called, not whether backward() is working.

Mention .item(): it pulls a Python float out of a one-element tensor. Printing
the tensor works but drags the whole graph along.
-->

---

# EXERCISE 1

Your loss falls for 5 epochs, then grows without bound and reaches `nan`.

Name the **two** most likely causes, and how you would tell them apart.

<!--
4 min. Answers: (i) a missing opt.zero_grad(), so gradients accumulate;
(ii) a learning rate far too large.

Telling them apart: a missing zero_grad usually gives you a few GOOD epochs
first, because early gradients are small. Too large a learning rate usually
misbehaves from the very first steps. Fix zero_grad first; if it still
diverges, it is the learning rate.
-->

---

# SGD or Adam?

| | `SGD` | `Adam` |
|---|---|---|
| is | the rule you derived last week | a per-parameter adaptive step |
| typical `lr` | $10^{-2}$–$10^{-1}$ | $10^{-3}$ |
| use when | you want to see the rule work | you want it to just train |

<mark>The learning rates are not interchangeable — Adam's $10^{-3}$ given to SGD trains imperceptibly.</mark>

<!--
4 min. When someone says "the network did not learn", the first question is
which optimizer and the second is what learning rate. Make that a habit.

Do not explain Adam's internals. It is enough that it keeps a per-parameter
step size and that this makes it forgiving.
-->

---

# Reading a pretrained network

```python
net = torchvision.models.alexnet(weights='DEFAULT')
net.eval()
W = net.features[0].weight.detach().numpy()   # (64, 3, 11, 11)
```

`.detach()` takes the tensor out of the autograd graph so it can become NumPy.

<mark>Pretrained models expect their inputs normalized the way they were trained. Skip it and the numbers are not comparable to anything published.</mark>

<!--
5 min. Read the shape out loud: 64 filters, 3 color channels, 11 x 11 pixels.
Then note that VGG's first conv is 3 x 3 — too small to show much structure at
all, which is a useful fact and a reason model choice is not arbitrary.

To display one filter: transpose(1,2,0) then rescale to [0,1], because imshow
of arbitrary floats is not meaningful.
-->

---

# When it does not learn — in this order

1. Is the loss going down **at all**? Print it. Log axis.
2. Did you call `opt.zero_grad()`?
3. **Learning rate** — try ×10 and ÷10 before anything else.
4. Targets the right dtype and shape? (`long`, $(N,)$, class indices)
5. Is there a nonlinearity **between** the linear layers?
6. Did you apply softmax or ReLU to the logits?
7. **Can it overfit 10 examples?**

<!--
6 min. Ordered by how often each is the answer. Item 3 fixes more cases than
everything else combined; say that.

Item 7 is the one to remember and deserves its own moment — next slide.
-->

---

# The thirty-second test

<mark>Train on ten examples and try to drive the loss to zero.</mark>

- **It cannot** → the bug is in the model, the loss, or the wiring.
- **It can** → the model works; your problem is the data, or the amount of training.

A model that cannot memorize ten points will never generalize from ten thousand.

<!--
4 min. This is what an experienced person does first, and it separates "broken
pipeline" from "hard problem" faster than any other check.

Note the connection forward: next week is about the opposite failure — a model
that memorizes and does NOT generalize. Both are about the same quantity, seen
from two sides.
-->

---

# EXERCISE 2

`nn.Sequential(nn.Linear(64,16), nn.Linear(16,10))`

1. How many parameters?
2. What function class can it express?

<!--
4 min. Answers: (64*16 + 16) + (16*10 + 10) = 1040 + 170 = 1210.

Function class: exactly the linear maps from R^64 to R^10 — the same class as a
single nn.Linear(64, 10), with 1,210 parameters instead of 650 and no more
expressive power.

Push: so is the extra layer useless? Not quite — it constrains the map to rank
at most 16, which is a real (and sometimes useful) restriction. Nice place to
mention that this is exactly what a linear autoencoder does, which is Lecture 10.
-->

---

# Where this lands in Assignment 2

- **Part 1** — reading AlexNet's first layer, normalizing inputs, `no_grad`
- **Part 4** — the provided `train_xor` loop: you read it, you do not write it
- **Everywhere** — `manual_seed`, and the fact that one run is one observation

Next week: **overfitting** — the model that trains perfectly and still fails.

<!--
3 min. Point at handout-05.pdf, especially the seven-item checklist, which is
the part worth photographing.

Close on the preview: today was "can it learn at all", next week is "did it
learn the right thing", and those turn out to be almost unrelated questions.
-->
