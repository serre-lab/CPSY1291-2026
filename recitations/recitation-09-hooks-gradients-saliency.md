---
marp: true
theme: cpsy1291
paginate: true
math: katex
---

<!-- ===========================================================================
  CPSY 1291 — RECITATION 9: Looking inside a network
  TA-led, 80 minutes. Optional. Week 9 — hold this after Lecture 13.
  Assignment 4 is due Tue 11/10, so this is the last session before it.

  SCOPE: hooks, gradients w.r.t. the input, saliency, Grad-CAM, maximally
  activating patches, feature visualization — each PAIRED with the check that
  says whether it means anything. The pairing is the pedagogy; do not present
  a method without its check.

  TONE: these methods are useful AND untrustworthy, and the session should not
  resolve that tension in either direction. Students who leave thinking
  saliency maps are worthless are as badly served as those who leave thinking
  they are explanations.

  IF YOU RUN LONG: cut feature visualization (slides 14-15). Do not cut the
  weight-randomization sanity check.

  MATERIAL: handout-09-hooks-gradients-saliency.pdf, six exercises.
============================================================================ -->

<div class="eyebrow">Recitation 9 · Week 9</div>

# Looking inside a network

## Hooks, input gradients, and how far to trust a picture

<!--
2 min. Set the tone immediately and honestly: everything today is useful and
everything today is less trustworthy than anything else in the course.

Say the structure: methods come in pairs. Here is how to compute it, here is the
check that tells you whether it means anything. Neither half is optional.
-->

---

# The one idea

Every interpretability method answers a question of the form:

<mark>"What would change if …?"</mark>

— if this pixel changed, if this unit were removed, if this feature were amplified.

**Which** counterfactual a method asks is what distinguishes the methods — and
reading a figure without knowing its counterfactual is how people over-read them.

<!--
4 min. This framing is the most portable thing in the session. Give an example
of each: saliency asks about an infinitesimal pixel change; ablation asks about
removing a unit; feature visualization asks what input would maximize one.

Three different questions, three different pictures, and papers routinely
present them as if they were the same claim.
-->

---

# Forward hooks, properly

```python
acts = {}
def save(name):
    def hook(module, inputs, output):
        acts[name] = output.detach()
    return hook

h = net.features[8].register_forward_hook(save('conv4'))
with torch.no_grad():
    net(x)                # the hooks fire HERE
h.remove()                # always
```

<!--
5 min. Read the code as three moments: register, run, remove. The middle one is
where everything happens and it is the one students omit.

Note that for most layers you do not need a hook at all — net.features[:9](x)
runs a prefix. Hooks are for layers that are not prefixes, for grabbing several
layers in one pass, or for models whose forward does something slicing cannot.
-->

---

# The three ways hooks go wrong

1. **No `.detach()`** → you keep the autograd graph for every saved tensor and run out of memory on the third batch.
2. **No `.remove()`** → the hook fires on every later forward pass and silently overwrites your data.
3. **Reading `acts` before running the model** → it is empty. Hooks fire during `net(x)`.

<!--
5 min. All three are silent except the first, and the first is silent about its
cause. Make them photograph this slide.

Number 2 is the nastiest: your stored activations become whatever ran last,
which is usually a different image, and every downstream analysis is quietly
about the wrong stimulus.
-->

---

# What comes out, and how you reduce it

```python
A = acts['conv4']                     # (N, C, H, W)
X    = A.flatten(1)                   # (N, C*H*W)  keeps position
chan = A.mean(dim=(2, 3))             # (N, C)      discards position
```

<mark>These answer different questions, and papers differ on which they use.</mark>

<!--
4 min. Flattening keeps spatial information and gives tens of thousands of
features; averaging over space keeps one number per feature map and throws
position away.

An RSA score computed on the two is not the same analysis. If a paper does not
say which it did, you cannot reproduce it — a good thing for them to notice now,
before the project.
-->

---

# Gradients with respect to the *input*

Training asks how the loss changes with the **weights**. This asks how the output
changes with the **input**. Same machinery, different variable.

```python
x = x.clone().requires_grad_(True)
score = net(x)[0, class_idx]      # a LOGIT, not a probability
net.zero_grad()
score.backward()
g = x.grad[0]                     # (3, H, W)
```

<!--
5 min. The logit-vs-probability point is next slide, so just plant it here.

Two things that break it: wrapping this in torch.no_grad() (the graph is the
point), and forgetting net.zero_grad() (leftover gradients get added in).
-->

---

# Differentiate the logit, never the probability

A softmax probability near $1$ has a nearly **flat** gradient.

<mark>Differentiate the probability of a confident prediction and you get a blank map — and conclude the model is looking at nothing.</mark>

<!--
4 min. This is the single most common bug in student saliency code, and the
symptom looks like a finding rather than an error.

Draw the softmax curve and point at the saturated tail. Once they have seen the
picture, "the gradient of a saturated function is zero" is obvious.
-->

---

# The map

```python
sal = g.abs().max(dim=0).values      # (H, W): max over color channels
plt.imshow(sal, cmap='hot')
```

Two conventions worth understanding rather than copying:

- **`abs`** — a pixel that would strongly *decrease* the score is also evidence the model uses it
- **`max` over channels** — sum, or norm, are equally defensible and give visibly different pictures

<!--
5 min. The second point matters: three reasonable choices, three different
figures, and papers rarely say which. Ask the room what that implies about
comparing saliency maps across papers.

If you keep the sign instead of taking abs, use a diverging colormap — the
picture then shows evidence for and against.
-->

---

# What a saliency map actually claims

<mark>That an infinitesimal change to this pixel would change this logit by this much, at this exact input.</mark>

A **local linear approximation**. It does not say the model "looked at" the region.
It does not say the region is sufficient. It does not say it is necessary.

<!--
5 min. Read all three negations slowly. Each corresponds to a claim people make
from these figures, and each is a different question that would need a different
experiment.

Sufficiency: black out everything else and see if the model still answers.
Necessity: black out the region and see if it stops. Neither is what a gradient
computes.
-->

---

# The check that should always accompany it

<mark>Randomize the weights, layer by layer from the top, and recompute the map.</mark>

If it **still** outlines the object, the map is being produced by the image and
the architecture — not by anything the model learned.

This check fails for several published methods.

<!--
6 min. This is the most important slide in the deck. Spend the time.

The logic: a method that explains a trained model should produce a different
explanation for an untrained one. Several popular methods do not, which means
they were showing edges all along.

Cheaper, weaker check worth mentioning: perturb the input imperceptibly and
recompute. Saliency maps are notoriously unstable, and seeing that for yourself
beats reading about it.
-->

---

# EXERCISE 1

You randomize a network's weights and its saliency map **still** outlines the
object cleanly.

What have you learned?

<!--
5 min. Answer: that the map reflects the image and the architecture, not the
learned weights. Any claim it was supporting about the model's strategy is
unsupported.

Push the room: is the map useless? Not quite — it is a decent edge detector. It
is just not an explanation of this model, which is what it was being used as.
-->

---

# Grad-CAM: coarser, and more honest about it

```python
w   = dA.mean(dim=(2, 3), keepdim=True)   # importance per channel
cam = torch.relu((w * A).sum(1))[0]       # (h, w)
```

Weight each feature map by how much raising it would raise the score, then sum.
The ReLU keeps only evidence **for** the class.

<mark>The output is $7\times7$, upsampled. That is genuinely the resolution at which this layer represents anything.</mark>

<!--
5 min. The coarseness is a feature. A per-pixel map from a layer with a 7x7
spatial grid is claiming a precision the layer does not have, and Grad-CAM's
blurriness is more faithful than saliency's sharpness.

Needs both a forward hook (for A) and a backward hook (for dA) on the same
layer — mention it, and point at the handout for the full version.
-->

---

# Maximally activating patches

```python
resp = acts['conv4'][:, unit].amax(dim=(1, 2))
top  = resp.argsort(descending=True)[:9]
```

The oldest method, and often the most informative. Crop around each peak, show
nine, ask what they have in common.

<mark>The bias: it shows what drives the unit <em>within your dataset</em> — and you sorted for the commonality you are about to see.</mark>

<!--
6 min. The bias is subtle and worth unpacking. Any nine images sorted by any
criterion will have something in common, because you sorted them.

The check: look at the response VALUES, not just the images. If the top nine are
barely above the median, you are looking at noise arranged by a sort.

Also mention polysemanticity: units frequently show two or three unrelated
things in their top patches. That is a real property of trained networks and the
motivation for sparse-autoencoder features in the lecture.
-->

---

# Feature visualization, and its prior

```python
x = torch.randn(1, 3, 224, 224, requires_grad=True)
opt = torch.optim.Adam([x], lr=0.05)
for _ in range(256):
    opt.zero_grad()
    (-net.features[:9](x)[0, unit].mean()).backward()
    opt.step()
```

Run exactly this and you get **high-frequency noise** that drives the unit hard
and resembles nothing.

<!--
5 min. And that is not a bug — the optimizer did what you asked, and there is no
reason the input maximizing a unit should look like a photograph.

Ask the room what to do about it before revealing the next slide.
-->

---

# Every published feature visualization is regularized

Jitter between steps · blur · penalize total variation · optimize in a
decorrelated color space.

<mark>Those regularizers are chosen because they produce pictures people find interpretable.</mark>

So the image is partly about the model and partly about the prior you imposed —
which is not how a measurement works.

<!--
5 min. This is the honest caveat and it is rarely stated in papers. Say it
plainly and let it be uncomfortable.

Balance it: the pictures are still informative, and the hierarchy they reveal —
edges, then textures, then parts, then objects — replicates across architectures
and priors, which is evidence that it is not an artifact of the prior alone.
-->

---

# Comparing a model's attribution with a human's

1. Resize both maps to the same resolution
2. **Spearman** on the flattened maps — unrelated scales, only ordering compares
3. Average over images
4. **Split your participants in half and correlate** → the noise ceiling

<mark>Step 4 is where this analysis usually goes wrong. Same error as last week.</mark>

<!--
5 min. Close the loop with Recitation 8 explicitly: a model scoring 0.28 against
a human-human ceiling of 0.35 is doing very well; against a ceiling of 0.9 it is
doing poorly, and the raw number cannot tell you which.

This is also the machinery behind the human-alignment work in Lecture 19, so it
is worth having.
-->

---

# Where this lands

- **Assignment 4** — hooks to get layer activations, and attribution you have to argue about
- **Lecture 19** — human-alignment scores are exactly the pipeline on the last slide
- **Your project** — if it involves a figure of what a model "looks at", every check here applies

Next week: **sequences and dynamics** — RNNs, and networks as dynamical systems.

<!--
2 min. Point at handout-09.pdf.

Last word worth saying out loud: the goal is not to distrust these methods, it
is to know what each one claims. A student who leaves thinking saliency maps are
worthless is as badly served as one who leaves thinking they are explanations.
-->
