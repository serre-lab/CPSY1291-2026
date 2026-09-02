---
marp: true
theme: cpsy1291
paginate: true
math: katex
---

<!-- ===========================================================================
  CPSY 1291 — RECITATION 3: Matrices, curve fitting, and methods with knobs
  TA-led, 80 minutes. Optional. Week 3 — hold this AFTER Lecture 4 and
  BEFORE Assignment 1 is due (Tue 9/29).

  HOW TO RUN THIS DECK: every slide's presenter note gives a timing, what to
  say, and (where relevant) the answer to the exercise on the slide. Slides
  marked EXERCISE are meant to be done live — put the prompt up, let them try
  for 2-3 minutes, then take answers from the room before revealing.

  SCOPE: the four things Assignment 1 needs that Recitations 1-2 did not give
  them — matrix products, curve fitting and R^2, Spearman on the upper
  triangle, and knobs (random_state, perplexity). Do NOT re-teach PCA; R02
  did that, and repeating it costs the time this session does not have.

  MATERIAL: handout-03-matrices-fitting-knobs.pdf covers the same ground in
  prose. Say so at the start so nobody takes notes.

  DO NOT: work any part of Assignment 1 on the board. Every example here uses
  different data on purpose.
============================================================================ -->

<div class="eyebrow">Recitation 3 · Week 3</div>

# Matrices, fitting, and knobs

## The last three things Assignment 1 needs

<!--
2 min. Say the frame: Assignment 1 is due Tuesday. This session is the last
scheduled help before then, and it covers only what R01 and R02 did not.

Set the expectation out loud: nothing from the assignment will be worked on
the board. Every example uses different data, deliberately. Point at office
hours for assignment-specific questions.
-->

---

# Where we left off

You have: arrays and shapes, `axis=`, broadcasting, dot products, cosine,
dissimilarity matrices, PCA, and how to read an $R^2$.

Today: **matrix products**, **fitting a curve**, **rank correlation**, and
what to do about a method that has a **knob**.

<mark>The theme: a method with a knob does not give you a result — it gives you a family of results.</mark>

<!--
2 min. Read the last line aloud; it is the through-line of the session and of
Part 4 of the assignment.

If the room is thin on R02 attendance, do NOT back up — point at
bootcamp-notes.pdf and recitation-02, and keep going. Backing up costs the
whole session.
-->

---

# The shape rule, and why it is everything

$(m \times k)$ times $(k \times n)$ gives $(m \times n)$.

The **inner** dimensions must match, and they vanish. The **outer** ones survive.

```python
A = np.zeros((120, 4096))     # 120 images x 4096 units
B = np.zeros((4096, 10))      # 4096 units x 10 categories
(A @ B).shape                 # (120, 10)  — 4096 cancelled
```

<mark>Say the cancelled dimension out loud: "summed over units." That is what a matrix product <em>is</em>.</mark>

<!--
4 min. Do the shape arithmetic on the board once, slowly, with the inner pair
circled and struck out. It is worth the chalk.

The payoff line is the last one: naming the summed index tells you instantly
whether the operation means anything. "Summed over units" is sensible;
"summed over images" usually is not, and that is how they will catch a
transpose error before it costs an hour.
-->

---

# `@` is not `*`

- `A * B` — element by element, and it **broadcasts**
- `A @ B` — contracts the shared index

They agree on nothing. NumPy will run the wrong one happily whenever the shapes permit.

<mark>If you wrote `*` and the shape came out the way you expected, check whether you meant `@`.</mark>

<!--
4 min. This is the single most common silent error in the first assignment.

Make it concrete: X * X.T on a square matrix runs and returns garbage. There is
no error message and no nan. The only defense is the habit on the next slide.
-->

---

# Three products, and what each one holds

$\boldsymbol{X}$ is $(n \times p)$: $n$ stimuli down, $p$ features across.

| you write | shape | it holds |
|---|---|---|
| `X @ X.T` | $(n \times n)$ | every pair of **stimuli**, dotted |
| `X.T @ X` | $(p \times p)$ | every pair of **features** → covariance |
| `X @ v`   | $(n,)$ | every stimulus projected onto direction $\boldsymbol{v}$ |

<!--
4 min. Land this table — it is most of the assignment in three lines.

Row 1 is where every dissimilarity matrix comes from. Row 2, after centering
and dividing by n-1, IS the covariance matrix whose eigenvectors PCA returns —
connect it back to last session explicitly. Row 3 is what pca.transform does,
one component at a time.

Then give the habit: every time you write .T, write the resulting shape in a
comment. Three seconds, highest-yield habit in the course.
-->

---

# EXERCISE 1

`X` is $(120, 4096)$ and `W` is $(4096, 64)$.

1. Shape of `X @ W`?
2. Shape of `W.T @ X.T`?
3. Shape of `X @ X.T`? Which of the three is dissimilarity-matrix-shaped?

<!--
4 min. Answers: (120, 64); (64, 120); (120, 120). The third — square, and
indexed by stimuli on both sides.

Push one step after revealing: what does (1) mean? "Each of 120 images
described by 64 numbers." What does (2) mean? The same thing, transposed —
which is why the transpose error is invisible.
-->

---

# A spectrum is a curve, not a number

`pca.explained_variance_ratio_` is one number per component, sorted, decreasing.

```python
plt.plot(np.cumsum(pca.explained_variance_ratio_))
plt.axhline(0.9, ls=':')     # a threshold you must choose and defend
```

- falls off a **cliff** → genuinely low-dimensional
- decays **slowly** → variance spread across many directions

<!--
4 min. R02 gave them what an eigenvector is; this is the part they need for
Part 3, and it is only about how to READ the output.

Draw both curve shapes on the board. The cliff and the slow decay look
completely different and lead to completely different claims, and students
tend to report only the single number at 90%.
-->

---

# The trap in "90% of the variance"

<mark>Out of how many possible components? On how many <em>items</em>? Are those items independent?</mark>

51 objects × 24 viewpoints = 1,224 images.

Rotating one object traces a **smooth curve** through feature space: many more
points, hardly any new directions.

The representation looks compact — and the number moves if you change the
number of viewpoints while the network stays byte-for-byte identical.

<!--
4 min. Plant this hard. Part 3 of the assignment is built on it and it is the
most common way a student produces a confident wrong claim.

The moral to state: dimensionality is a property of the data AND the stimulus
set, never of the network alone.

Do NOT give away the numbers they will get — that is theirs to find.
-->

---

# Fitting a curve: the two calls

```python
c    = np.polyfit(x, y, deg=1)     # polynomial
yhat = np.polyval(c, x)
```

```python
from scipy.optimize import curve_fit

def expo(d, a, b):  return a * np.exp(-b * d)
p, _ = curve_fit(expo, d, s, p0=[1.0, 1.0])
yhat = expo(d, *p)
```

`curve_fit` does **local** search — it needs a starting guess `p0`, and it can land in a bad minimum.

<!--
4 min. Emphasize p0. When a fit comes back absurd, the first move is a
different starting guess, not a conclusion about the data. They will hit this.

Note that they write the model function themselves — which means they have to
have decided what shape they believe the relationship has. That decision is the
science; curve_fit is the arithmetic.
-->

---

# Three ways an $R^2$ misleads you

$$R^2 = 1 - \frac{\sum_i (y_i - \hat{y}_i)^2}{\sum_i (y_i - \bar{y})^2}$$

1. **It is a ratio.** Bin your data first and $R^2$ rises with the curve unchanged.
2. **More parameters never fit worse.** That is arithmetic, not evidence.
3. **It can be negative** — a fit worse than predicting the mean. Real outcome, not a bug.

<!--
4 min. R02 covered point 1; do it fast and spend the time on point 2, which is
new and which they need for the three-model comparison in Part 2.

Point 2 in one sentence: if you compare a 2-parameter law against a
3-parameter law and the 3-parameter one wins, you have learned nothing. Compare
models with the same number of free parameters, or say plainly that you did not.
-->

---

# EXERCISE 2

You fit an exponential and get $R^2 = 0.84$. A paper reports $0.96$ for the
same law on the same kind of data.

Name **two** things that could differ, neither of which means their model is better.

<!--
4 min. Answers: (i) they binned/averaged before fitting, shrinking the
denominator; (ii) they fitted a model with more free parameters. Also fine:
they averaged over subjects first, or their stimulus set had more spread in y.

The point to land: a published number is a number computed under choices, and
you cannot compare against it until you know the choices. This comes up
directly in the assignment.
-->

---

# Pearson or Spearman?

- **Pearson** — how close is the relationship to a straight line?
- **Spearman** — rank both, then Pearson. Is the relationship **monotonic**?

```python
from scipy.stats import pearsonr, spearmanr
rho, p = spearmanr(a, b)
```

<mark>Comparing two RDMs uses Spearman, and the reason is substantive, not cosmetic.</mark>

<!--
4 min. Give the reason properly: a network's distances and a human's similarity
judgments are on unrelated scales, and there is no reason to expect a linear
relation. The question you actually want is "do the two systems ORDER the pairs
the same way?" — that is Spearman's question.

Pearson would answer a different question and report a lower number for a reason
that has nothing to do with the models. Students who use it will conclude the
network is a worse match than it is.
-->

---

# Never correlate the whole matrix

An RDM is symmetric with a zero diagonal — more than half of it is duplicates.

```python
iu = np.triu_indices(D1.shape[0], k=1)    # k=1 skips the diagonal
rho, p = spearmanr(D1[iu], D2[iu])
```

<!--
4 min. Draw the matrix and shade the upper triangle. k=1 vs k=0 is worth one
sentence: k=0 includes the diagonal, which is 120 guaranteed zeros in both
matrices and will inflate any correlation.
-->

---

# The $p$-value here is wrong — say so

7,140 pairs, but only **120 stimuli**.

Move one stimulus and a whole row and column move with it. The pairs are **not independent**, so the $p$-value's assumption fails.

<mark>Report the correlation and the number of stimuli. Do not report that $p < 10^{-9}$.</mark>

The honest fix is a permutation test over stimuli — Recitation 8.

<!--
4 min. This is a research-methods point, not a course technicality, and it is
one they can carry into a lab. Say that.

If asked "so is the correlation meaningless?" — no. The correlation is a fine
descriptive statistic. It is the significance claim that is unsupported, and
the distinction between those two is the thing to learn.
-->

---

# `random_state`: reproducible ≠ meaningful

```python
Y = TSNE(n_components=2, perplexity=30, random_state=0).fit_transform(X)
```

MDS and $t$-SNE start from a random initialization. Two runs, two pictures.

- Setting the seed makes it **reproducible** — same picture tomorrow.
- It does **not** make it meaningful — you picked one member of a family and hid the rest.

<!--
4 min. This is the conceptual centre of the session, so do not rush it.

The analogy that lands: quoting one run of a random algorithm is like reporting
one participant and calling it an experiment. Reproducible, and still not a
result.
-->

---

# The test to apply

<mark>Rerun with three seeds. Structure that appears in all three is a candidate finding. Structure that appears in one is a property of that seed.</mark>

Reporting the second kind is an error — a perfectly reproducible one.

<!--
3 min. Short slide on purpose; it is the takeaway, and Part 4 of the assignment
is exactly this exercise.

Ask the room: how many seeds is enough? There is no correct number — the point
is that one is definitely not enough, and that you say in the write-up how many
you ran.
-->

---

# What a $t$-SNE map is not

Three things students read off the picture anyway:

- **Distances between clusters are not distances.**
- **Cluster sizes are not sizes** — sparse regions expand, dense ones compress, by design.
- **Apparent clusters are not evidence of clusters** — it will produce clean blobs from data with none.

What it *is* good for: seeing whether points you **already have labels for** land together.

<!--
5 min. Spend the time. Every one of these three appears in published figures.

perplexity in one sentence: roughly how many neighbors each point tries to stay
near — small preserves local detail, large preserves coarse structure. Sweeping
it changes the picture substantially, which is the point of the sweep in Part 4.
-->

---

# EXERCISE 3

A $t$-SNE map shows two clean, widely separated clusters.

Give **two** explanations — one a real finding, one not — and the **single check** that separates them.

<!--
4 min. Answers: real = the two groups genuinely occupy separate regions of the
high-dimensional space. Not real = t-SNE manufactured the separation at this
perplexity, or this seed.

The check: rerun across several seeds AND several perplexities and see whether
the split survives. Accept "compare against distances in the original space" as
an even better answer, and use it to introduce the next slide.
-->

---

# What two dimensions cost

Any method that puts high-dimensional data on a page throws something away. Ask **what**.

```python
hi = np.argsort(D_hi, axis=1)[:, 1:k+1]      # skip self at position 0
lo = np.argsort(D_lo, axis=1)[:, 1:k+1]
overlap = [len(set(hi[i]) & set(lo[i])) / k for i in range(n)]
```

One number **per point** — so you can color the map by it and see which regions you are allowed to trust.

<!--
4 min. Neighborhood preservation. The `1:k+1` slice is worth pointing at: every
point is its own nearest neighbor at position 0, and forgetting that shifts
every answer by one.

The last line is the good idea: a single average is a summary, but a per-point
map tells you WHERE the embedding lied. That is a better figure than any
average.
-->

---

# When it breaks

1. **Last line of the traceback first** — it names the error.
2. **Then the last line that is *your* file** — the frames below are library code.
3. **Print shapes, not values.**
4. **`np.isnan(X).sum()`** — one `nan` poisons a whole matrix silently.
5. **Shrink the input to 5 rows.** Bug survives → inspect by hand. Bug vanishes → it is about scale.

<!--
4 min. Read these out; do not elaborate. It is a checklist to photograph, and
they will use it on Sunday night.

Note for step 4: correlation distance produces nan for any constant row — a
dead unit — which is exactly why the assignment makes them look for those.
-->

---

# Where this lands in Assignment 1

- **Part 2** — fitting three laws and comparing $R^2$; Spearman on the upper triangle
- **Part 3** — reading a spectrum, and the "out of how many items?" question
- **Part 4** — the perplexity sweep, seeds, and what two dimensions cost

Everything else today is a habit, not a step.

<!--
2 min. Close by pointing at handout-03.pdf for the prose version, at office
hours for anything assignment-specific, and at the fact that the next session
(Week 4) starts the calculus you will need for Assignment 2 — derivatives,
gradients, and the chain rule. Nothing in today's session carries forward to it.
-->
