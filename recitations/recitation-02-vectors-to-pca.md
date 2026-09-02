---
marp: true
theme: cpsy1291
paginate: true
math: katex
---

<!-- ===========================================================================
  CPSY 1291 — BOOTCAMP RECITATION 2 of 2: from vectors to PCA
  TA-led, 80 minutes. Optional. Assumes Recitation 1 or equivalent numpy.

  SCOPE: reconciled against the FINAL Assignment 1 (2026-09-01). Students are
  deep in parts 2-3 this week. The LECTURES own the concepts (psychological
  space, Shepard's law, the ventral stream) — this deck points at them in one
  slide and spends its time on the technical side: the library call sheets
  (curve_fit, polyfit/polyval, spearmanr, sklearn, silhouette_score), the 2g
  fitting pipeline worked end-to-end on toy data, R^2 by hand, the
  groupby-without-groupby template, and the four skeletons the TAs' review
  flagged as blockers (2j, 3c, 3f, 3i). Every worked example uses toy data,
  never the assignment's data. Do NOT teach how to derive PCA, how MDS
  optimizes, or anything about training networks. Those belong to later
  lectures and later recitations.

  REGISTER: mixed room, no linear algebra prerequisite. Every abstraction gets
  a picture and a worked number. Students are not being asked to implement
  these methods — they are being asked to know what goes in, what comes out,
  and what could go wrong.
============================================================================ -->

<div class="eyebrow">Bootcamp · Recitation 2 of 2</div>

# From vectors to PCA

## The methods the rest of Assignment 1 leans on

<!--
2 min. Set the frame explicitly: part 1 is behind them, parts 2 and 3 are this
week, and A1 is due Monday 9/29. This session front-loads the ideas part 2
assumes and the library mechanics parts 2-3 use, then hands out the code
skeletons for the four sections that the test-solve showed stop people cold.

Reassure the room that nobody will be asked to derive an eigendecomposition.
They will be asked what a principal component IS, which is different and more
useful.
-->

---

# Where you are

Part 1 built the machinery: three distances, twelve RDMs, the axioms tested on real data.

Parts 2 and 3 use it to do **science**: human similarity and Shepard's law, then PCA and the shape of object space.

<mark>Today: the library calls parts 2–3 use, one fitting pipeline worked end to end, and the code skeletons for the four sections most likely to stall you.</mark>

<!--
2 min. One-slide map of the session. If someone has not started part 2, tell
them everything today still lands — the examples use toy data, and the
skeletons are printed in the handout so nothing needs transcribing.
-->

---

# Concepts you'll lean on (the lectures own these)

- **Psychological space** *(Lecture 2)* — ratings behave as if stimuli sit at points in an internal metric space; MDS recovers the map. Why 2e–2f are a claim, not a plot.
- **Shepard's law** *(Lecture 2)* — generalization falls off like $e^{-d}$; the Gaussian is the rival, and they differ **near $d=0$**. What 2g and 2j actually test.
- **Depth ↔ ventral stream** *(Lecture 3)* — early layers ↔ early areas, deep layers ↔ IT. Why 2k tracks depth, and why part 3 puts fc6 next to IT recordings.

<mark>Today is the technical side: the library calls, the pitfalls, the skeletons.</mark>

<!--
2 min. A pointer slide, not a lesson — the lectures teach these; do not
re-teach them here. Concept questions go to lecture and office hours; the
notebooks' glossary cells carry the vocabulary.

Three sentences worth saying aloud while pointing. One: 2e is not "make a
scatter plot" — it is the claim that an internal space exists, and 2f tests it
by putting the images on the recovered map. Two: near d = 0 the exponential is
already plunging while the Gaussian is still flat — that left-hand stretch is
where 2g question 1 sends them. Three: "generalization" in part 2 is the
BEHAVIORAL sense (a pigeon trained on one tone pecks at nearby tones too), not
the ML train/test sense — students with ML exposure mix it up anyway.

Recitation 3 picks up the Bao et al. cortical-maps question when they are
inside part 3.
-->

---

# The identity doing quiet work in part 2

<mark>Pearson correlation IS cosine similarity, after subtracting each vector's own mean.</mark>

```python
Xc = X - X.mean(axis=1, keepdims=True)   # center each ROW
# cosine of Xc == correlation of X
```

The provided helpers (`correlation_similarity` in 2b, `correlation_distance` in 2j) z-score each row instead — the extra division cancels inside the cosine, so they compute **exactly your 1b function**.

<!--
4 min. Kept from part 1 because it comes back twice this week. The 2j notebook
text says the helper "computes exactly Part 1's measure" — this identity is
how a student can VERIFY that claim rather than take it on faith: z-scoring is
centering plus dividing by the std, and after the cosine's own normalization
the std cancels.

It is also how they can tell they reproduced Peterson et al.'s method (inner
products of z-scored features — which IS correlation) rather than something
that merely resembles it.

If asked why "each row": because we compare STIMULI, and each stimulus's
vector gets centered on its own mean.
-->

---

# Condensed vs square — the error you will hit

$n \times n$ numbers to store $n(n-1)/2$ facts, so SciPy has both forms:

```python
d = pdist(X, metric='euclidean')    # CONDENSED: (n*(n-1)/2,)
D = squareform(d)                   # SQUARE:    (n, n)
squareform(D, checks=False)         # and back
```

`linkage` wants **condensed** — 2h passes it `squareform(D_human, checks=False)`. `imshow` wants **square**. `MDS(metric='precomputed')` wants **square**.

<!--
3 min. Purely practical, and it will save several hours across the class.

When a scipy call complains about a distance matrix, this is nearly always the
reason. The 2h hint hands them the squareform(D_human, checks=False) call;
this slide is why it is there. checks=False because D_human's off-diagonal
was built from ratings and squareform's symmetry check is stricter than it
needs to be. 2g's provided cell also uses pdist(X_human) directly — condensed
is exactly the "each pair once" order that s_ij = S_human[iu] matches.
-->

---

# Clustering: build a tree, then cut it

```python
Z   = linkage(squareform(D_human, checks=False), method='average')
grp = fcluster(Z, 6, criterion='maxclust')
```

The tree holds structure at **every** level at once.

<mark>Choosing where to cut is choosing which level you want to describe — not discovering how many clusters there "really" are.</mark>

<!--
3 min. Draw a small dendrogram and a horizontal cut line, then slide the line
up and down.

Concrete version: cut a set of animals into 4 groups and you get roughly
mammal / bird / reptile / amphibian; cut into 8 and you get primates,
carnivores, hoofed mammals separately. Neither is wrong. 2h has them cut at 6;
2i then cuts at k = 2, 4, 6, 8 and scores each cut against `taxon` and
`kingdom` with adjusted_rand_score — and the two label sets peak at DIFFERENT
k, which is this slide's point made measurable.
-->

---

# Comparing two RDMs

```python
iu  = np.triu_indices(120, k=1)          # each pair once, no diagonal
rho = spearmanr(D1[iu], D2[iu]).statistic
```

Using the whole matrix counts every pair **twice** and adds $n$ meaningless zeros — which inflates the correlation for no reason.

<mark>`.statistic` — that is where the number lives. 2k builds its whole table from this one line.</mark>

<!--
2 min. Recitation 1 taught D[iu]; this adds the call around it. Ask before
revealing: "why not correlate the two full matrices?" Someone gets the
double-counting; the 120 diagonal zeros are the half people miss, and the more
damaging half. (7,140 pairs go in, not 14,400.)

spearmanr returns a result object — .statistic is the correlation, .pvalue the
p-value 2k tells them not to lean on (the pairs are not independent). Spearman
rather than Pearson because only the ORDERING of dissimilarities is meaningful.
-->

---

# `curve_fit` and `polyfit` — the call sheet

```python
from scipy.optimize import curve_fit
popt, _ = curve_fit(f_exp, x, y, p0=[10, 1], maxfev=40000)
y_hat   = f_exp(x, *popt)          # * unpacks the fitted parameters back in
```

- `p0` — start near the right scale: similarities run 0–10, so $a \approx 10$
- `maxfev` — raise it, or hard fits die with "Optimal parameters not found"
- wrap in `try/except RuntimeError` when a fit is *allowed* to fail (2j says so)

The straight line has its own pair:

```python
coef  = np.polyfit(x, y, 1)        # fit:      slope and intercept
y_hat = np.polyval(coef, x)        # evaluate: the line at every x
```

<!--
4 min. This is 2g's toolbox as a call sheet; the next slide runs it end to end
on toy data. The notebook defines f_exp and f_gauss and says WHICH tool to
use; this slide is HOW to call them, which the notebook deliberately does not
spell out.

Three things students otherwise learn the hard way: curve_fit returns a TUPLE
(parameters first, covariance second — hence the popt, _ unpacking); the
fitted parameters go back into your own function with a *, nothing is
"applied" automatically; and the defaults assume an easy problem — p0 and
maxfev are how you tell it otherwise. On 2j's untrained layers the exponential
may legitimately not converge even then, which is why the notebook says
try/except — and calls that failure a finding, not a bug.

polyfit/polyval: fit and evaluate are separate calls, same pattern. Keep the
patter brief here — every line gets typed and run on the next slide.
-->

---

# WORKED — 2g's pipeline, end to end on toy data

```python
rng = np.random.default_rng(0)
d = np.sort(rng.uniform(0, 4, 60))                  # toy "distances"
y = 10 * np.exp(-1.3 * d) + rng.normal(0, 0.5, 60)  # toy "similarities"

coef    = np.polyfit(d, y, 1);   y_lin = np.polyval(coef, d)             # line
popt, _ = curve_fit(f_exp, d, y, p0=[10, 1]);  y_exp = f_exp(d, *popt)   # exponential

print(r2(y, y_lin), r2(y, y_exp))            # 0.71 vs 0.97
plt.scatter(d, y); plt.plot(d, y_lin); plt.plot(d, y_exp)
```

Make data → fit both models → score both with the **same** $R^2$ → look at the plot.

<mark>This four-move shape is 2g exactly — swap the toy arrays for `(d_ij, s_ij)` and you are done.</mark>

<!--
6 min. Type and run this live, top to bottom — it is the technical spine of 2g
and, inside a loop, of 2j. f_exp is the notebook's own definition
(a * exp(-b * d)); r2 is the 2g setup cell's helper, and the NEXT slide opens
it up by hand.

Three beats while it runs. One: we KNOW the truth here — a = 10, b = 1.3 — and
curve_fit hands back about (10.05, 1.28); fitting toy data with a known answer
is how you convince yourself a pipeline works before pointing it at real data.
Two: both models are scored by the same r2 against the same y — 0.71 for the
line, 0.97 for the exponential — never compare fits scored against different
targets. Three: the plot is the check the numbers cannot give — the line goes
negative where similarities never can, and you only see that by looking.

If someone asks about the Gaussian: same curve_fit call with f_gauss, and 2g
has them do exactly that.
-->

---

# EXERCISE — R² by hand

$R^2 = 1 - \dfrac{\sum_i (y_i - \hat{y}_i)^2}{\sum_i (y_i - \bar{y})^2}$ — the fraction of the variance the curve accounts for.

A fit predicts $\hat{y} = (3, 4, 5, 8, 10)$ where the data say $y = (2, 4, 6, 8, 10)$.

1. Compute $R^2$.
2. What would a model that predicts $\bar{y}$ everywhere score?
3. Can $R^2$ be negative?

<!--
6 min. Do this one slowly; Thomas wants R^2 actually worked in recitation, not
gestured at. Answers:

(1) residuals (-1, 0, -1, 0, 0), so the top sum is 2. Mean of y is 6;
deviations (-4, -2, 0, 2, 4), squares (16, 4, 0, 4, 16), bottom sum 40.
R^2 = 1 - 2/40 = 0.95.

(2) Exactly 0 — the two sums are then identical. R^2 = 0 does not mean
"broken"; it means "no better than a horizontal line through the mean."
That baseline is the whole scale.

(3) Yes: predict WORSE than the mean (e.g. yhat = y reversed) and the top sum
exceeds the bottom. Negative R^2 = worse than knowing nothing but the average.

1 = perfect, 0 = the flat line, negative = worse than that. The 2g setup cell
defines r2(y, yhat) with this exact formula — they are not asked to write it,
they are asked to READ it, and now they can.
-->

---

# The R² you compute vs the R² they published

It is a **ratio**: it depends on how variable $y$ was to begin with.

Average your data into bins first and $R^2$ rises — with the fitted curve completely unchanged.

<mark>Two $R^2$ values are comparable only if computed against the same target.</mark>

<!--
3 min. This is 2g question 2, almost verbatim: they fit 7,140 individual pairs
and get about 0.8-something; published gradients fit BINNED averages and
report 0.95+. The difference is the denominator of the exercise they just did
— binning removes single-pair noise from the bottom sum, nothing about the
model improved.

The provided running_mean() in 2g is a guide for the eye and explicitly NOT
the fitting target; that comment is this slide in one line.
-->

---

# The scikit-learn pattern

```python
est = PCA(n_components=50)
Y   = est.fit_transform(X)
est.explained_variance_ratio_    # what it learned: trailing underscore
```

Same shape of call for `PCA`, `MDS`, `TSNE`.

Three things to watch:
- the trailing underscore is **not a typo** — it marks attributes that exist only after `.fit`
- some take a **dissimilarity matrix**, not features → `metric='precomputed'`
- some are **random** → two runs differ unless you set `random_state`

<!--
4 min. Once they see the pattern they can read documentation for any of these,
which matters more than memorizing arguments.

The trailing underscore is 3b's exact attribute — explained_variance_ratio_ —
and a student who types it without the underscore gets an AttributeError
before fitting or a subtly different thing with other estimators. Convention:
settings go in the constructor, data goes to .fit, learned results grow an
underscore.

The random_state point is not housekeeping — part 4 is entirely about what
changes between runs and what does not. Flag it as a preview, not a lesson.
-->

---

# What PCA is doing

Center the data (PCA does this for you), form the covariance, take its eigenvectors:

$$\boldsymbol{C}\boldsymbol{v} = \lambda\boldsymbol{v}$$

<mark>They are the directions along which the data varies most, and each $\lambda$ says how much.</mark>

So **PC1** is simply the single direction in which your points are most spread out.

<!--
4 min. Draw a tilted elongated cloud; its long axis is PC1, the perpendicular
short axis PC2. No algebra beyond the statement on the slide.

Two footnotes that pay off this week: PCA centers each FEATURE for you — 3b's
REMEMBER says so, and it is the axis=0 cousin of the row-centering in the
correlation identity (same verb, different axis, different purpose). And the
components are DIRECTIONS in feature space; 3d asks them to argue PC1
corresponds to a shape property, and that argument is impossible if a
component is just a column of output.
-->

---

# Variance explained, and the question to ask

```python
np.cumsum(est.explained_variance_ratio_)
```

"50 components explain 85% of the variance" sounds like a fact about the network.

<mark>Out of how many possible components? On how many items? Are those items independent?</mark>

$n$ items can never need more than $n-1$ components — raw counts are not comparable across sets of different size; **fractions of the maximum** are.

<!--
3 min. Plant this hard — 3c is built on it. The set-up: 1,224 images that are
51 objects seen from 24 angles each. Rotating one object traces a smooth
low-dimensional curve, so the extra images add points without adding
directions; the data therefore looks far more "compact" than it is.

The n-1 bound is 3c's own hint (51 items can never need more than 50
components), and it is why 3c demands each count as a raw number AND as a
fraction of the maximum. Do not give away the numbers — those are theirs to
find.
-->

---

# Groupby without groupby — THE template

One line, four moves:

```python
means = np.stack([F[obj == k].mean(axis=0) for k in range(51)])   # (51, 4096)
```

1. `obj == k` — boolean mask: which rows belong to object $k$
2. `F[obj == k]` — those rows: `(24, 4096)`
3. `.mean(axis=0)` — collapse the 24 views: `(4096,)`
4. `np.stack([...])` — pile the 51 results into one array

<mark>"Per-group anything" is this template with a different word in step 3.</mark>

<!--
5 min. Taught once, as THE pattern — walk the four moves on the board and do
not let it read as a one-liner trick. Steps 1-2 are recitation 1's boolean
mask; step 3 is the axis rule; step 4 is the only new word.

Where it pays: 3c needs exactly this line (the 51 object means). 3h's
per-category silhouette scores loop the same way over categories. And in part
4 it counts t-SNE islands per object:
[len(np.unique(isl[obj == k])) for k in np.unique(obj)] — same mask, different
statistic. When they see a "for each object / category / cluster" sentence in
an assignment, this is the shape of the answer.
-->

---

# Skeleton: 3c's variance split

How much of the variance is *identity*, and how much is *viewpoint*?

```python
total   = F.var(axis=0).sum()          # variance per feature, summed: one number
between = means.var(axis=0).sum()      # variance OF the 51 object means
within  = total - between              # what viewpoint contributes
```

`means` is the template from the last slide. Each part is a single number; report the split.

<!--
4 min. The second half of 3c, spoon-fed as the review recommended — the
notebook's hint says "the between-object part is the variance of the 51
object means, and the within-object part is what is left of the total"; this
is that hint as three lines of code.

Say what each line is in words: total = how spread out all 1,224 points are,
summed over the 4,096 features; between = how spread out the 51 object
CENTERS are; within = the spread of views around their own center, obtained
by subtraction. The interpretation question 3c asks (is "50 dimensions at
85%" a fact about the network or the stimulus set?) belongs to them, not to
this slide.

The three PCA counts in 3c's first half need no skeleton — cumsum plus the
boolean argmax from recitation 1, three times, on F, on means, on F[::24].
-->

---

# Skeleton: 2j's loop

Six fits — three layers, two networks — all reusing your 2g code:

```python
for tag, a in (('trained', act_tr), ('untrained', act_un)):
    for L in ('early', 'middle', 'late'):
        d_net = correlation_distance(a[L].astype(np.float64))[iu]
        # ... your 2g fitting code, on (d_net, s_ij)
```

- `[iu]` puts the network distances in the **same pair order** as `s_ij`
- wrap the exponential fit in `try/except` — on untrained layers it may **legitimately fail to converge**, and that failure is a finding, not a bug

<!--
4 min. The structure is the blocker, not the content — every piece already
exists (correlation_distance is provided in 2j, iu and s_ij come from 2g's
setup), and this loop is just plumbing them together. Students who stall here
stall on "where do I even start", so hand them the start.

The two bullets are the two sentences worth saying out loud. Pair order: iu
always walks the upper triangle the same way, so d_net[i] and s_ij[i] describe
the SAME pair of images — that is the whole reason the fit is meaningful.
try/except: the notebook's NOTE says this in so many words; a student who
treats the convergence failure as their own bug will burn an evening on it.

2j also asks for the printed distance ranges and fitted parameters (question 2
needs them) and fixed y-limits on the plot — read the TODO list in full.
-->

---

# Skeleton: 3f's perimeter, handed out entire

Spikiness = perimeter² / area, computed from the silhouette mask alone:

```python
def perimeter(m):
    return ((m[1:, :] != m[:-1, :]).sum()     # edges between vertical neighbors
          + (m[:, 1:] != m[:, :-1]).sum())    # edges between horizontal neighbors

spik = np.array([perimeter(m)**2 / m.sum() for m in masks])
o    = np.argsort(spik)                        # imgs[o[:8]], imgs[o[-8:]]: the gallery
```

Shift the mask against itself; count where the two copies **disagree** — every disagreement is a piece of boundary.

<!--
4 min. Given away whole, on purpose: 3f's point is EARNING the name of an axis
(correlate log spikiness with PC1), not image processing, and the shifted-
comparison trick is not something one reinvents mid-assignment. Draw a tiny
5x5 mask and its shifted copy; the != cells trace the outline.

Piece by piece: m[1:, :] vs m[:-1, :] compares every pixel with the one below
it — a True next to a False is a horizontal stretch of boundary. Same
sideways. Area is m.sum() from recitation 1 (True counts as 1). The argsort
line is the sanity gallery 3f requires — blobs at one end, spiky things at the
other, or the measure is wrong.

What stays theirs: take the LOG (the raw index is heavily skewed — the
notebook says so), then Spearman against PC1 and PC2, and 3f question 2's
point about 1,224 images vs 51 objects.
-->

---

# Skeleton: 3i's NaNs — pick one, and say which

Real electrodes drop samples. PCA accepts none of it. Both of these are defensible:

```python
IT = it_raw[:, ~np.isnan(it_raw).any(axis=0)]                       # DROP affected units
IT = np.where(np.isnan(it_raw), np.nanmean(it_raw, axis=0), it_raw) # IMPUTE unit means
```

<mark>Ordering rule for step 4: check for zero-variance units **before** z-scoring. Dividing by a std of 0 makes NaNs, and one NaN poisons everything downstream.</mark>

<!--
4 min. The anatomy first, for anyone who has never met an ephys dataset:
electrode -> spikes -> a firing rate per stimulus. Rows are still stimuli,
columns are now recorded units — same convention as everything else this
month. Baseline rates differ tenfold across neurons for boring reasons
(electrode placement, cell size), which is exactly why 3i step 4 asks for the
PCA again with each unit z-scored: without it, PCA partly reports which
electrodes were LOUD.

The two one-liners: drop reads "keep the columns where no entry is NaN" —
recitation 1's boolean machinery, ~ and any(axis=0). Impute reads "where it
is NaN, take that unit's nanmean, else keep the value". The notebook accepts
either and demands only that they SAY WHICH — the habit of stating an
analytical choice out loud is the actual lesson.

The ordering rule is the trap that survives both choices: z-scoring a
zero-variance unit is 0/0. The IT array has no fully dead units (the notebook
says so) but 2c showed the untrained network has hundreds — the rule is
general, and cheap: variance check first, then normalize.
-->

---

# `silhouette_score` — what goes in

```python
from sklearn.metrics import silhouette_score
silhouette_score(Y[:, :2], (cat == c).astype(int))
```

First argument: the **2-D coordinates** in the PC1–PC2 plane. Not the RDM.

Second: a binary labeling — this category against everything else. Near 1 = sits apart; near 0 = on a boundary; negative = closer to another group than its own.

<!--
2 min. One call, one common mistake. 3h asks for a silhouette score per
category, for fc6 and for pixels, so the comparison is a number rather than
an impression — the per-category loop is the groupby template again, with
silhouette_score as the statistic.

The first-argument mistake is worth naming because everything else this month
took a distance matrix; silhouette_score (as called here) takes the
coordinates themselves and computes its own distances.
-->

---

# Two warnings about the clock

- **4b runs t-SNE twelve times** (4 perplexities × 3 seeds). That takes **minutes**, not seconds. Do not interrupt the cell — 4c and 4d reuse the dictionary it builds.
- **Restart-and-run-all before submitting re-runs everything**, the t-SNE sweep included. Budget for it; do not start it five minutes before the deadline.

<!--
2 min. Pure logistics, and the difference between a calm submission and a
panicked one. The failure mode: a student interrupts the "hung" cell, loses
embeds[(perplexity, seed)], and every later cell in part 4 breaks in ways that
look unrelated.

While on logistics: part 2's setup cell upgrades scikit-learn on a fresh Colab
runtime — the first run of that cell takes an extra half-minute, and that is
expected, not a hang.
-->

---

# Where this lands in Assignment 1

| Today | Where it shows up |
|---|---|
| the concept pointers (lectures 2–3) | reading 2e–2g and 2k as science, not plotting |
| condensed vs square, the tree | clustering (2h, 2i) |
| `iu` + `spearmanr(...).statistic` | the alignment table (2k) |
| `curve_fit`, `polyfit`/`polyval`, R² | testing Shepard's law (2g, 2j) |
| sklearn pattern, variance explained | PCA on fc6 (3b, 3c) |
| the groupby template + skeletons | 2j, 3c, 3f, 3i |

One loop to close: 1f asked where the choice of distance could change a conclusion — 2e is an answer, because **MDS trusts the triangle inequality** you showed can fail.

<!--
2 min. Read the table out; every row carries a section label they can go find
tonight.

The closing callback is worth thirty seconds: they measured triangle
violations in 1f, and MDS quietly assumes a geometry where none exist — the
assignment plants the question and this is where it pays. (It returns once
more in 4e, where the nonmetric-MDS row is exactly "dissimilarities you trust
only ordinally.")

What recitation 3 carries, before the 9/29 deadline: a working clinic on
parts 3-4, the Bao et al. cortical-maps story (category patches: modules or
landmarks on a smooth shape map), and part 4's method fingerprints. Then move
straight into the buffer slide — do not let this one absorb its minutes.
-->

---

# Questions / catch-up

- Anything from today you want run again, slower — say so now
- Or open your own part 2 and get unstuck **while the room is full of help**
- Not covered today ≠ not on A1: `bootcamp-notes.pdf` and office hours carry the rest

<!--
3 min. A deliberate buffer, budgeted on purpose — do not fill it with new
material. If the room is silent, re-run the worked pipeline with a bad p0
(say p0=[0.001, 50]) and let them watch the "Optimal parameters not found"
failure from the call-sheet slide actually happen, then fix it.

Close with logistics: A1 is due Monday 9/29; restart-and-run-all before
submitting re-runs the t-SNE sweep, so budget for it. Office hours and
bootcamp-notes.pdf for everything else.
-->
