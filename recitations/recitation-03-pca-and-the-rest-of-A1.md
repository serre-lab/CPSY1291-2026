---
marp: true
theme: cpsy1291-recitation
paginate: true
math: katex
---

<!-- ===========================================================================
  CPSY 1291 — RECITATION 3: PCA, and the rest of Assignment 1
  TA-led, 80 minutes. Optional. Hold this the evening of the PCA lecture,
  one week BEFORE Assignment 1 is due (Tue 9/29).

  HOW TO RUN THIS DECK: every slide's presenter note gives a timing, what to
  say, and (where relevant) the answer to the exercise on the slide. Slides
  marked EXERCISE are done live: put the prompt up, let them try for 2-3
  minutes, then take answers from the room before revealing. The companion
  notebook follow-along-03.ipynb has the same sections in the same order;
  the room works in the notebook while you talk to the slides.

  SCOPE: Recitation 1 was Python and numpy. This session assumes nothing
  else. First half: PCA made concrete, because the lecture is dense and the
  assignment leans on it. Second half: the pipelines the rest of Assignment
  1 needs (variance explained, naming an axis, groupby, curve fitting and
  R^2, comparing RDMs), plus the knobs of t-SNE at the end if time allows.

  DO NOT: work any part of Assignment 1 on the board. Every example here uses
  the toy world in the notebook, on purpose.
============================================================================ -->

<div class="eyebrow">Recitation 3</div>

# PCA, and the rest of Assignment 1

## The lecture's four steps, in code, then every pipeline the assignment needs

<!--
2 min. Say the frame: Assignment 1 is due next Tuesday. Today's lecture was
PCA, and it was dense; the first half of tonight is that lecture again, as
code you can run, on data small enough to draw. The second half is the rest
of what the assignment needs.

Open follow-along-03.ipynb in Colab now (File > Upload notebook) and run the
first cell. Everything tonight is in it; nobody needs to take notes.

Set the expectation out loud: nothing from the assignment will be worked on
the board. Every example uses a toy world with five objects, deliberately.
Point at office hours for assignment-specific questions.
-->

---

# Where you are

**Have:** arrays and shapes, `axis=`, boolean masks, broadcasting, dot products (Recitation 1).

**Heard today:** centre, covariance, eigenvectors, project. Variance along an axis is an eigenvalue.

**Need by Tuesday:** PCA in scikit-learn, "how many components", what an axis means, per-object means, curve fitting and $R^2$, comparing two RDMs, and what a knob does to a method.

<mark>Tonight: each of those is one cell you can run, on data where we planted the answer.</mark>

<!--
2 min. Read the three lines; the third is the checklist for the session.

The planted-answer line is the method of the whole notebook: five toy objects,
three views each, six units, and one fact we put in ourselves (object size
moves the response along a fixed direction). When a method recovers what we
planted, we trust it enough to point at real data.

If someone missed Recitation 1: point at the follow-along-01 notebook and
keep going. Backing up costs the whole session.
-->

---

# The toy world

`F`: a `(15, 6)` **stimuli × units** matrix. Five objects, three views each, six units.

Planted: object **size** runs 1 to 5, and the response moves along one fixed direction as size grows.

```python
obj  = np.repeat(np.arange(5), 3)        # which object each row shows
size = np.array([1., 2., 3., 4., 5.])    # one number per object
F    = proto[obj] + view_jitter          # (15, 6)
```

One row is one stimulus's **response vector**: a point in a space with one axis per unit.

<!--
3 min. Run section 1 with them. Say the row/column convention once and
never again: rows are stimuli, columns are units, and every matrix in the
assignment has that shape.

The response-vector-as-point picture is what the next four slides draw. If
that picture is not in the room, nothing after it lands.
-->

---

# Step 0: pick two units and look

Two units that co-vary most, drawn as a scatter: one point per stimulus, one axis per unit.

The cloud is **tilted**. The two units rise and fall together across stimuli.

<mark>That tilt is what PCA finds: the axis the cloud actually lies along, which is neither unit.</mark>

<!--
3 min. Run the cell; the scatter appears with size as colour. Point at the
tilt. Ask: if you could describe each point with ONE number instead of two,
which direction would you measure along? Someone will point along the long
axis. That is PC1, and they have just done PCA by eye.

Say why two units: six units cannot be drawn, and a method you cannot draw
you cannot check. We go to six once the picture is solid.
-->

---

# Steps 1 and 2: centre, then covariance

```python
Xc = X2 - X2.mean(axis=0)          # each unit's mean becomes 0
C  = Xc.T @ Xc / (n - 1)           # (2, 2): the covariance matrix
```

$C$ holds each unit's **variance** on the diagonal and how the two **co-vary** off it.

The off-diagonal entry is the tilt as a number.

<!--
4 min. Run it and read the printed matrix. The diagonal equals the two
variances (the cell prints them separately to prove it). The off-diagonal is
negative here: as unit 4 goes up, unit 5 goes down. Sign does not matter for
what follows; the size of it does.

Two things to say. Centring is axis=0: per column, per unit. And Xc.T @ Xc is
every pair of units dotted over stimuli: "summed over stimuli", which is why
the result is units x units. Name the summed index out loud; it is the habit
that catches transposes.
-->

---

# Step 3: eigenvectors are the axes of the cloud

```python
eigenvalues, eigenvectors = np.linalg.eigh(C)
```

- `eigh` returns eigenvalues **ascending** and eigenvectors as **columns**: flip both so the largest is first
- the first eigenvector is a **unit vector** pointing along the cloud's long axis
- its eigenvalue is the **variance** along that axis; the two eigenvalues sum to the total variance

<!--
5 min. Run the cell; the arrows appear on the cloud. PC1 lies along the long
axis, PC2 across it. Point at the printed check: total variance equals the
sum of the eigenvalues, so nothing was lost, the variance was re-divided
between two new axes.

Two mechanics they will trip on: eigh sorts ascending (the cell flips), and
eigenvectors come out as columns (column k is eigenvector k). Both are on
the slide because both cost time when met alone.

If asked what eigh does: for a symmetric matrix, it finds the directions the
matrix only stretches, and by how much. The lecture drew that; tonight we
only use it.
-->

---

# Step 4: a coordinate on PC1 is a dot product

```python
v1         = eigenvectors[:, 0]
pc1_scores = Xc @ v1               # 15 dot products at once
```

Each stimulus's PC1 coordinate is its centred response vector dotted with the first eigenvector.

And `pc1_scores.var(ddof=1)` **equals the first eigenvalue**: that is the sense in which PC1 is "the direction of largest variance".

<!--
4 min. Run it. The printed variance of the scores equals the first
eigenvalue to three decimals. Say the sentence on the slide slowly; it is the
one line that connects the lecture's algebra to a number they can compute.

Xc @ v1 is (15, 2) @ (2,) -> (15,): the shape rule from Recitation 1. One
dot product per stimulus, summed over units.
-->

---

# EXERCISE 2.1: the second component, by hand

Compute `pc2_scores` (every stimulus's coordinate on the **second** eigenvector) and `var_pc2`, their variance with `ddof=1`.

Then compare the two variances: how much of the cloud's spread does PC1 alone carry?

<!--
4 min. Let them try; then reveal. Answer: pc2_scores = Xc @ eigenvectors[:, 1];
var_pc2 = pc2_scores.var(ddof=1); it equals the second eigenvalue. The
check prints the share: PC1 carries about 97% of the variance of these two
units.

The point to land: two numbers per stimulus were nearly one number all
along. That is the whole promise of PCA, stated on data where you can see
it.

Common miss: eigenvectors[1] (a row) instead of eigenvectors[:, 1] (a
column). The check message says so.
-->

---

# From two units to six: let scikit-learn do the four steps

```python
est = PCA()                    # settings in the constructor (defaults here)
Y   = est.fit_transform(F)     # centres for you, then projects
est.components_                # the eigenvectors, stored as ROWS
est.explained_variance_        # the eigenvalues
```

The notebook computes the six-unit covariance and its eigenvectors by hand and checks: **same eigenvalues, same directions**.

<mark>Signs can differ. An eigenvector and its negative are the same axis; PC1 and −PC1 are equally valid.</mark>

<!--
5 min. Run the cell; the two routes agree. Say what fit_transform did: the
same four steps, on six units, where nobody can draw the cloud.

Three things to say once. The trailing underscore marks what the estimator
learned; it exists only after .fit. components_ stores directions as rows,
so projecting by hand needs a transpose (section 8 of the notebook triggers
that error on purpose, at home). And signs: two machines can disagree on
the sign of a component and both be right; the assignment grades
accordingly, and every check in the notebook compares absolute values.
-->

---

# How many components? `cumsum`, then the first `True`

```python
cum = np.cumsum(est.explained_variance_ratio_)   # running total
np.argmax(cum >= 0.85)                           # index of the first True
```

`argmax` of a boolean array returns the **first `True`**. It returns an **index**; the count of components is index + 1.

"Two components explain 85%" sounds like a fact about the network.

<mark>Out of how many possible components? On how many items?</mark>

<!--
5 min. Run section 3. The scree plot appears with the 85% line. Then the
five-object-means cell: five items, and the fifth eigenvalue is exactly
zero whatever the network did. n items can never need more than n - 1
components, so raw counts are not comparable across sets of different
size; fractions of the maximum are. The assignment asks for both; do not
give away its numbers.

The off-by-one is the most common error of the week. Say it twice.
-->

---

# EXERCISE 3.1: components to reach 95%

Compute `k95`, the **number of components** (a count, not an index) whose cumulative explained variance first reaches 0.95, on the full `F`.

<!--
3 min. Answer: k95 = np.argmax(cum >= 0.95) + 1, which is 3 here. If a
student gets 2, they forgot the + 1; the check message says so.
-->

---

# Naming an axis is a hypothesis; a correlation is the test

The PC1–PC2 plane: five clumps of three, marching along PC1 in **size order**.

To test the name you hang on an axis, correlate its scores with the property, per stimulus:

| property | test | call |
|---|---|---|
| continuous (size) | rank correlation | `spearmanr(Y[:, 0], size[obj]).statistic` |
| binary (round or not) | point-biserial | `pointbiserialr(is_round[obj], Y[:, 0]).statistic` |

<!--
5 min. Run section 4. The plane plot first: five clumps, ordered by size
along PC1, because we planted size. Then the two correlations: strong for
size (planted), weak for roundness (not planted). The assignment asks the
same question of two published axis names on real data; the test is the
same call.

Two mechanics: .statistic is where the number lives (the result is an
object), and size[obj] gives one size per stimulus, matching Y's rows;
size alone is one per object and the shapes will not match.
-->

---

# EXERCISE 4.1: does PC2 know about size?

Compute `rho_size_pc2`, the Spearman correlation between the **PC2** scores and the stimuli's sizes.

Predict first: strong or weak? PC2 is perpendicular to PC1, and PC1 already carries size.

<!--
3 min. Answer: weak (about 0.1 in absolute value against 0.97 for PC1).
The prediction is the point: perpendicular axes share nothing, so what PC1
captured, PC2 cannot. The check compares against PC1's correlation.
-->

---

# Groupby without groupby: the template

```python
means = np.stack([F[obj == k].mean(axis=0) for k in range(5)])   # (5, 6)
```

1. `obj == k`: boolean mask, which rows belong to object $k$
2. `F[obj == k]`: those rows, `(3, 6)`
3. `.mean(axis=0)`: collapse the 3 views, `(6,)`
4. `np.stack([...])`: pile the 5 results into one `(5, 6)` array

<mark>"For each object / category / cluster, compute ..." is this line with a different word in step 3.</mark>

<!--
4 min. Run section 5. Steps 1-2 are Recitation 1's masks; step 3 is the
axis rule; np.stack is the only new word. When the assignment says "per
object" or "per category", this is the shape of the answer.
-->

---

# EXERCISE 5.1: the template with a different word

Compute `pc1_spread`: for each of the 5 objects, the **standard deviation** of its 3 views' PC1 scores (`Y[:, 0]`). Shape `(5,)`.

It says how far viewpoint moves an object along PC1: the **within**-object spread the assignment asks you to separate from the **between**-object spread.

<!--
3 min. Answer: np.array([Y[obj == k, 0].std() for k in range(5)]). Small
numbers (views jitter by 0.35) against clumps several units apart. The
within/between split is section 10a of the notebook, at home, and it is
the assignment's question about identity versus viewpoint.
-->

---

# Fitting a curve: four moves

```python
popt, _ = curve_fit(f_exp, d, s, p0=[10, 1], maxfev=40000)   # fit
y_exp   = f_exp(d, *popt)                                     # evaluate, * unpacks
r2(s, y_exp)                                                  # score, same target
```

- `curve_fit` returns a **tuple**, parameters first
- the parameters go back into **your** function with a `*`
- `p0` starts at the scale of the data: ratings run 0–10, so $a \approx 10$
- score every model with the **same** `r2` against the **same** target, then **look at the plot**

<!--
6 min. Run section 6. We planted s = 10 exp(-0.8 d) plus noise, so the
recovered a and b land near 10 and 0.8: the pipeline works, and now they
know it. The plot shows what the numbers cannot: the straight line goes
negative at large distances, where a rating never can.

R^2 in one sentence: the fraction of the mean-predictor's error the curve
removes; 1 perfect, 0 no better than the mean, negative worse. It is a
ratio, so it depends on how variable the target was: two R^2 values are
comparable only against the same target. The assignment asks about
published values in exactly those terms.
-->

---

# EXERCISE 6.1: the rival law

Shepard's exponential has a rival, the Gaussian `f_gauss`, flat near $d = 0$ where the exponential already plunges.

Fit it with the same `curve_fit` call, evaluate it, and score it into `r2_gauss`, same scorer, same target.

<!--
3 min. Answer: popt_g, _ = curve_fit(f_gauss, d, s, p0=[10, 1],
maxfev=40000); r2_gauss = r2(s, f_gauss(d, *popt_g)). Line < Gaussian <
exponential, because the exponential is what we planted. The assignment
asks the same question of data where nobody planted anything.
-->

---

# Comparing two RDMs: upper triangles, then Spearman

```python
iu  = np.triu_indices(15, k=1)                    # each pair once, no diagonal
rho = spearmanr(D[iu], D_early[iu]).statistic
```

The whole-matrix shortcut counts every pair twice and shares 15 diagonal zeros: **free agreement** that inflates the number.

<mark>`iu` on both matrices, always. Spearman = Pearson on ranks: only the ordering is trusted.</mark>

<!--
5 min. Run section 7. The two printed numbers differ: whole matrices
inflate. Then the RDM picture: five dark 3x3 blocks, the five objects.
Always look at an RDM before computing on it.

The p-value that spearmanr returns is not to be leaned on: the pairs are
not independent samples. The assignment says so.
-->

---

# EXERCISE 7.1: which half carries the geometry?

Build `D_late`, the RDM from the **last three units** (`F[:, 3:]`), and `rho_late`, its Spearman correlation with the full `D`, upper triangles only.

Which three units carry more of the full geometry?

<!--
3 min. Answer: D_late = squareform(pdist(F[:, 3:])); rho_late =
spearmanr(D[iu], D_late[iu]).statistic, about 0.95 against about 0.48 for
the first three units. The two halves of this little network do not carry
the geometry equally, and the assignment builds a whole table from this one
line.
-->

---

# A method with a knob gives you a family of results

t-SNE starts from a random layout and has a **perplexity** knob.

- two runs without a seed: two different maps of the same data
- the same `random_state`: the same map
- two perplexities: two different pictures, both "correct"

<mark>What survives across the family is the finding. What changes is the knob.</mark>

<!--
4 min if time allows; otherwise point at section 9 of the notebook, at
home. Three maps of the same 15 stimuli: the five clumps survive; their
positions, distances and orientation do not. That distinction is the
assignment's last question in one sentence.
-->

---

# Where this lands in Assignment 1

| you did tonight | the assignment asks |
|---|---|
| centre, covariance, eigenvectors, project | PCA on a network's responses, and on neurons |
| `cumsum` and the first `True`; the $n-1$ bound | how many components, and what that count is a fact about |
| correlate an axis with a property | test two published axis names |
| the groupby template | per-object and per-category anything |
| `curve_fit`, `*popt`, `r2` on the same target | Shepard's law, and published $R^2$ values |
| upper triangles, then Spearman | networks against people, layer by layer |

<!--
3 min. Read the left column as the recap. At home: section 8 (components_
and the transpose error), 9 (the knobs), 10 (the variance split, missing
values, a loop over fits that is allowed to fail).

Office hours for anything assignment-specific. Bring the failing cell open,
with its shapes already printed.
-->

---

# Questions / catch-up

Bring up the cell that resisted you. Print its shapes first.

<!--
Remaining time. Work from the room's questions. If none: run section 8 of
the notebook together (the transpose error, triggered on purpose), which is
the single most common PCA error of the week.
-->
