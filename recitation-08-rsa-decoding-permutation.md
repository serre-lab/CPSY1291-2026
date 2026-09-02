---
marp: true
theme: cpsy1291
paginate: true
math: katex
---

<!-- ===========================================================================
  CPSY 1291 — RECITATION 8: Comparing representations
  TA-led, 80 minutes. Optional. Week 8 — hold this after Lecture 12.
  Assignment 4 is due Tue 11/10.

  SCOPE: the machinery behind RSA and decoding, and the three statistical traps
  that make published versions of both analyses wrong. Lecture 12 supplies the
  motivation and the science; this supplies the code and the caveats.

  THE DEBT: Recitation 3 promised that the meaningless RSA p-value would be
  fixed by a permutation test "in Recitation 8". This is that session. Say so
  explicitly — students remember being told something was owed.

  TIMING: the permutation-test block (slides 8-11) is the core. If you run
  short, cut CKA and shorten the decoding block; do not cut permutation or the
  noise ceiling.

  MATERIAL: handout-08-rsa-decoding-permutation.pdf, with six exercises.
============================================================================ -->

<div class="eyebrow">Recitation 8 · Week 8</div>

# Comparing representations

## RSA, decoding, and an honest $p$-value

<!--
2 min. Open by paying the debt: five weeks ago we said the p-value from
correlating two RDMs was meaningless and that a permutation test would fix it.
Today is the fix.

Frame the session: Lecture 12 gave you the science. This is the code, plus the
three ways these analyses go wrong in print.
-->

---

# The one idea

<mark>A model–brain similarity score means nothing on its own.</mark>

It only means something against two other numbers:

- what score **chance** would produce
- what score a **perfect** model could produce, given the noise

Compute both. Report all three.

<!--
4 min. This is the frame for the whole session, and it is the single most
portable idea in it.

Give the concrete version now so it lands: a model scoring 0.35 against a
ceiling of 0.40 is doing very well; the same 0.35 against a ceiling of 0.85 is
doing poorly. The raw number cannot tell those apart.
-->

---

# RSA, end to end

```python
def rdm(X, metric='correlation'):
    return squareform(pdist(X, metric=metric))     # (n, n)

def rsa(D1, D2):
    iu = np.triu_indices(D1.shape[0], k=1)
    return spearmanr(D1[iu], D2[iu]).statistic
```

Six lines, and **four decisions** buried in them — each of which changes the answer.

<!--
6 min. Name the four: which distance; whether to standardize the units first;
Spearman or Pearson; and the stimulus set.

Say why correlation distance is the field's default for neural data: it removes
each stimulus's overall response level, which is often driven by attention or
arousal rather than by stimulus identity. Euclidean keeps it. Neither is neutral
and you have to say which you used.
-->

---

# The stimulus set *is* the experiment

40 animals and 40 tools. Almost any representation that separates animate from
inanimate will score well.

<mark>The score is then telling you about your stimulus set, not about your model.</mark>

**The question to ask of any RSA result:** what would a model that knows *only*
the coarsest category distinction score? If it is about the same, you have
distinguished nothing.

<!--
6 min. This is the most important methodological slide of the session, and it is
not in the lecture.

Make it vivid: two models that disagree about everything can produce nearly
identical RSA scores on a stimulus set with one dominant division. The
literature has this problem, and it is why stimulus-set design gets as much
scrutiny as model design in careful papers.
-->

---

# A permutation test, in nine lines

```python
obs  = rsa(D1, D2)
null = np.empty(n_perm)
for i in range(n_perm):
    p = rng.permutation(n)                  # shuffle STIMULI
    null[i] = rsa(D1, D2[np.ix_(p, p)])     # rows and columns together
p_value = (np.sum(null >= obs) + 1) / (n_perm + 1)
```

The logic: *if there were no relationship, how often would I see a score this
large?* Destroy the relationship 10,000 times and look.

<!--
6 min. Walk through the logic before the code. A p-value from a formula is an
answer to this question computed under assumptions; a permutation test answers
it by direct simulation, and works when the assumptions do not.

np.ix_(p, p) is the line to point at: it reindexes rows AND columns with the
same permutation, which keeps a valid dissimilarity matrix.
-->

---

# Two details that *are* the test

**Shuffle stimuli, not pairs.** Permuting the $n(n-1)/2$ triangle entries
destroys the matrix structure too, so the null is far too narrow and
<mark>everything comes out significant</mark>.

**Add one to both sides.** With 10,000 permutations the smallest honest $p$ is
$1/10001$, not $0$.

<!--
6 min. The first point is the one that produces wrong papers. If a student's
permutation test says every model including a random one is significant, this is
why, every time.

The +1 has a reason worth giving: your observed value is itself one possible
arrangement, so it belongs in the null count. Reporting p = 0 claims you have
proved something impossible with 10,000 samples.
-->

---

# Plot the null, not just the $p$

A histogram of the 10,000 null scores with your observed value marked tells you
more than the number it produces.

<mark>A null distribution that is a narrow spike near zero means you permuted the wrong thing.</mark>

<!--
5 min. This is the diagnostic that catches the previous slide's error, and it
takes two lines.

General habit worth naming: whenever a procedure produces a single summary
number, look at the distribution it came from at least once. That applies to
p-values, to cross-validation scores, and to loss curves.
-->

---

# The noise ceiling

Split your subjects in half. Correlate one half's RDM with the other's.

<mark>That is roughly the best any model could do — no model can predict the noise.</mark>

- $0.35$ against a ceiling of $0.40$ → very good
- $0.35$ against a ceiling of $0.85$ → poor

Raw scores cannot tell those apart.

<!--
6 min. Say why this changes conclusions and not just presentation: model
rankings in this literature sometimes REVERSE when the ceiling is added,
because different datasets have different ceilings.

Counterintuitive point worth making: a high noise ceiling makes your models look
WORSE. Clean data is harder to explain, and that is honest rather than
unfortunate.
-->

---

# EXERCISE 1

Model A scores $0.31$. Model B scores $0.29$. Same data, same stimuli.

What **two** further numbers do you need before saying A is better?

<!--
6 min. Answers: (1) the noise ceiling — is 0.02 large relative to the achievable
range? (2) A confidence interval on the DIFFERENCE, obtained by bootstrapping
stimuli with replacement.

The second is the one they will not say. Emphasize: bootstrap the difference,
not the two scores separately — overlapping intervals on two quantities do not
imply their difference is uncertain.
-->

---

# A confidence interval, by resampling stimuli

```python
boot = []
for _ in range(1000):
    idx = rng.integers(0, n, n)                  # with replacement
    boot.append(rsa(D1[np.ix_(idx, idx)], D2[np.ix_(idx, idx)]))
lo, hi = np.percentile(boot, [2.5, 97.5])
```

Permutation asks *is it above chance?* Bootstrap asks *how precisely do I know
it?*

<!--
5 min. Two different questions, two different resampling schemes — permutation
shuffles WITHOUT replacement and destroys the relationship; bootstrap samples
WITH replacement and preserves it.

Students conflate these constantly. Say the distinction in those terms and it
sticks.
-->

---

# Decoding: a decoder is a classifier

```python
clf = make_pipeline(StandardScaler(), LinearSVC(C=1.0))
cv  = StratifiedKFold(n_splits=5, shuffle=True, random_state=0)
scores = cross_val_score(clf, X, y, cv=cv)
```

<mark>The `make_pipeline` is not stylistic. Standardizing outside the cross-validation fits the scaler on the data you are about to test on.</mark>

<!--
5 min. Same leak as Recitation 6, in its most common disguise — and it is
invisible, because the code looks tidy and the numbers look better.

Tell them the size of it: a few percentage points, which is exactly the size of
effect people publish.
-->

---

# Three ways a decoding result is wrong

1. **Chance is not always 50%.** With imbalanced classes, "always answer the
   majority" scores the majority's proportion. 90% on a 90/10 split is *nothing*.
2. **Trial structure leaks.** Same trial, run, or block on both sides of a split
   → the classifier recognizes the trial. Use `GroupKFold`.
3. **Decodable is not used.** Linearly readable ≠ the organism reads it.

<!--
6 min. Spend the time; all three appear in published work.

Point 3 is the interpretive limit of the entire method and belongs in any
write-up that uses it. The clean way to say it: decoding tells you what
information is PRESENT, not what the brain DOES with it.

Point 1's fix: report balanced accuracy or the confusion matrix, ideally both.
-->

---

# EXERCISE 2

A classifier decodes stimulus category from **V1** at 78%, chance 50%.

Your advisor asks whether V1 "represents category". Answer in two sentences.

<!--
6 min. Model answer: category information is linearly readable from V1's
population response under these conditions. That does not establish that V1
represents category — low-level features correlated with category in this
stimulus set would produce the same result, and nothing here shows that the rest
of the brain uses this information.

This is the single most useful thing in the session for anyone going into a lab.
Let the room argue about it.
-->

---

# CKA, for when you want a second opinion

```python
def cka(X, Y):
    X, Y = X - X.mean(0), Y - Y.mean(0)
    num = np.linalg.norm(X.T @ Y, 'fro') ** 2
    return num / (np.linalg.norm(X.T @ X, 'fro') * np.linalg.norm(Y.T @ Y, 'fro'))
```

Invariant to rotation and isotropic scaling, **not** to arbitrary invertible maps.

<mark>Running two metrics on the same data tells you how much your conclusion depends on the metric. Often: a lot.</mark>

<!--
6 min. Both centerings are per-unit (column). Forgetting them inflates the score
toward 1 for almost any pair of matrices — a good bug to warn about, because the
result looks like a strong finding.

Callback to Lecture 5: the metric zoo disagrees with itself, and that is a live
problem in the field rather than a technicality. Running two is cheap.
-->

---

# Where this lands

- **Assignment 4** — RSA between CNN layers and neural data, decoding, and the honest $p$
- **Your project** — every one of these traps is one you can fall into in three weeks
- **Any lab you join** — the noise ceiling and `GroupKFold` are working knowledge

Next week: looking **inside** a network — hooks, gradients, and saliency.

<!--
3 min. Point at handout-08.pdf, especially the cheat sheet and the six
exercises.

Preview next week honestly: today was about comparing a network's
representations from the outside; next week is about opening it up, and the
methods for doing that are much less trustworthy than the ones today.
-->
