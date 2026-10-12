# 03 — Partial / outlier-aware correspondence & registration

*Research track: how to make "this block tissue has no counterpart on the slide"
an explicit, measurable output instead of something the registration absorbs.*

Toy experiment: [`scripts/toy_partial_matching.py`](../../scripts/toy_partial_matching.py)
(plus `scripts/toy_partial_matching_figure.py`), results in `results/toy_partial_matching/`.

---

## 1. Summary

* **Plain CPD fails here because of how it is built, not because it was tuned badly.**
  CPD is a GMM fit where every data point has to be explained, the mixing weights are
  fixed at 1/M, there is one global isotropic σ², and the outlier term is a uniform
  density with a fixed weight *w*. Missing tissue is a spatially coherent block of
  "outliers", and that model has no way to represent it. The cheapest explanation is
  to inflate σ² and bend the smooth displacement field. Section 2 has the math.
* **The fix is to use the asymmetry of the problem: slide ⊆ block.** Every slide
  pixel should land on block tissue; block tissue may receive nothing. Any method that
  (i) fixes the slide-side marginal or coverage and (ii) leaves the block side free
  or bounded produces the missing-tissue map directly: block tissue that receives no
  mass. Methods that do this:
  - **directed (one-sided) chamfer / distance-transform registration**, slide → block
    only, then a residual coverage map;
  - **partial OT** (POT `ot.partial.partial_wasserstein`), with transported mass
    bounded by the slide's share;
  - **partial / semi-relaxed Gromov–Wasserstein** (POT `ot.gromov.*`), which needs no
    alignment and is invariant to rotation and flip;
  - **fragment-level assignment** (Hungarian or min-cost flow with a dummy "unmatched"
    node, plus many-to-one merging for torn pieces).
* **Toy result (section 5): 80 synthetic biopsy cases, paired with and without a
  removed fragment.** Plain CPD reproduces the reported failure (figure in section 5):
  - CPD bends neighbouring fragments over the gap. It reports a median of only
    **22 %** of the missing area, and its localisation IoU is **0.14**.
  - Piecewise-rigid directed-chamfer coverage reports **100 %** of the missing area,
    with localisation IoU **0.90** and AUROC **0.93** (0.96 without partial-face cases).
  - CPD's own outlier posterior (1 − Pt1) and the reversed-role CPD are close to
    useless (AUROC 0.62–0.67).
  - Partial OT does **not** beat plain coverage when both use the same pose. Its value
    is a calibrated area-unit score.
  - Partial GW reaches the same accuracy as rigidly aligned partial W *without any
    alignment* (AUROC 0.84, IoU 0.52).
  - Semi-relaxed GW collapses (AUROC 0.64–0.68).
  - The main remaining confounder is "partial face" (a fragment present but 80 %
    size). Every method that does not absorb it flags it, and that sets
    sensitivity at 95 % specificity.
* **Recommended design (section 4).** Fragment-aware, asymmetric, rigid-first.
  1. **Pipeline A**: global similarity plus flip search on masks with a directed
     chamfer, then per-slide-piece rigid refinement, then a coverage deficit map.
     This is the baseline.
  2. **Pipeline B**: partial OT or partial FGW on area-weighted samples, using the
     block marginal deficit. This is the principled score and is robust to the choice
     of alignment.
  3. **Pipeline C**: fragment graph matching with dummy nodes. This gives
     per-fragment explanations and handles torn or merged fragments.

  Scores are "largest connected unmatched area" (not total), so small fragments are
  not drowned out.
* **Learned RGB matchers** (LoFTR, LightGlue, RoMa, XFeat) are unlikely to work from
  wax photo to H&E at the texture level. MatchAnything is the only one trained for
  cross-modality, and it is worth a quick test for *pose initialisation* only. None of
  these matchers outputs "unmatched area" in a calibrated way. **Histology toolkits**
  (VALIS, DeeperHistReg, ANHIR/ACROBAT winners) assume the same tissue on both sides
  and have no missing-tissue output. They are not the right tool, but their
  rigid-then-non-rigid, mask-based preprocessing is reusable.

---

## 2. Why CPD fails (the math)

CPD (Myronenko & Song 2010) treats the moving set **Y** = {y_m}, m = 1..M, as GMM
centroids and the fixed set **X** = {x_n}, n = 1..N, as data:

$$
p(x_n) \;=\; w\,\frac{1}{N} \;+\; (1-w)\sum_{m=1}^{M}\frac{1}{M}\,\mathcal N\!\left(x_n \mid \mathcal T(y_m),\,\sigma^2 I\right)
$$

The E-step posterior is

$$
P_{mn}=\frac{\exp\!\left(-\tfrac{\|x_n-\mathcal T(y_m)\|^2}{2\sigma^2}\right)}
{\sum_{k}\exp\!\left(-\tfrac{\|x_n-\mathcal T(y_k)\|^2}{2\sigma^2}\right)+c},
\qquad c=(2\pi\sigma^2)^{D/2}\frac{w}{1-w}\frac{M}{N}.
$$

The non-rigid M-step minimises

$$
Q(W,\sigma^2)=\sum_{m,n}P_{mn}\frac{\|x_n-y_m-G(m,\cdot)W\|^2}{2\sigma^2}
+\frac{N_P D}{2}\log\sigma^2+\frac{\lambda}{2}\operatorname{tr}(W^\top G W),
\qquad G_{ij}=e^{-\|y_i-y_j\|^2/2\beta^2},
$$

and the variance update is $\sigma^2 = \frac{1}{N_P D}\sum_{mn}P_{mn}\|x_n-\mathcal T(y_m)\|^2$.

Take the usual set-up, slide = Y (moving) and block = X (data), with a block fragment F
that has no slide counterpart. Five things go wrong.

1. **Each data point must be explained by an inlier unless c is large.** Row
   normalisation across the M centroids means $\sum_m P_{mn} + c/(\cdot) = 1$. A block
   point in F has no nearby centroid. Its posterior goes either to the outlier term or,
   spread thinly, to the nearest centroids. Those nearest centroids get a gradient that
   pulls them toward F.
2. **The outlier term is uniform and fixed, and its strength is coupled to σ².**
   $c \propto \sigma^{D}\,w/(1-w)$. Early in EM σ² is large (it is initialised to the
   mean squared distance between all pairs), so every Gaussian reaches F. By the time
   σ² shrinks, the field has already moved. The outlier model also has no notion of a
   *coherent region* of outliers: each point is independently "1/N uniform". A whole
   fragment of outliers is improbable under this model, so EM prefers to deform.
3. **There is one global isotropic σ².** The residuals of F inflate σ² for everyone.
   That in turn widens every Gaussian's reach, so more of F gets explained, and the
   loop reinforces itself.
4. **Coherence prior (motion coherence theory).** The displacement field is
   $v = GW$, a Gaussian-kernel RKHS function regularised by $\operatorname{tr}(W^\top G W)$.
   For moderate β it is cheap to translate a *whole neighbourhood* of centroids toward
   F. Covering F costs a smooth stretch, which the prior hardly penalises, while
   leaving F uncovered costs a likelihood penalty on every point of F. The prior
   penalises *roughness*, not *coverage of tissue that isn't there*.
5. **Equal mixing weights 1/M force the centroids to be used.** If you swap the roles
   (block = moving centroids, slide = data, so that the empty block fragment shows up
   as low $P1_m=\sum_n P_{mn}$), you get the opposite problem. Unused components waste
   mixture mass, so the log-likelihood increases if the empty centroids are dragged
   onto data. CPD cannot learn that a component has zero weight.

   **BCPD** (Hirose 2021) fixes this with learnable mixing weights $\alpha_m$ under a
   Dirichlet prior. It also estimates σ² variationally, though σ² is still global, and
   its outlier weight is still a fixed uniform *w*. In BCPD the value $\langle\alpha_m\rangle$
   (or the expected match count $\nu_m$) is a usable per-centroid "matched mass" output.
   It is worth trying, but the coherence prior still smooths through gaps.

Taken together: CPD is effectively a *one-sided* (data-normalised) soft assignment
with a smoothness prior. It is a balanced model, and nothing in the objective bounds
how much mass each centroid may attract. Partial OT puts a *capacity* on both sides,
**and** caps the total mass. That is the missing ingredient.

**Mitigations that remain inside the CPD family**, in order of usefulness:

* Rigid or similarity CPD with a high *w* first, and only then a constrained,
  large-β non-rigid stage.
* Re-weighting: fix w per point, or use the "missing-data" CPD variants.
* Run CPD **per fragment** after fragment assignment, so there is nothing for the
  field to bridge.
* Read the result from *post-hoc coverage* (block points farther than τ from every
  warped slide point), never from the posterior.

The toy experiment shows these remain clearly worse than the alternatives below.

---

## 3. Method catalogue

Legend: **Inv.** = invariance to rotation (R), flip (F), scale (S). **Unmatched
output** = whether "missing tissue" is a native, measurable output. Wheel status was
checked on **Python 3.14 / Windows** with `pip download --only-binary=:all:` (October 2026).

| Family | Method | Inv. | Unmatched output | Handles tears / many-to-one | Win/Py3.14 | Verdict |
|---|---|---|---|---|---|---|
| GMM registration | CPD (rigid/affine/non-rigid), `pycpd` | none (needs init) | weak: 1 − Pt1 is unreliable (§2) | non-rigid forces cover | `pycpd` 2.0.0 pure-Python ✔ | Baseline only; do not use its posterior |
| | BCPD / BCPD++ (Hirose) | similarity is built in, flip is not | ⟨α_m⟩ and ν_m per point (better) | partially | C binary (ohirose/bcpd) with Win build; no pip; `probreg` has **no wheel** ✘ | Worth a check via CLI; still smooths over gaps |
| | TPS-RPM (Chui & Rangarajan) | none | **yes**: explicit outlier row/column in softassign | TPS bends | implement (~100 LOC) | Historically the "correct" robust version of CPD |
| Robust rigid | RANSAC / TEASER++ / trimmed ICP | R (+F by search) | trimmed residual = unmatched | per-fragment ICP | OpenCV, SciPy ✔ | Good first stage |
| | **Directed chamfer / DT registration on masks** | R, F (search), S (search or fixed) | **yes**: coverage deficit | per-piece refinement | OpenCV, SciPy ✔ | **Core of Pipeline A** |
| | ECC on masks (`cv2.findTransformECC`) | local only | via residual | no | ✔ | Refinement only; symmetric objective, so missing tissue biases it |
| Optimal transport | **Partial Wasserstein** (`ot.partial.partial_wasserstein`, Chapel 2020) | none (needs alignment) | **yes**: block marginal deficit $p_i-\sum_j T_{ij}$ | yes (mass can split) | **POT 0.9.7 cp314 win wheel ✔** | **Core of Pipeline B** |
| | Unbalanced Sinkhorn (KL marginals; `ot.unbalanced.sinkhorn_unbalanced`) | none | yes (soft), but τ (KL weight) trades off with scale | yes | ✔ | Softer and differentiable; needs τ calibration |
| | Robust OT for registration (RobOT, Shen 2021) | none | yes | yes | GPU-oriented (KeOps) | Ideas only |
| Gromov–Wasserstein | **Partial GW** (`ot.gromov.partial_gromov_wasserstein`) | **R, F, translation** (intrinsic distances) | **yes** | yes | ✔ (slow, O(N³)-ish; N ≤ 500) | No alignment needed; local minima |
| | **Semi-relaxed GW** (Vincent-Cuaz 2022) | R, F, T | yes: free block marginal | yes | ✔ | Exact "slide ⊆ block" model; in practice collapses mass (toy) |
| | Unbalanced / fused UGW (Séjourné 2021) | R, F, T | yes (soft) | yes | ✔ `fused_unbalanced_gromov_wasserstein` | Promising, more hyperparameters |
| | **Fused GW** (Vayer 2019) with mask-derived node features (local thickness, distance to boundary, curvature) | R, F | with partial variant (`partial_fused_gromov_wasserstein`) | yes | ✔ | **Best cross-modal candidate**: shape-only features avoid the appearance gap |
| Fragment level | Hungarian (`scipy.optimize.linear_sum_assignment`) with dummy nodes at cost θ | per descriptor | **yes**: assignment to dummy = missing fragment | no (1-to-1) unless merged | ✔ | Simple; brittle to fused/torn fragments |
| | Min-cost flow / fragment-level partial OT (area as mass) | per descriptor | yes: unmatched area per fragment | **yes** | ✔ (POT / networkx) | **Core of Pipeline C** |
| | Graph matching on fragment layout (spectral / FGM / GW on centroids) | R, F | dummy nodes | partially | implement | Biopsy pieces float on the water bath, so the layout is weakly preserved; use as a soft term |
| Shape descriptors | Hu moments (`cv2.HuMoments`) | R, F (sign of h7), S | via thresholded distance | no | ✔ | Weak for blobs; cheap prefilter |
| | Elliptic Fourier (Kuhl & Giardina), Fourier on centroid distance | R, (F), S | threshold | no | `pyefd` pure-Python | Good for smooth contours |
| | Shape context (Belongie 2002), inner-distance SC (Ling & Jacobs 2007) | R (with care), S | **dummy-node matching built in** | partial matching possible | implement or OpenCV `ShapeContextDistanceExtractor` ✔ | Robust to articulation and tears; supports partial matching |
| | Turning function (Arkin 1991), area / perimeter / solidity / eccentricity | R, F, S | threshold | no | ✔ | Cheap fragment-level features |
| Learned matchers | SuperPoint+LightGlue, LoFTR, RoMa, XFeat, DISK | trained R-ish | **no**: unmatched area is not calibrated | n/a | `kornia` (LoFTR, LightGlue on CPU) ✔ | Unlikely to work on wax vs H&E RGB; masks have no texture |
| | MatchAnything (He 2025; ELoFTR/RoMa trained on synthetic cross-modal pairs, tissue slices among unseen domains) | yes | no | n/a | torch CPU, slow | Worth a 20-case test for **pose initialisation** only |
| Histology toolkits | VALIS (rigid feature + non-rigid), DeeperHistReg, ANHIR/ACROBAT winners | R, F (VALIS has flip search) | **no**: they assume the same tissue | n/a | VALIS needs pyvips/JVM (painful on Windows) | Not built for missing tissue; borrow their mask preprocessing and rigid-then-non-rigid design |

### Notes on specific items

* **Partial W vs unbalanced Sinkhorn.**
  - Partial OT has one interpretable knob, the transported mass m. Set m to the
    slide's expected share of block mass: area ratio × shrinkage prior. The plan is
    exact and sparse, and the deficit $d_i=p_i-\sum_j T_{ij}$ is directly in area
    units.
  - Unbalanced Sinkhorn replaces the hard marginals with KL penalties ρ. Its deficit
    depends on both ρ and the cost scale, so it is harder to calibrate, but it is
    smooth and fast.
* **Partial GW is the only option that needs no registration and is invariant to
  flips.** That matters because sections are often mounted mirrored. Its weakness is
  that biopsy fragments float independently, so *inter*-fragment distances are not
  preserved. Fix this by computing intrinsic distances **within** each fragment
  (∞ or capped across fragments: a "fragment-blocked" C), or by using FGW with
  per-point shape features.
* **Semi-relaxed GW** is exactly the "slide ⊆ block" model on paper. In practice its
  free marginal lets many slide points pile onto a few block points, because nothing
  penalises over-density. Add an upper bound on the block marginal (capacity), which
  turns it back into partial GW, or use entropic srGW.
* **Hungarian with dummies**: for a block with fragments $B_1..B_K$ and slide pieces
  $S_1..S_L$, build a $(K+L)\times(L+K)$ cost matrix. Real–real entries hold the shape
  cost. Real–dummy entries hold a constant θ (the "price of declaring unmatched").
  Dummy–dummy entries are 0. A block fragment assigned to a dummy is "missing". This
  is 1-to-1 only, so do many-to-one by greedily merging adjacent slide pieces (union
  of masks) when that lowers cost, or use min-cost flow with fragment areas as
  capacities.
* **Learned matchers on silhouettes.** Running SuperPoint or LoFTR on binary masks
  gives keypoints only along the boundary, with ambiguous descriptors. A distance
  transform or signed-distance image as input helps a little, but classical
  boundary descriptors (shape context, turning function) do the same job with known
  invariances. For the RGB route, the useful shared signal is the *dark ink/cautery
  margins and fat holes*, which are sparse. That is a long shot.

---

## 4. Recommended pipelines

All three pipelines share the same front end and the same score definitions.

**Shared front end**

* `Mb`, `Ms`: tissue masks for block and slide. Block segmentation is the weak link;
  see the segmentation track.
* `components(M)`: connected components.
* `sample(M, h)`: a jittered grid with spacing h inside M, so every point carries
  equal area.
* Scale `s`: the slide-pixels-per-block-pixel factor. Estimate it as a cohort median
  of the area ratio or bounding-box ratio, then refine it with a bounded search
  (for example ±15 %). Never estimate it from the total area of the *current* pair
  alone, because missing tissue biases that estimate.

**Scores, computed from a per-block-pixel unmatched map U ∈ [0,1]**

* `missing_area = Σ U · pixel_area / area(Mb)`
* `largest_blob = area(largest connected component of U > .5) / area(Mb)`. This is
  the primary score: it keeps small fragments visible.
* `per_fragment_unmatched[k] = mean(U on block fragment k)`. Flag if any value
  is > 0.6 and the fragment's area is ≥ a_min.
* Localisation = U itself, or the flagged fragments' contours.

### Pipeline A — asymmetric rigid/piecewise-rigid coverage (baseline, fast, interpretable)

```python
def pipeline_A(Mb, Ms, s):
    DTb = distance_transform(~Mb)                 # block distance map
    S   = sample(Ms, h*s) / s                     # slide points in block units
    # 1) global pose: search flip × rotation grid × translation grid, minimise
    #    DIRECTED chamfer  mean(DTb[T(S)]^2)  (slide→block only!)
    T0  = argmin_{flip, θ, t, s'∈s·[0.9,1.1]} mean(DTb[T(S)]**2)   # + Nelder-Mead polish
    # 2) piecewise refinement: each slide piece floats independently
    for piece in components(Ms):
        T_piece = local_rigid_min(mean(DTb[T(piece_pts)]**2) + λ‖Δt‖²,
                                  init=T0, θ∈±30°)
    # optional: per-piece small affine / TPS with strong regularisation
    # 3) coverage: block pixels farther than τ from any warped slide pixel
    Ws  = rasterise(union_pieces_warped)
    U   = DT(~Ws) > τ  restricted to Mb           # τ ≈ 2–3 % of sqrt(area(Mb))
    # 4) hygiene: flag double coverage (two pieces on the same block area): it means a
    #    piece slid into the wrong place, so lower the confidence
    return scores(U), U
```

* **Why it works.** A directed chamfer cannot be improved by moving slide tissue over
  empty block area, because there is no reward for covering block tissue. Missing
  tissue therefore stays uncovered.
* **Failure modes.**
  - Fused block fragments give large components.
  - A slide piece can slide onto the *wrong* block fragment of similar shape and leave
    its true partner "missing". Mitigate by checking double coverage and by using
    Pipeline C to assign pieces before refining.
  - Partial-face sections (smaller but present) leave a rim of uncovered tissue.
    Mitigate by eroding by τ and using `largest_blob`, not `missing_area`.

### Pipeline B — partial OT / partial FGW mass accounting (principled score)

```python
def pipeline_B(Mb, Ms, s, use_gw=False):
    B = sample(Mb, h);     a = full(len(B), 1/len(B))           # block mass = 1
    S = sample(Ms, h*s)/s; b = full(len(S), 1/len(B))           # same areal density
    m = min(1, b.sum()) * (1 - eps)      # eps≈0.03–0.05 absorbs boundary/shrinkage noise
    if not use_gw:
        S_al = pipeline_A_pose(S)                               # global/piecewise rigid
        C = cdist(B, S_al)**2;  C /= C.max()
        T = ot.partial.partial_wasserstein(a, b, C, m=m, nb_dummies=50)
    else:   # no alignment: rotation/flip invariant
        C1 = fragment_blocked_geodesic(B, Mb)    # intra-fragment distances; cap across fragments
        C2 = fragment_blocked_geodesic(S, Ms)
        F  = shape_feature_cost(B, S)            # e.g. dist-to-boundary, local thickness (FGW)
        T = ot.gromov.partial_fused_gromov_wasserstein(F, C1, C2, a, b, m=m, alpha=.7,
                                                       nb_dummies=50, numItermax=2e5)
        # multi-start: several G0 (identity on fragment centroids under flip / no flip)
    rho = T.sum(1) * len(B)                      # 1 = fully matched, 0 = unmatched
    U   = smooth_over_radius(rho, 2h) < .5       # mean in a 2h ball
    # transport-cost sanity: per matched pair, a cost above the q99 of clean pairs
    # means a suspicious match
    return scores(U), U, T
```

* **Why it works.** The total mass is capped at the slide's share. A missing block
  fragment cannot be "covered" without taking mass away from a region where it is
  cheaper to keep it. The deficit $1-\rho_i$ is literally the unmatched area.
* **Knobs.**
  - `eps` (tolerance).
  - Point budget: about 300–600 per side. Use sliced or entropic variants beyond that.
  - `alpha` in FGW.
* **Risk.** The scale error enters through b's density. The ±8 % error in the toy was
  tolerable; a 20 % error would not be.

### Pipeline C — fragment-level assignment with an explicit "unmatched" option (explainable)

```python
def pipeline_C(Mb, Ms, s):
    Bk = components(Mb, min_area);  Sl = components(Ms, min_area)
    # candidate slide groups: singletons + unions of nearby pieces (torn fragments),
    # capped at ≤3 pieces per group to stay tractable
    G  = singletons(Sl) + adjacent_unions(Sl, max_size=3, max_gap=g)
    for k, g in product(Bk, G):
        # flip-aware partial shape distance; scale fixed by s (± tolerance)
        cost[k,g] = w1*|log(area_g*s^-2 / area_k) - log(shrink)|   # area agrees?
                  + w2*shape_context_or_IDSC(k, g, allow_flip=True)
                  + w3*hu_or_efd_distance(k, g)
    # integer program (small: K ≤ ~15): choose groups so that every slide piece is used
    # at most once; every block fragment is assigned to one group OR to "missing" at cost θ_k
    #   θ_k = θ0 * area_k^γ   (γ<0 makes missing small fragments still costly to ignore)
    assign = min_cost_flow_or_ILP(cost, theta)          # Hungarian if no unions
    # optional global consistency: score assignment layout with GW on fragment centroids
    missing_fragments = [k for k in Bk if assign[k] == 'missing']
    # also partial: matched-but-much-smaller fragments -> per-fragment Pipeline A/B
    return score = max_k [assigned_missing_k * area_k] or calibrated margin
                    (cost of best assignment that includes k) - θ_k,
           localisation = contours(missing_fragments)
```

* **Why it works.** "Missing" is a real decision variable with a price. The output is
  one interpretable fragment-level reason per case.
* **Failure modes.**
  - Fused block fragments (one block component vs several slide pieces) are handled by
    unions, but only on the slide side. For the block side, either over-segment
    (watershed on the DT) or fall back to Pipeline B inside the fragment.
  - Large resections are a single component, so C is a no-op there and A/B carry the
    decision.

**Proposed combination.**

* Use C to assign pieces to block fragments, and to flag whole-fragment omissions.
* Use A per assigned group to get a residual coverage map.
* Use B as the calibrated score.
* Combine them with a small logistic model trained on synthetic omissions plus the
  hard negatives from `PROBLEM.md`.

---

## 5. Toy experiment

**Setup.** Script: [`scripts/toy_partial_matching.py`](../../scripts/toy_partial_matching.py).
It was run with
`python scripts/toy_partial_matching.py --n 80 --out results/toy_partial_matching`
in a venv with `pot==0.9.7.post1` and `pycpd==2.0.0` on Python 3.14, CPU only.

* **Blocks.** 2–6 non-overlapping star-shaped, elongated fragments, with radii from
  18 to 75 px on a 512² canvas. About 300 grid points per block.
* **Slides.** The slide is built from the same fragments with these perturbations:
  - independent per-fragment float: ±20° rotation, σ = 15 px translation;
  - shrink in [0.92, 1.05];
  - elastic warp;
  - **tear** (p = 0.4): the largest fragment is split along a random line and the
    halves are separated by 10–20 px;
  - **partial face** (p = 0.3): one fragment is scaled to 80 %;
  - then a global rotation in U(0, 360°), a mirror flip with p = 0.5, and a scale in
    [0.8, 1.25]. The methods only know the scale to within ±8 %.
* **Pairs.** Each block yields a NEGATIVE (complete slide) and a POSITIVE (one random
  fragment removed) with *identical* deformations. For half of the positives the
  removed fragment is the *smallest* one. Missing fraction is the removed fragment's
  area as a share of block area.
* **Read-outs.** Every method produces a per-block-point unmatched indicator and is
  scored the same way. For CPD this is coverage after warping or its posterior; for OT
  it is the smoothed block marginal deficit.

**Results.** n = 80 cases, i.e. 80 negatives and 80 positives. The score is
`cluster`, the largest connected unmatched area as a fraction of block area.
Missing-fraction bins hold 9 (<5 %), 37 (5–15 %) and 34 (>15 %) positives, so the
<5 % column is very noisy. Full table: `results/toy_partial_matching/summary.csv`.
Per case: `per_case.csv`.

| Method | AUROC (all) | Sens@95%Spec (all) | AUROC excl. partial-face | Sens@95%Spec excl. pf | AUROC miss 5–15 % | reported / true missing area (median, positives) | Localisation IoU |
|---|---|---|---|---|---|---|---|
| rigid_cov (global rigid+flip, directed chamfer) | 0.80 | 0.30 | 0.82 | 0.32 | 0.76 | — | 0.76 |
| **piece_cov** (+ per-piece rigid refinement) | **0.93** | **0.51** | **0.96** | 0.71 | **0.91** | **1.00** | **0.90** |
| piece_pw (piece pose + partial Wasserstein) | 0.88 | 0.44 | 0.94 | **0.75** | 0.86 | 0.99 | 0.78 |
| partial_w (global rigid pose + partial W) | 0.85 | 0.35 | 0.91 | 0.63 | 0.82 | — | 0.73 |
| **partial_gw (no alignment at all)** | 0.84 | 0.34 | 0.89 | 0.64 | 0.82 | 0.99 | 0.52 |
| srgw (semi-relaxed GW) | 0.64 | 0.11 | 0.63 | 0.13 | 0.56 | — | 0.34 |
| cpd_w0 (plain non-rigid CPD, coverage read-out) | 0.89 | 0.35 | 0.89 | 0.34 | 0.88 | **0.22** | **0.14** |
| cpd_w0.5 (coverage read-out) | 0.91 | 0.48 | 0.91 | 0.50 | 0.91 | 0.30 | 0.30 |
| cpd_w0.5_post (CPD outlier posterior 1−Pt1) | 0.67 | 0.35 | 0.67 | 0.36 | 0.70 | — | 0.26 |
| cpd_rev (block as centroids, low P1 mass) | 0.63 | 0.31 | 0.66 | 0.38 | 0.61 | — | 0.21 |

Mean `cluster` score on negatives by type (piece_cov / cpd_w0):

| Negative type | piece_cov | cpd_w0 |
|---|---|---|
| clean | 0.013 | 0.008 |
| tear | 0.028 | 0.014 |
| partial face | 0.049 (q90 0.19) | 0.008 |

![CPD vs coverage](../../results/toy_partial_matching/example_cpd_vs_coverage.png)

*Figure (`scripts/toy_partial_matching_figure.py 3`). A 3.9 % fragment is removed.
Plain CPD stretches the top-right fragment downwards to cover the gap and flags 2.3 %
of the block, scattered over the wrong places. Piecewise-rigid coverage flags exactly
the removed fragment.*

**Interpretation.**

1. **CPD's AUROC is misleading.** It looks decent (0.89) because the forced stretching
   leaves *some* residual somewhere. But it reports only about a fifth of the missing
   area, and in the wrong place (IoU 0.14). That gives no usable localisation and
   poor sensitivity at high specificity. This is the "absorbed" failure, quantified.
   - Raising the outlier weight helps only marginally (w = 0.5 gives IoU 0.30).
   - CPD's native outlier posterior is the worst read-out.
2. **Asymmetry plus rigid-first is the decisive ingredient**, not OT as such.
   - piece_cov and piece_pw use the same pose and perform alike.
   - Global-only rigid (rigid_cov) suffers on clean pairs because biopsy fragments
     float independently. Per-piece refinement removes most of that noise.
3. **Partial GW is the alignment-free fallback.** With no pose search at all it matches
   rigid-aligned partial W. Its weaker localisation (0.52) comes from floating
   fragments breaking the inter-fragment isometry. The fragment-blocked distance
   matrices proposed in section 4 should help, but this is untested.
4. **Partial face is the key hard negative.** Partial-face cases are where CPD's
   absorption *looks* like an advantage: low scores on negatives. Any honest coverage
   method flags a fragment that is 80 % present as 20 % uncovered. Separating "missing
   fragment" from "smaller fragment" needs fragment-level reasoning (Pipeline C:
   *is there any slide counterpart for this fragment?*) or blob-shape features (a
   compact blob vs a rim).

**Caveats.**

* Masks are perfect.
* Block fragments never fuse.
* Scale is known to ±8 %.
* About 300 points per set.
* Single seed.
* Runtime is about 3–15 s per pair on CPU, dominated by pycpd and Nelder–Mead. Partial
  GW takes about 1 s at N ≈ 300.

Treat these numbers as a *ranking* only.


---

## 6. Risks and open questions

1. **Segmentation dominates.** All of these methods work on masks. Low-contrast,
   glare-ridden block photos and fused fragments are the real bottleneck.
   - A false-positive block region (glare, wax debris) looks exactly like missing
     tissue.
   - Mitigation: a per-pixel "block-tissue confidence", fed in as OT mass weights so
     low-confidence block pixels count for little.
2. **Scale.** Partial mass accounting needs the block↔slide scale. Since the crops
   were resized differently, estimate scale per cohort plus a bounded per-pair search.
   Do not use the total-area ratio, which is confounded by the very thing we want to
   detect.
3. **Partial face / trimming.** The first sections from a block often show less tissue
   than the block surface. This is "present but smaller" and the methods will report
   it as unmatched rims.
   - Use blob-shape scores (a compact blob vs a thin rim).
   - Erode by τ.
   - Treat it as a separate class, as `PROBLEM.md` requires.
4. **Wrong-partner assignment** (symmetric or similar fragments, floating pieces). A
   slide piece can be put on the wrong block fragment, so the wrong fragment gets
   flagged. Detection is still correct but localisation is wrong.
   - Use double-coverage checks.
   - Use GW layout consistency as a tie-breaker.
5. **Folds and overlaps on the slide** reduce the slide's area, which inflates the
   deficit in Pipeline B. Use folded-tissue detection (darker H&E) to up-weight
   folded slide pixels (mass × 2).
6. **GW local minima** and O(N³) cost. Use multi-start (with and without flip), cap N
   at about 500, and use the fragment-blocked distances.
7. **Synthetic-to-real gap.** The toy has clean masks and no fusion. Real numbers will
   be lower. The *ranking* of methods (asymmetric/partial ≫ CPD) is the robust
   conclusion, not the absolute AUROC.
8. **Tooling on Windows / Python 3.14.**
   - Available: POT has a cp314 wheel; pycpd is pure Python; OpenCV, SciPy and torch
     are already installed; kornia is pure Python.
   - Not available: `probreg` has no wheel; BCPD is a C binary (build it or use the
     released exe); VALIS needs libvips and a JVM.

---

## 7. References

* Myronenko & Song, *Point Set Registration: Coherent Point Drift*, TPAMI 2010. https://arxiv.org/abs/0905.2635
* Hirose, *A Bayesian Formulation of Coherent Point Drift*, TPAMI 2021. https://doi.org/10.1109/TPAMI.2020.2971687 — code: https://github.com/ohirose/bcpd
* Hirose, *Acceleration of Non-Rigid Point Set Registration With Downsampling and Gaussian Process Regression* (BCPD++), TPAMI 2021. https://doi.org/10.1109/TPAMI.2020.3043769
* Chui & Rangarajan, *A new point matching algorithm for non-rigid registration* (TPS-RPM), CVIU 2003. https://doi.org/10.1016/S1077-3142(03)00009-2
* Chapel, Alaya, Gasso, *Partial Optimal Transport with Applications on Positive-Unlabeled Learning*, NeurIPS 2020. https://arxiv.org/abs/2002.08276
* Chizat, Peyré, Schmitzer, Vialard, *Scaling algorithms for unbalanced optimal transport problems*, Math. Comp. 2018. https://arxiv.org/abs/1607.05816
* Vayer et al., *Optimal Transport for structured data with application on graphs* (Fused GW), ICML 2019. https://arxiv.org/abs/1805.09114
* Séjourné, Vialard, Peyré, *The Unbalanced Gromov Wasserstein Distance*, NeurIPS 2021. https://arxiv.org/abs/2009.04266
* Vincent-Cuaz et al., *Semi-relaxed Gromov-Wasserstein divergence with applications on graphs*, ICLR 2022. https://arxiv.org/abs/2110.02753
* Flamary et al., *POT: Python Optimal Transport*, JMLR 2021. https://pythonot.github.io/ — https://github.com/PythonOT/POT
* Shen et al., *Accurate Point Cloud Registration with Robust Optimal Transport*, NeurIPS 2021. https://arxiv.org/abs/2111.00648
* Yang, Shi, Carlone, *TEASER: Fast and Certifiable Point Cloud Registration*, T-RO 2020. https://arxiv.org/abs/2001.07715
* Chetverikov et al., *The Trimmed Iterative Closest Point algorithm*, ICPR 2002. https://doi.org/10.1109/ICPR.2002.1047997
* Evangelidis & Psarakis, *Parametric Image Alignment Using Enhanced Correlation Coefficient Maximization* (ECC), TPAMI 2008. https://doi.org/10.1109/TPAMI.2008.113
* Belongie, Malik, Puzicha, *Shape Matching and Object Recognition Using Shape Contexts*, TPAMI 2002. https://doi.org/10.1109/34.993558
* Ling & Jacobs, *Shape Classification Using the Inner-Distance*, TPAMI 2007. https://doi.org/10.1109/TPAMI.2007.41
* Kuhl & Giardina, *Elliptic Fourier features of a closed contour*, CGIP 1982. https://doi.org/10.1016/0146-664X(82)90034-X — `pyefd`: https://github.com/hbldh/pyefd
* Arkin et al., *An efficiently computable metric for comparing polygonal shapes* (turning function), TPAMI 1991. https://doi.org/10.1109/34.75509
* Hu, *Visual pattern recognition by moment invariants*, IRE Trans. IT 1962. https://doi.org/10.1109/TIT.1962.1057692
* Leordeanu & Hebert, *A Spectral Technique for Correspondence Problems Using Pairwise Constraints*, ICCV 2005. https://doi.org/10.1109/ICCV.2005.20
* Kuhn, *The Hungarian method for the assignment problem*, 1955; SciPy `linear_sum_assignment`. https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.linear_sum_assignment.html
* DeTone et al., *SuperPoint*, CVPRW 2018. https://arxiv.org/abs/1712.07629 ; Lindenberger et al., *LightGlue*, ICCV 2023. https://arxiv.org/abs/2306.13643
* Sun et al., *LoFTR*, CVPR 2021. https://arxiv.org/abs/2104.00680 ; Edstedt et al., *RoMa*, CVPR 2024. https://arxiv.org/abs/2305.15404 ; Potje et al., *XFeat*, CVPR 2024. https://arxiv.org/abs/2404.19174
* He et al., *MatchAnything: Universal Cross-Modality Image Matching with Large-Scale Pre-Training*, 2025. https://arxiv.org/abs/2501.07556 — https://zju3dv.github.io/MatchAnything/
* Kornia (LoFTR / LightGlue on CPU). https://kornia.readthedocs.io/
* Gatenbee et al., *Virtual alignment of pathology image series for multi-gigapixel whole slide images* (VALIS), Nat. Commun. 2023. https://doi.org/10.1038/s41467-023-40218-9 — https://github.com/MathOminics/valis
* Wodzinski et al., *DeeperHistReg: Robust Whole Slide Images Registration Framework*, 2024. https://arxiv.org/abs/2404.14434
* Borovec et al., *ANHIR: Automatic Non-rigid Histological Image Registration Challenge*, IEEE TMI 2020. https://doi.org/10.1109/TMI.2020.2986331
* Weitz et al., *The ACROBAT 2022 challenge: automatic registration of breast cancer tissue*, 2023. https://arxiv.org/abs/2305.18033
* Tang et al., Virchows Archiv 2026 (dataset and case-identity baseline). https://doi.org/10.1007/s00428-026-04535-w
