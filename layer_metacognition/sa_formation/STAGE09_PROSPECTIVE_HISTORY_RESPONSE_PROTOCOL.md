# Stage 09 v2 — Prospective History Response Pilot

## Status and claim boundary

This document freezes the design before any Stage-09 model outcome is written.
It replaces the unrun 72-case draft in
`prospective_history_reliance_panel.py`; that draft is retained only as design
history and must not be executed as the formal Stage-09 experiment.

The experiment manipulates the external conversation History.  It may support
a prompt-level causal statement of the form “this complete History bundle
changed this measured outcome.”  It cannot establish that either internal
readout is a mediator, cannot authorize activation steering, and cannot revise
the failed authorization gates in Bridge 01/03/07.

## Objects measured

- `B_D`: fixed-answer deletion sensitivity,
  `log P(A* | NoText) - log P(A* | NoImage)`.
- `B_M56`: the mean of two symmetric fresh-donor replacement sensitivities.
- `U`: the byte-frozen Stage-03 raw shared Actual-Reliance readout at its
  fold-specific post-answer layer.  It is predictive and noncausal.
- `U_L18`: the byte-frozen Bridge-08 fixed-L18 candidate, reported as a
  same-layer secondary readout.
- `A`: the byte-frozen Stage-10/Bridge-06 L18 PANL attribution readout.
- `V`: the restricted common-9 verbal Source-Attribution score.

`U`, `U_L18`, and `A` are readouts, not interventions.  No hidden state is
modified (`alpha=0` throughout).

## Candidate pool and cohort

The unit is a fresh exact context, not a fresh item.  The pool is rebuilt from
read-only artifacts as follows:

1. Start from completed conflict rows in Truth Audit 01 with nonempty and
   different `A_T` and `A_I`.
2. Restrict items to the 76 completed Method-v2 confirmatory items.
3. Exclude every exact case previously used as a Bridge-01/02 target or donor
   1–4.
4. Exclude by item every prior History target used in the seven frozen History
   artifacts listed in the cohort manifest.

This produces exactly 395 contexts on 67 items.  Any count drift is a hard
failure.

Select 40 item-unique contexts with seed 42:

- exactly eight per fold;
- exactly four easy-image-side contexts per fold;
- four hard contexts per fold;
- hard Text/Image quotas are 2/2 in folds 0, 1, 3, and 4, and 1/3 in fold 2,
  where only one eligible hard-text item exists;
- within each cell, sort by
  `SHA256("stage09-v2|42|" + case_id)` and take the first unused item;
- retain every unselected context in the same deterministic order as a reserve.

The historical joint-answer side is used only for this outcome-blind balance.
No new Stage-09 `B/U/A/V` value participates in selection.

## Phase 0: endpoint freeze

For each selected context, the no-History Full answer-only prefix defines:

`A* = unique restricted top-1 answer`.

The same forward saves answer logits, probabilities, margin, entropy, and the
hard answer side.  A second, causally prefix-identical teacher-forced forward
appends exactly the one-token `A*` and captures the post-answer hidden states
needed by `U` and `U_L18`.

Only these predeclared structural events can invoke the next reserve:

- tied restricted top-1;
- `A*` is neither `A_T` nor `A_I`;
- irrecoverable technical failure after an identical retry.

Reserve replacement may not use any History outcome or any `B/U/A/V` value.
Once 40 valid endpoints exist, target cases and `A*` are immutable.

The reserve cell is fold + difficulty for easy targets.  It is fold +
difficulty + historical answer side for hard targets, because only the hard
allocation has a frozen side quota.  Within that cell the runner takes the
first unattempted hash-ranked context whose item is absent from the other 39
active targets.  The failed target is removed before this uniqueness check.

## Donors

Each recipient receives one irrelevant-History donor and two symmetric
replacement donors (`d5`, `d6`).

- Donor items are outside the final 40 target items.
- Within a recipient, target, History donor, d5, and d6 are item-distinct.
- Donors match fold and difficulty; prior bin, condition, prompt length, answer
  pair, and historical side are deterministic ranking variables rather than
  eligibility filters.
- The same d5/d6 pair is reused in every History branch for that recipient.
- Cross-recipient reuse is allowed because strict global uniqueness is not
  feasible in all strata.  Reuse counts and donor clusters are frozen and
  reported.  Primary inference is conditional on these frozen donors and
  resamples recipient items within fold.  A leave-one-reused-donor-cluster-out
  diagnostic is descriptive; no unimplemented multiway bootstrap is claimed.

The irrelevant History branch uses the donor's own `A_T` and `A_I`.  It is a
coherent unrelated-History bundle.  Consequently, Relevant minus Irrelevant is
not described as a pure “relevance” effect: the historical item and answer
identity also differ.

In the initially selected 40 targets, only 6 have an irrelevant donor with the
same ordered `(A_T,A_I)` pair; 34 use the declared fallback.  A Phase-0
structural replacement may change these counts, so the final endpoint manifest
recomputes and freezes them before Phase 1.  All summaries report this match
tier.  Exact-pair and fallback strata are shown separately as descriptive
sensitivity analyses; the small exact-pair stratum is not used for a powered
confirmatory claim.

## Phase 1: no-History qualification

For all 40 fixed endpoints, measure the seven answer-only evidence conditions:

1. Full
2. NoText
3. NoImage
4. ReplaceText d5
5. ReplaceImage d5
6. ReplaceText d6
7. ReplaceImage d6

Then measure `U/U_L18` at answer-only post-answer and `A/V` with the frozen
common-9 joint protocol.  Qualification has two independently frozen tracks:

- the behavior/readout track authorizes History `B/U` measurement;
- the report-formation track authorizes History `A/V` measurement.

Both require exactly 40 unique cases and items and eight per fold.  Shared
structural checks are the Phase-0 causal-prefix equality, absence of a verbal-SA
request in the audited answer-only messages, and `alpha=0`.  Each track then
requires only its own 40/40 finite outcomes and exactly-once hook: the B/U
fields and answer hook for behavior/readout, or the A/V fields and joint hook
for report formation.  A technical failure in one track is saved independently
and cannot erase a successfully measured row in the other track.  The
behavior/readout track further requires:

- 40/40 technically complete, with no SA request in answer-only messages;
- `D` versus `M56` item-bootstrap Spearman CI lower bound is positive;
- `M5` versus `M6` CI lower bound is positive and ICC is at least .60;
- frozen primary `U` has R² above zero, positive Spearman CI lower bound, and
  positive foldwise Spearman in at least four folds;
- primary `U` has a positive paired squared-error improvement over the
  reconstructed frozen Stage-03 nuisance-only predictor;

The report-formation track separately requires frozen `A` versus `V` to have a
positive Spearman CI lower bound.  Therefore a failed `B/U` qualification does
not erase a valid prospective `A/V` History experiment, and a failed `A/V`
qualification does not authorize report claims from a valid `B/U` experiment.

Each failed track is skipped and receives its own gate-failure artifact.  If
both pass, the complete four-layer panel runs.  If neither passes, Phase 2 is
skipped.  This prevents an unreliable measurement from being carried into
History change scores while preserving the independent report-formation path.

## Phase 2: full History factorial

There are nine branches per context:

- `no_history`;
- Relevant target History: Text/Image × replay `A_T/A_I`;
- Irrelevant donor History: Text/Image × replay donor `A_T/A_I`.

Every branch keeps the final target question, text clue, image, fixed `A*`,
replacement donors, and measurement protocols identical.  Within each
Relevant/Irrelevant stratum, the 2×2 design separates:

- modality main effect: Image minus Text, averaged over replay side;
- replay main effect: AI minus AT, averaged over modality;
- modality × replay interaction, which captures evidence-answer congruence.

For each fully authorized branch, seven answer-prefix forwards measure `B`, one
teacher-forced answer-only forward measures `U/U_L18`, and one separate
common-9 joint forward measures `A/V`.  The 40 final retained objects therefore
contain exactly 2,520 successful behavior cells, 360 successful post-answer-
hidden cells, and 360 successful joint cells: `3,240` successful formal
forwards.  Phase 1 contains 360 successful forwards.  The corresponding
successful-cell budgets are 3,240 when both tracks pass, 2,920 when only `B/U`
passes, 680 when only `A/V` passes, and 360 when neither passes.

These counts exclude the non-formal GPU smoke and any failed Phase-0 endpoint,
identical retry, or interrupted/failed branch attempt.  The runner records
attempt and failure overhead separately; it must not claim that wall-clock
model invocations are always exactly 3,240.
Runs are resumable; a 60-minute invocation is a budget boundary, not a promise
that all 3,240 forwards finish in one process.

For primary `U`, answer identity and answer margin nuisance inputs are frozen
to the no-History Phase-0 `A*` and Full margin for every History branch.  This
keeps nuisance fixed while History changes the captured hidden state.  A second
off-policy sensitivity projection uses the branch's current Full margin and is
labelled explicitly; if the branch top-1 differs from `A*`, it must not be
described as an on-policy readout.

## Primary analyses

For `Y in {B_D, B_M56, U_prediction, A_prediction, V}` report:

1. Relevant and Irrelevant modality, replay, and interaction effects.
2. Relevant-minus-Irrelevant difference-in-differences, labelled as a bundle
   comparison rather than pure relevance.
3. Change-score reliability: `delta M5` versus `delta M6`, their ICC, and
   `delta D` versus `delta M56`.
4. Itemwise shift alignment among B/U/A/V.
5. Direct bootstrap comparisons such as
   `rho(delta B, delta U) - rho(delta B, delta A)`; a significant-versus-null
   pair alone is not evidence of specialization.
6. Full-answer `log P(A*)`, margin, entropy, and hard side as secondary total
   effects.  Hard side is explicitly `text`, `image`, or `other`, with separate
   finite Image and Other indicators.  No post-treatment endpoint matching or
   dropping of Other outcomes is allowed.
7. Relevant-minus-Irrelevant contrasts are labelled History-bundle contrasts.
   Report exact ordered-pair and fallback donor tiers separately as descriptive
   sensitivity analyses; do not infer a pure relevance effect from the six
   exact-pair cases.

All uncertainty uses 1,000 recipient-item bootstrap resamples within immutable
folds.  A qualification CI is usable only when at least 950 of 1,000 replicates
are finite.  Co-primary p-values, when reported, use Holm correction.  A null
result is not called equivalence unless a separately frozen equivalence bound
is met.
