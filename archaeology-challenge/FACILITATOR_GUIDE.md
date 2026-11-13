# Facilitator guide: From Tells to Cyprus

This guide supports the participant and instructor notebooks. Frame the task as helping a survey team identify locations that deserve further investigation. Image review includes interpreting possible traces of settlement, comparing them with existing records and deciding where better imagery or a field visit would be useful. The budget of 50 images is an explicit teaching assumption for a first review pass; it is not a real team's measured capacity.

Explain the data choice before starting the code. Enki supplies image-class labels for learning and evaluating a ranking. Cyprus supplies a settlement catalogue and satellite imagery for asking whether the model's assumptions carry over to a different setting. The bundled Cyprus records describe locations and occupation periods, not image visibility or verified absences. A project intended for Cyprus would need representative local imagery, suitable expert labels and an independent geographic evaluation. A high score is a reason to inspect a tile, not evidence that it contains an archaeological site.

## Learning goals

By the end, participants should be able to:

1. Explain what the source labels and the balanced teaching sample represent—and what they do not.
2. Interpret precision and recall at a fixed review budget, and compare image representations using validation data.
3. Explain why model selection and final evaluation use separate data, and why removing similar images does not establish geographic independence.
4. Calculate Cyprus catalogue agreement, distinguish it from detection accuracy, and propose evidence needed for a defensible transfer study.
5. Make a proportionate recommendation supported by the workshop results and their limitations.

The audience has basic Python and pandas; previous scikit-learn experience helps
but is not required. The core path has two short coding exercises (two Enki and three Cyprus expressions),
short parameter-change experiments, image discussion and one five-sentence report.
The optional block between sections 5 and 6 offers three alternatives. Choose one,
or use that time for support; completing every lab is not the four-hour task.

## Before the session

- Run the participant notebook from top to bottom in a fresh environment. Confirm the Jupyter kernel, dependencies, internet access, and that figures and the final files appear. The default cached-feature path needs a CPU, not a GPU.
- Budget roughly 73 MB of network transfer for the first Enki sample download and about 400 MB of free space. Download time and service availability vary. For weak connectivity, distribute a previously prepared `enki_workshop_sample.zip`; place it in `archaeology_workshop/` before the Enki data cell. It is validated before reuse. The ZIP may be supplied separately from the repository. Package installation still needs network access unless dependencies are already present.
- Review the instructor notebook's outputs and source notices. Do not share it with participants until your planned debrief; it includes worked interpretation and held-out results. Treat the public-source holdout as educational: source labels are recoverable and it is not a secure competition.
- Choose how learners will answer prompts: individually in notes, in pairs, or in a brief group discussion. If tools for displaying notebook widgets are unavailable, text progress displays are acceptable. Set expectations that time estimates vary across groups.

## Completion-time estimate

This is a planning estimate, not a timed learner study. Count the participant copy,
excluding optional labs, credits and the folded setup source. It has about **4,376
words of core explanation**. At an assumed 150 technical words per minute, reading
takes about **29 minutes**. The active-work allowance below covers running and
understanding code, completing tasks and discussion; it excludes that reading allowance.

| Core block | Prose words | Reading minutes | Active-work minutes |
|---|---:|---:|---:|
| Orientation and setup | 453 | 3.0 | 18 |
| 1. Images and split | 451 | 3.0 | 16 |
| 2. Shortlist metrics | 329 | 2.2 | 14 |
| 3. Model comparison | 1481 | 9.9 | 40 |
| 4. Mistake inspection | 292 | 1.9 | 17 |
| 5. Cyprus comparison and calculations | 1048 | 7.0 | 17 |
| 6. Recommend, save and check | 322 | 2.1 | 35 |
| Total | 4376 | 29.2 | 157 |

Calculation: **29.2 + 157 + 10-minute break + 15 minutes for support
≈ 211 minutes**, approximately **3½ hours** for the core path.
One optional lab adds 25–35 minutes, giving roughly **4 hours**.
All three labs add 75–100 minutes and belong in a longer session or follow-up.

Assumptions: a working Python environment, the supplied cached features, basic Python
and pandas, and a normal first image download. The setup allowance covers routine
installation and loading; 15 minutes is contingency, not a guarantee against network
problems. A cold setup or learners new to notebooks can take longer. Reading aloud
and extended discussion can also change the balance. Use the checkpoints below to
adjust the session; the arithmetic does not establish a precise human completion time.

## Four-hour suggested sequence

The slots below round the estimate into a teaching schedule. Support is absorbed
within the slots and the optional/catch-up block, rather than added again.

| Time | Activity | Facilitator action |
|---|---|---|
| 00:00–00:20 | Orientation and setup | Explain the 50-image scenario and start loading. |
| 00:20–00:40 | 1. Inspect images and the split | Discuss one labelled pair; keep the three data roles clear. |
| 00:40–01:00 | 2. Shortlist metrics | Walk through two entries, then predict and check four. |
| 01:00–01:50 | 3. Compare models | Fit, score, complete two expressions and interpret the selection. PCA comes later. |
| 01:50–02:00 | Break | Check who needs setup or coding support. |
| 02:00–02:20 | 4. Inspect mistakes | Keep one observation and try ten extra review slots. |
| 02:20–02:40 | 5. Measure Cyprus catalogue agreement | Inspect two pairs, complete three calculations and interpret their limits. |
| 02:40–03:10 | Optional lab or catch-up | Choose A, B or C; skip extensions if the core work needs support. |
| 03:10–03:40 | 6. Recommendation and export | Reuse the earlier notes, exchange drafts and save the evidence. |
| 03:40–04:00 | Final reveal and debrief | Assess the saved choice once; discuss which limitations remain. |

**Pacing decisions:** by 01:50 aim to have chosen a model; by 02:40 aim to have the
Cyprus calculations and notes. If behind, skip the optional lab and the five-minute all-positives
challenge in section 4. Preserve both calculation exercises, the distinction
between Cyprus records and labels, and time to write and share the recommendation.
If ahead, choose one lab; do not repeat every earlier discussion or start new downloads.

## Prompts and feedback

Give people time to make a prediction or write a one-sentence reason before showing or discussing the explanation. Ask a follow-up such as “Which evidence supports that claim?”; allow pairs to compare answers. These checks are for learning, not high-stakes assessment.

### 1. The sample and the label

**Ask:** “What do `1` and `0` mean here? What does a 50/50 sample tell us about the real landscape?”

**Look for:** `1` and `0` are labels supplied by the Enki archive for tell and negative images. The workshop deliberately selected 300 of each before deduplication; this balance is a sampling choice, not an estimate of operational prevalence or a guarantee of archaeological truth. The source categories describe archive context; they are not verified independent regions.

**If someone says “half of all tiles contain tells”:** distinguish the selected teaching sample from the population of future survey imagery. Ask what sampling design and independent labels would be needed to estimate prevalence.

For the image discussion, use the 12 training images displayed in the notebook. Ask each
pair to choose one image with each label, record both tile IDs, and describe a visible
difference before suggesting an explanation. Participants are not being asked to authenticate
archaeological sites from these thumbnails. A possible mound shape is a candidate cue related
to the task; a systematic difference in sharpness could let a model predict labels from image
quality instead. The size audit now establishes a source acquisition confound: all
300 retained tells are 350 × 350, whereas 249 negatives are 64 × 64 and 44 are 350 × 350.
Both models now receive a common 64 × 64 RGB input; DINO enlarges that input to 224 × 224.
This reduces the difference in available pixel detail, without matching ground sampling
distance or removing all acquisition cues. Whether a classifier uses residual sharpness
differences remains a hypothesis.
Colour and land cover are not automatically shortcuts. Accept uncertainty and ask what further
image comparisons or archaeological evidence would distinguish the explanations. This small
preview cannot establish a pattern across the dataset or show which features a model uses.

### 2. What does a shortlist measure?

**Set up the task:** this section is a counting exercise on the invented list
`[1, 1, 0, 1, 0]`. No images are loaded or judged. The list does not refer to the
training-image grid. Explain that taking the first `k` entries represents selecting
which images would be opened first in a real review; the calculations use labels
already supplied with the dataset.

**Walk through:** selecting the first two entries gives `[1, 1]`. There are two entries
and two `1`s in that selection, while the complete list has three `1`s. Precision is
`2/2 = 100%`; recall is `2/3 ≈ 67%`.

**Ask:** change `k` to four, giving `[1, 1, 0, 1]`. Count its `1`s, divide by four for
precision, and divide by three for recall. Which extra entries changed the fractions?

**Look for:** precision is `3/4 = 75%` and recall is `3/3 = 100%`. The extra entries are
`0` and `1`: the selection now includes every positive, but also includes a negative.
The example is invented; it is not an Enki result. Section 3 applies the calculation
to actual model rankings of the same validation images, with `k = 50`. More positives
in that shortlist means higher recall at the fixed budget. AP evaluates the whole
ranking and breaks a tie; it is not part of this five-label calculation.

**If someone uses the same denominator for both:** point to the two lists. Precision
uses the number of entries in the selection; recall uses the number of `1`s in the
complete list. A real reviewer would inspect images for possible archaeological
features, but learners do not need to do that to calculate these metrics.

### 3. What does the comparison establish?

**Frame the comparison:** return to the real Enki images from section 1. We are asking
which representation helps a logistic regression classifier put more source positives
into a 50-image shortlist. Both classifiers use the same training images, labels,
validation images and settings. Their input features differ, and each learns its own
weights. Do not describe this as training DINOv2 to detect archaeology.

**Walk through the five steps in the notebook:**

1. **Features:** one image becomes one row of numbers. Colour/texture supplies 68 fixed
   measurements; DINOv2 supplies 384 cached values from a pretrained image encoder.
   “Frozen” means the encoder's parameters stay unchanged. The cache avoids running
   the neural network during the workshop. In the summary table, 200 training images
   and 68 features means 200 rows and 68 columns, not 68 training images.
2. **Fit:** a pipeline first scales the features using training means and standard
   deviations, then learns logistic regression weights from the training labels.
   Ask learners to point to the two arguments of `.fit()`. Validation and test data
   are not used to fit either step. `C=1` fixes the penalty on large weights; this is
   a comparison at shared settings, not a search for the best possible configuration.
3. **Score:** `predict_proba` receives validation features only. Its second column is
   the score for label `1`. The preview attaches existing labels after prediction and
   shows the four highest-scoring images for each model, colour/texture then DINOv2,
   with a matching table and row of thumbnails for each.
   Image captions give rank, score and tile ID; read left to right in descending score order.
   This preview does not select a winner.
   Both full rankings enter the next calculation. No new image labelling is required.
4. **Compare:** use the table and printed fractions to connect `hits`, `reviewed`,
   precision and recall to actual counts. Higher recall at 50 wins. Equal recall means
   equal hits on the same pool; AP then compares ordering across the full ranking.
   An exact tie in both metrics is resolved by name for reproducibility, not evidence.
5. **Interpret:** compare recall with the random reference and budget ceiling. With the
   supplied split, validation has 119 images and 60 positives. A random 50-image selection
   includes about 25.2 positives on average: expected recall is 50/119, about 42.0%.
   At most 50 of 60 positives fit, so the recall ceiling is 83.3%. Read model results
   from the current run rather than assuming a particular winner.

PCA is no longer part of the fitting sequence. Move directly from the pipeline
explanation to fitting, scoring and comparing; lab A offers the separate visualisation.

**Section 3.4 short exercise (2–3 minutes):** pause before displaying the comparison
table. The supplied code takes the first 50 rows of the full colour/texture ranking
from section 3.3 and counts their positives. Learners fill in
`precision = hits / reviewed` and `recall = hits / total_positives`, then run the
comparison cell to check their answers. Explain that each earlier four-image display
was only a preview of its model's full ranking. The instructor copy includes both expressions.

**Ask:** “What was learned in this session, and what was already fixed? Which data were
used to learn, which to choose, and which are reserved for the final assessment?”

**Look for:** both scalers and logistic regressions were fitted on training data.
The colour/texture calculation and DINOv2 encoder were fixed. Validation selects the
pipeline reused for mistakes and Cyprus; test assesses the frozen choice later.
Computing fixed features for held-out images does not fit a classifier on their labels.
Using test performance to change the choice would turn test into selection data.

**For the output discussion:** ask learners to state the extra hit count, explain each
fraction's denominator, and compare recall with both references. If they report “83%
means it made 17% mistakes”, distinguish recall from overall accuracy and show why
some positives cannot fit into the budget. If both models have equal hits, neither
finds more positives in 50 slots; a higher AP describes a better full ranking.

**If AP remains unclear:** use the supplied `[1, 1, 0, 1, 0]` label order, assuming no
tied scores. Positives occur at ranks 1, 2 and 4; average the precisions at those ranks:
`(1/1 + 2/2 + 3/4) / 3 ≈ 0.917`. AP is not the average of the two models' precisions.
The helper handles tied scores using scikit-learn's definition.

**Conclude within the evidence:** the comparison establishes performance on this
selected sample and budget. It does not show which visual cues caused a score or
establish transfer to a new region. If someone says “the pretrained model must win”,
return to the results: feature count or prior training alone does not decide the winner.
A simple baseline can capture useful cues or source-specific rendering differences.
Proceed to section 4 to inspect actual mistakes; keep test results for the final reveal.

**Resolution audit:** a metadata-only reference learns the positive fraction for each
original width/height on training rows, then predicts validation labels. It achieves
92.4% accuracy on the default split without seeing pixels. It does not compete for
model selection. The following table measures each image model within each native
size after common preprocessing. Native 350 × 350 validation images include 60 tells
and nine negatives: compare AP with the positive fraction (about 0.870), not with zero.
The 50 native 64 × 64 validation images are all negatives, so their AP is undefined.
The small same-size subset tests one confound; it does not establish geographic
generalisation or prove that the common resizing removed every source cue.

### 4. Inspect mistakes and the review budget

**Clarify the decision:** source negatives inside the shortlist are false alarms;
source positives outside are misses. These categories depend on the review budget,
not a score threshold of 0.5. A high-scoring positive at rank 51 is outside a 50-image
shortlist. Neither category independently verifies archaeological presence or absence.

The cell prints complete group counts and shows at most four examples per group,
highest score first. Use the counts from the corrected run. If the false-alarm
gallery is empty, that is a reported outcome, not a plotting failure; direct learners
to the missed positives in that case.

**Image reading:** ask for a tile ID, rank and visible observation, then distinguish
that observation from a hypothesis about the model. Accept uncertainty. A rounded
boundary or vegetation is observable; saying the classifier used it requires further
evidence. Ask which comparison with other images would test the proposed explanation.
The default recommendation uses the displayed `CY5_site` image from section 5,
so it does not depend on a particular model placing an Enki tile in its mistake gallery.

**Exercise (about five minutes):** before running the budget cell, ask whether ten
extra reviews necessarily recover ten positives. Participants change just
`inspection_budget = 50` to `60`, then inspect the added-image table and the newly
included source-negative images. Which original missed example was recovered?

Read the new positives, negatives and remaining misses from the current output.
The model's order is unchanged; the budget decides how far down it we inspect.
For the harder extension, locate the last positive's rank in the current ranking;
that is the minimum budget that recovers all positives on this validation sample.
The instructor copy runs the 60-review example. The original comparison and export
retain `REVIEW_BUDGET = 50`.

**Geographic independence:** connect this error review to the limits of the split.

**Ask:** “Does removing perceptually similar images guarantee the model will work in a new region?”

**Look for:** no. Filtering reduces some repeated or near-repeated images, but can miss transformed views and does not identify every site or geographic area. Neighbouring landscapes, capture conditions and repeated sites can make random partitions easier than a new-region deployment. This sample lacks verified site groups or coordinates needed to establish geographic independence.

**If someone says “any spatial split is always unbiased”:** explain that evaluation should match the deployment question. A geographically separated, site-independent holdout is a useful design for a new-region claim, but the size and placement of blocks and how sites are sampled still matter. The present data do not support measuring that claim.

### 5. What does Cyprus tell us?

Begin with the changed data. The fitted Enki pipeline and training scaler are reused;
there is no Cyprus classifier fit. The twelve chips are 1 km squares at 10 m per pixel
from 5 August 2024, paired into six catalogued and six nearby unlabelled locations.
Their common 64 × 64 model inputs span about 15.6 m per input pixel. Matching array
size is not matching Enki's physical scale. The galleries display original chips.
`paired_site` joins the two images of a pair; `tile_id` identifies one image.
The table ranks all twelve images; the first gallery previews four.

**Clarify the records:** catalogue membership says where a settlement is recorded,
not whether remains are visible in this image. `unlabelled_background` is a stored
category name, not a confirmed negative class. Several occupation rows can describe
one settlement. Do not calculate detection precision or recall using these categories
as image labels. Differences in resolution, land cover, rendering, target form and
location uncertainty can matter; a score alone cannot identify which caused its value.

**Pair exercise (about five minutes):** run `pair_id = "CY5"`, then change the ID to
`"CY479"`. The catalogued image is on the left, its nearby unlabelled image on the
right, with ranks and scores for reference. Ask learners for a two- or three-sentence
comparison that names the images, records an observation and explains the missing
evidence needed to judge a detection error. They can edit the supplied Markdown notes cell.

**Section 5.3 calculation (six to eight minutes):** the supplied table shows all six
pairs and the four-image Cyprus shortlist. Participants complete three expressions:

```python
pair_agreement = (pair_wins + 0.5 * pair_ties) / pair_count
catalogue_fraction = cyprus_hits / cyprus_reviewed
catalogue_recovery = cyprus_hits / catalogued_total
```

For the current DINOv2 run, these are 4/6 = 66.7%, 3/4 = 75%, and 3/6 = 50%.
Random ordering averages 50% pair agreement, 50% catalogue fraction and 4/12 = 33.3%
catalogue recovery. Have learners explain why the last two calculations have different
denominators, connecting them to Enki precision and recall. The next cell checks their
answers and computes both models' references; unfilled blanks produce a completion
prompt so that a fresh notebook still runs. The instructor supplies the expressions.

Ties use unrounded scores. Pair ties receive half credit. The colour/texture model
has two wins and four ties, also giving 66.7% agreement; that is different evidence
from DINOv2's four wins and two losses. Its tied shortlist yields two catalogue
locations under image-ID ordering, but zero to four could be selected with another
tie ordering. The output flags this ambiguity. Do not use this tiny sample to choose
a new model or threshold. Ask for one Cyprus calculation in the final recommendation.

These are measurements of catalogue recovery and paired ranking, not detection
precision, recall or accuracy. The six pairs are selected, small in number, and
some nearby image footprints overlap. Independent visibility labels and geographic
sampling remain necessary for a performance claim about archaeological detection.

Read both models' current score distributions against Enki validation before the pair
exercise. An overall shift towards zero or one, or disagreement between the models,
exposes a transfer concern; it does not measure Cyprus accuracy or establish causality.
Read the current pair table to compare `CY5` and `CY479`. The `CY5` images show larger bright angular patches in the catalogued
tile and winding pale lines through darker terrain in the other. These observations
do not show which cues the classifier used. Neither ordering is evidence of a false
positive or false negative without suitable independent image labels.

Use current outputs if settings change. Either model can give extreme scores;
displayed scores can round to the same value. High scores do not
establish presence, and close scores should not invite a strong interpretation of rank.
Accept uncertainty about what is visible in these small images.

Do not tune preprocessing, thresholds or model selection to make these twelve
unlabelled images look convincing. The preprocessing correction is fixed by the
source-data audit. Local detection work needs a defined visible target, suitable
imagery, expert annotations and a geographic holdout; catalogue membership cannot
supply those labels automatically.

The main section measures catalogue agreement and identifies missing evidence.
Lab C makes the next step concrete with a five-answer plan and an invented test result.

### 6. Recommend, save and check

**Write (about ten minutes):** run the evidence cell, then ask for one paragraph of
five sentences: a decision, Enki counts and precision/recall, an identified image
observation, Cyprus catalogue recovery with its limitation, and the next evidence
needed. The summary supplies the selected model's actual results at the original
budgets. The instructor example follows the same five prompts. A recommendation
against deployment is valid when supported by the evidence.

**Peer feedback:** does the evidence support the proposed use? Revise any claim that
counts newly discovered sites, equates catalogue recovery with detection accuracy,
or presents a visual observation as an explanation of the model. Reuse lab C's plan
if completed, but keep its invented percentages out of the report.

**Save and share:** export the completed paragraph and evidence as `team_submission.zip`,
then save the notebook to retain notes and figures. Share both before revealing test
labels. A blank recommendation creates a draft; complete it and rerun export before
sharing. Later edits also require another export. The [README](README.md#submission-files)
lists the ZIP contents; the export records the run without serializing the fitted estimator.

**Check and reflect:** participants enable `REVEAL_TEST` after sharing; the instructor
copy already displays the result. The table compares validation with the saved test
ranking at the exported budget, using counts and percentages, followed by a random
reference. Keep the original recommendation and add one debrief sentence: does the
test support or weaken it, why, and what still needs testing? Save the notebook again.
Stop model selection here. This Enki check establishes neither geographic independence
nor Cyprus detection accuracy.

For an optional later follow-up, revisit the shortlist and transfer prompts with a
different example and ask learners to reconstruct their reasoning before comparing
with feedback. A single workshop does not itself establish long-term retention.

## Optional lab facilitation

These activities appear **between sections 5 and 6**, before export and the final
reveal. Choose one for the 30-minute slot. The instructor notebook shows both plots
and the Lab C practice for reference; participants still choose just one lab.

**Lab A: PCA (25–35 minutes, including interpretation):** ask learners to project the same
200 training images for both feature representations. The participant function has
three blanks; the separate plotting cell is supplied and starts with
`RUN_PROJECTION = False`. After completing the function, run it, enable plotting and
run the plotting cell. This guard allows a fresh notebook to run before the exercise
is completed. The instructor copy supplies the solution and enables the plot.

The three replacements are:

```python
X = feature_sets[name]["train"]
X_scaled = scaler.fit_transform(X)
coordinates = pca.fit_transform(X_scaled)
```

PCA uses feature variation to choose its axes; it never receives the class labels.
Labels are added as colours and marker shapes afterwards. Each dot is one training
image; each axis combines many original features. Standardisation matches the scaling
used by our classifiers, and a separate scaler and PCA are fitted for each representation.
Keep validation and test out of this exploration. The code leaves the original matrices
intact, so the fitted classifiers still use all 68 or 384 features.

**Debrief the plots:** ask learners to describe an area of overlap or separation and
report the retained variance for each representation. That percentage measures variation
preserved in the standardised features, not classification accuracy or archaeological
relevance. PCA's leading directions need not carry the differences between labels.
Overlap can hide separation in omitted dimensions; visible training separation does
not establish generalisation. Each panel has its own axes, so coordinates cannot be
matched across representations. The model comparison already used validation recall; this plot does not replace it.

For the extension, set `N_COMPONENTS = 3`. Inline plots may be static; change `AZIMUTH`
and rerun to inspect another viewing angle. A third component can expose hidden structure,
but neither a favourable angle nor a higher retained-variance percentage establishes
a better classifier. If someone used `.fit(...)` in a blank, explain that it returns
the fitted estimator; `.fit_transform(...)` returns the transformed numeric rows needed
by the next operation. Rerun the function-definition cell after editing it.

**Lab B: is it worth labelling more training images? (25–35 minutes)**

Start by pointing to the two quantities: training uses 20, 50, 100 or 200 images;
review always opens the first 50 images in the same validation ranking. Learners
already have the labels. They are simulating a smaller labelled training collection,
not being asked to label new imagery or change the review budget.

Allow roughly three minutes to predict, five to run/read the output, ten to complete
the answer template, and seven to discuss the choice. Participants change only
`RUN_LEARNING_CURVES` to `True`. The supplied code fits each model three times per
training size. The table and plot show the mean, fewest and most tells found in
50 reviews. Explain a fractional mean with the supplied 40/42/41 example.

The deliverable is the two-row table plus three short sentences in the provided
Markdown cell. Learners copy means at 20 and 200 training images and subtract them;
then compare DINOv2 at 50 and 200, including its 50-image range. They choose 50 or
200 for the next small experiment and state what the extra 150 labels buy. Accept
either choice if its reason uses the observed counts.

The instructor notebook computes its worked answer from the current run. With the
default data, colour/texture goes from 30.3 to 40.0 mean hits (gain 9.7), and DINOv2
from 41.3 to 47.0 (gain 5.7). DINOv2 at 50 training images finds 47.0 on average,
ranging from 46 to 48; at 200 it also finds 47.0. A 50-image starting experiment is
therefore a defensible choice for these data, not a universal requirement. At 200,
all repeats use the same full training set. Their zero range is not proof of
certainty on new regions. The helper clones the models and does not use test labels.

**Lab C: what would a fair Cyprus test look like? (25–30 minutes)**

Use the supplied imagined situation: detailed imagery in three areas, A for training,
B for choosing settings and C for the final check. Two experts label all 100 sampled
C images before seeing scores. The output is five one-sentence answers and two
calculated decisions. Learners do not need to invent a research proposal or collect data.

Allow about five minutes to read the situation, eight to complete the five-row plan,
seven for the arithmetic and changed-count experiment, and five for discussion.
The plan asks for a visible target, who labels it, why all 100 images need labels,
which area is reserved, and how uncertain cases are recorded. The instructor copy
provides specific example answers. Check that learners explain how positives outside
the top 20 would be counted and do not turn uncertainty into a negative label.

The calculation uses explicitly invented counts: 25 visible positives among 100 images,
20 reviewed, and 14 hits. Participants complete `practice_hits / practice_reviewed`
and `practice_hits / practice_positives`, then enable `RUN_CYPRUS_PRACTICE`. The
supplied rule requires precision at least 70% and recall at least 50%. The first
result, 70%/56%, passes. Ask learners to predict the result for 10 hits before
changing that one value: 50%/40% fails. These thresholds are teaching assumptions,
not an operational recommendation, and the counts assume all labels were resolved.

The invented exercise must not appear as measured Cyprus performance in the final
recommendation. Use a next step from the plan there; section 5.3 supplies the real
catalogue-agreement measurement. A successful hypothetical test would support a
larger study, while deployment would need evidence suited to the actual use.

**Extra reference for catalogue questions:** this is not another required exercise.
`workshop.summarise_sites(pd.read_csv(assets / "Cyprus_Settlements.csv"))` audits the
catalogue. Expect 8,766 occupation rows and 1,559 settlement IDs. `CY453` has conflicting
coordinates and no summary point. Blank/F quality codes are undocumented;
`Long_Cord`/`Lat_Cord` are degrees, while `Longitude`/`Latitude` are UTM zone 36N metres.
These codes describe provenance, not numerical error radii.

## If something goes wrong

- **Cell reports an undefined name:** restart the kernel if needed, then rerun setup and every prerequisite cell in order. Restarting clears variables; save any written reflections first.
- **Download fails or the server ignores range requests:** retry later or use the facilitator-supplied sample ZIP at the expected path. Do not rename the full archive to the sample ZIP name.
- **Model weights or GPU unavailable:** use the supplied feature cache. Both notebooks use no GPU or model download.
- **No image or progress widget is visible:** confirm the active kernel matches the environment with notebook dependencies and restart it after installation. Some notebook frontends cannot render widgets; a plain-text progress indicator is sufficient. Check that the frontend can display ordinary Matplotlib figures.
- **Session reset or interrupted execution:** rerun in order, reuse the validated sample when available, and save/export the submission and notebook before the session ends. Colab runtime files are not permanent.

## Research informing the design

These studies motivate the activities; their findings do not guarantee a particular effect in this workshop or for every learner. Adapt prompts and pacing to participants' knowledge and check comprehension.

- **Worked examples followed by practice:** Van Gog et al. compared worked examples and problem-solving sequences for novices; outcomes depended on the sequence and learner task. This supports runnable examples followed by short changes, not simply passively viewing completed code. [Effects of worked examples, example-problem, and problem-example pairs on novices' learning](https://doi.org/10.1016/j.cedpsych.2010.10.004).
- **Gradually reduce guidance:** Tepgeç and Demiraslan Çevik compared faded and fully worked programming examples in a 2025 experiment. Faded examples improved algorithmic performance in that setting; the groups did not significantly differ in reported cognitive load. This motivates optional short transfer questions alongside runnable examples, not a claim of proven results here. [An experimental evaluation of worked example strategies for efficient programming instruction](https://doi.org/10.1007/s10758-025-09901-2).
- **Explain the reasoning:** Chi et al. analysed learners' self-explanations while studying physics examples. The association motivates asking learners why an example works; the study does not prove that any specific prompt will work for all participants. [Self-Explanations: How Students Study and Use Examples in Learning to Solve Problems](https://doi.org/10.1207/s15516709cog1302_1).
- **Retrieve and revisit:** Karpicke and Roediger studied repeated retrieval of vocabulary and later recall. The results motivate low-stakes prediction and later revisiting; this study alone does not establish the best schedule for an archaeology workshop. [The Critical Importance of Retrieval for Learning](https://doi.org/10.1126/science.1152408).
- **Retrieve conceptual explanations:** Karpicke and Blunt compared retrieval practice with concept mapping for science texts, including questions requiring inferences. This motivates the later recap questions, with feedback available to learners. [Retrieval practice produces more learning than elaborative studying with concept mapping](https://pubmed.ncbi.nlm.nih.gov/21252317/).
- **Feedback after retrieval:** Ludowicy et al. found that performance feedback, alongside correct-answer feedback, improved next-day recall of word pairs in two experiments. This motivates checking an answer and explaining the reasoning, with the caveat that a laboratory memory task differs from this workshop. [Performance feedback enhances test-potentiated encoding](https://pmc.ncbi.nlm.nih.gov/articles/PMC10157468/).
- **Scientific leakage:** Kapoor and Narayanan identified leakage in published ML research and propose systematic documentation of evaluation choices. This supports the explicit train/validation/test distinction and baseline comparison. [Leakage and the reproducibility crisis in machine-learning-based science](https://pmc.ncbi.nlm.nih.gov/articles/PMC10499856/).
- **Spatial dependence:** Kattenborn et al. compared random and spatially blocked validation in drone imagery; the performance gap was study-specific. This motivates questioning the target of generalisation, not assuming every spatial split yields an unbiased answer. [Spatially autocorrelated training and validation samples inflate performance assessment of convolutional neural networks](https://doi.org/10.1016/j.ophoto.2022.100018).
