# Facilitator guide: From Tells to Cyprus

This guide supports the participant and instructor notebooks. The workshop asks learners to decide whether a model can help an archaeologist prioritise imagery for review. A high model score is a reason to inspect a tile, not evidence that it contains an archaeological site.

## Learning goals

By the end, participants should be able to:

1. Explain what the source labels and the balanced teaching sample represent—and what they do not.
2. Interpret precision and recall at a fixed review budget, and compare image representations using validation data.
3. Explain why model selection and final evaluation use separate data, and why removing similar images does not establish geographic independence.
4. Describe at least two ways the Cyprus imagery and labels differ from the source task, and propose evidence needed for a defensible transfer study.
5. Make a proportionate recommendation supported by the workshop results and their limitations.

The intended audience has basic Python, pandas and scikit-learn familiarity; archaeological or remote-sensing expertise is useful but not assumed. The examples run as supplied. Invite participants to explain and modify them rather than requiring everyone to implement them from scratch.

## Before the session

- Run the participant notebook from top to bottom in a fresh environment. Confirm the Jupyter kernel, dependencies, internet access, and that figures and the final files appear. The default cached-feature path needs a CPU, not a GPU.
- Budget roughly 73 MB of network transfer for the first Enki sample download and about 400 MB of free space. Download time and service availability vary. For weak connectivity, distribute a previously prepared `enki_workshop_sample.zip`; place it in `archaeology_workshop/` before the Enki data cell. It is validated before reuse. The ZIP may be supplied separately from the repository. Package installation still needs network access unless dependencies are already present.
- Review the instructor notebook's outputs and source notices. Do not share it with participants until your planned debrief; it includes worked interpretation and held-out results. Treat the public-source holdout as educational: source labels are recoverable and it is not a secure competition.
- Choose how learners will answer prompts: individually in notes, in pairs, or in a brief group discussion. If tools for displaying notebook widgets are unavailable, text progress displays are acceptable. Set expectations that time estimates vary across groups.

## Four-hour suggested sequence

| Time | Activity | Facilitator action |
|---|---|---|
| 00:00–00:20 | Frame the task and run setup | Define a tell, an image tile and a review budget. Ask what a ranked tile can and cannot establish. |
| 00:20–00:45 | Inspect Enki images and sampling | Ask learners what the labels and image categories mean and what visual shortcuts might exist. |
| 00:45–01:15 | Understand review metrics | Use the ranked example below; let learners predict and explain the denominators before checking. |
| 01:15–01:50 | Compare image features and models | Explain the fixed DINOv2 encoder, the simpler baseline and the roles of training and validation. |
| 01:50–02:00 | Break | If setup is still running, use the metrics and splitting discussions offline. |
| 02:00–02:30 | Explore label efficiency and errors | Ask learners to identify evidence in the example images, then distinguish observation from speculation. |
| 02:30–03:05 | Audit the Cyprus catalogue and imagery | Contrast one settlement with its occupation rows; examine what a catalogue location and an unlabelled location mean. |
| 03:05–03:35 | Freeze the choice and prepare the report | Have teams cite their validation evidence, limitations and proposed next study. |
| 03:35–04:00 | Debrief and final reveal | Discuss competing recommendations, then reveal the held-out results once choices are frozen. |

If the group is behind, keep the three decision points—review metrics, validation versus test, and Cyprus label limitations—and shorten optional model recomputation and extended discussion. Do not require the optional live imagery refresh. Run prerequisite cells in order; skipping an upstream computation can make a later cell fail.

## Prompts and feedback

Give people time to make a prediction or write a one-sentence reason before showing or discussing the explanation. Ask a follow-up such as “Which evidence supports that claim?”; allow pairs to compare answers. These checks are for learning, not high-stakes assessment.

### 1. The sample and the label

**Ask:** “What do `1` and `0` mean here? What does a 50/50 sample tell us about the real landscape?”

**Look for:** `1` and `0` are labels supplied by the Enki archive for tell and negative images. The workshop deliberately selected 300 of each before deduplication; this balance is a sampling choice, not an estimate of operational prevalence or a guarantee of archaeological truth. The source categories describe archive context; they are not verified independent regions.

**If someone says “half of all tiles contain tells”:** distinguish the selected teaching sample from the population of future survey imagery. Ask what sampling design and independent labels would be needed to estimate prevalence.

### 2. What does a shortlist measure?

**Ask:** “In a ranking whose labels are `1, 0, 1, 0, 0, 1, 0, 1, 0, 0`, an expert reviews the first four tiles. What are precision and recall?”

**Look for:** two positives are reviewed. Precision is 2/4 = 0.5 because four were reviewed; recall is 2/4 = 0.5 because there are four positives in the full ten. With a budget of six, precision is 3/6 = 0.5 and recall is 3/4 = 0.75. The example is invented; it is not an Enki result. Average precision summarises the full ranking and is not the same as either single-budget measure.

**If someone uses the same denominator for both:** ask “Of what set are we taking a fraction: the reviewed tiles, or all positives?” Evaluation needs labels; a reviewer making a new shortlist does not know them in advance.

### 3. What does the comparison establish?

**Ask:** “If DINOv2 wins the validation comparison, what can we conclude? Why not choose using the test set?”

**Look for:** compare two representations on the workshop split and stated review budget. A better validation score is evidence about this selected sample, not proof of archaeological detection generally. The DINOv2 encoder is pretrained and frozen; a simple classifier is fitted to each representation on training data. Validation selects among candidates; the held-out test is for assessing the frozen choice. Repeated choices based on test results make it part of model selection.

**If someone says “the pretrained model must win”:** invite them to read the table. A simpler colour/texture baseline might do well because of useful signal, shortcuts or dataset-specific rendering; performance alone does not reveal which explanation is correct.

### 4. Similar images versus new places

**Ask:** “Does removing perceptually similar images guarantee the model will work in a new region?”

**Look for:** no. Filtering reduces some repeated or near-repeated images, but can miss transformed views and does not identify every site or geographic area. Neighbouring landscapes, capture conditions and repeated sites can make random partitions easier than a new-region deployment. This sample lacks verified site groups or coordinates needed to establish geographic independence.

**If someone says “any spatial split is always unbiased”:** explain that evaluation should match the deployment question. A geographically separated, site-independent holdout is a useful design for a new-region claim, but the size and placement of blocks and how sites are sampled still matter. The present data do not support measuring that claim.

### 5. What does Cyprus tell us?

**Ask:** “A catalogue location gets a low score; a nearby unlabelled location gets a high score. Are these a false negative and a false positive?”

**Look for:** not necessarily. A catalogue entry records a location and periods, not whether archaeological remains are visible in the chosen image. An unlabelled location is not a verified absence. Each Cyprus chip is a 1 km square from one Sentinel-2 scene at 10 m per pixel; this differs from the rendered source imagery. Resolution, rendering, target morphology, land cover and location uncertainty may all matter. Scores permit an interpretation exercise, not a numerical target-domain detection accuracy.

**If someone treats an unlabelled location as a negative:** ask what independent expert visibility labels, confirmed absences, imagery and geographically separated sites would be needed before calculating error rates. A site can have several occupation-period rows; counting rows as separate sites inflates the apparent number of sites.

### 6. Recommendation and transfer

Ask each group for a concise claim, one supporting measurement, one limitation, and the next decision or piece of evidence. Accept “do not deploy yet” when it follows from the evidence. Separate observations in the plots from proposed explanations, and avoid treating scores as calibrated probabilities in Cyprus.

For an optional later follow-up, revisit the shortlist and transfer prompts with a different example and ask learners to reconstruct their reasoning before comparing with feedback. A single workshop does not itself establish long-term retention.

## If something goes wrong

- **Cell reports an undefined name:** restart the kernel if needed, then rerun setup and every prerequisite cell in order. Restarting clears variables; save any written reflections first.
- **Download fails or the server ignores range requests:** retry later or use the facilitator-supplied sample ZIP at the expected path. Do not rename the full archive to the sample ZIP name. The optional Cyprus refresh is not required.
- **Model weights or GPU unavailable:** leave DINOv2 recomputation disabled and use the supplied feature cache. The default task uses no live GPU or model download.
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
