# Archaeology challenge workshop

Two standalone notebooks for “From Tells to Cyprus”, an archaeological survey
prioritisation workshop. The six sections cover image inspection, shortlist metrics,
a two-model comparison, mistakes, Cyprus transfer and one written recommendation.
The audience knows basic Python; no prior machine learning or pandas is required.
Allow roughly three to three-and-a-half hours for the core, with one optional lab
and discussion in the four-hour plan. This is a planning estimate, not a timed learner study.
The instructor copy adds worked exercises and the final test results. See
[START_HERE.txt](START_HERE.txt) for the workshop instructions, data provenance,
licensing notices and scientific limitations. The [facilitator guide](FACILITATOR_GUIDE.md)
contains a teaching plan, discussion feedback and the research informing the design.

## Install and run

Install [uv](https://docs.astral.sh/uv/getting-started/installation/) and run
these commands from the repository root (Python 3.10 or newer):

```sh
cd archaeology-challenge
uv sync --locked
uv run --locked jupyter notebook archaeology_participants.ipynb
```

`uv` creates the project environment and installs the locked dependencies.
Setup installs only missing dependencies, so an installed local environment is reused.
The notebooks also work in Colab without this project environment. They include
frozen image features and Cyprus assets; the Enki images download on first use.
No GPU or model download is needed. `ARCHAEOLOGY_SKIP_INSTALL=1` disables installation
entirely for offline validation.

The notebooks explain features, training, validation and ranking before asking learners
to interpret results. The two calculation exercises use counts printed immediately
above them, with feedback after running. Editable notes cells collect evidence for
the final recommendation. Model comparisons show counts, precision and recall in
plain-language columns; technical audit measures are reserved for the export and notes below.

The notebooks keep calculation code, training code and settings visible. Cells marked **Supplied**
contain loading, plotting, checking or export code and start folded in frontends
that support the metadata. Run these cells normally; expand them to inspect the code.
After changing lab or reveal settings, run the settings cell and then its supplied
runner. Detailed teaching prompts stay in the facilitator guide.

Both feature pipelines now start from the same 64 × 64 RGB input, including Cyprus.
This reduces the original size/detail confound (all Enki tells are 350 × 350, while
most negatives are 64 × 64). Section 3.6 explains this problem with the example of
guessing labels from image dimensions. The detailed size-only reference and results
within original-size groups remain in the exported audit. Matching pixel dimensions does not match ground
resolution or remove all acquisition differences. Cyprus has no image-visibility labels;
its scores remain an uncalibrated transfer exercise. Both models' score distributions
and the data audit are included in the exported evidence.

Section 3.4 prints how many images the colour/texture model would have reviewed,
how many are tells, and how many tells exist in the full validation set. Participants
use those counts to calculate precision and recall, with an immediate answer check.
The subsequent model comparison displays both measures as percentages.

Section 5.3 asks participants to calculate three Cyprus catalogue measures:
paired ranking agreement (with half credit for ties), the catalogue fraction in a
four-image shortlist, and recovery of the six catalogue locations. It includes an
automatic answer check, explained random references, tie diagnostics and worked instructor
answers. These measure catalogue agreement, not accuracy for visible archaeology.
The exports include `cyprus_catalogue_metrics.csv` and `cyprus_pairs.csv`.

Section 6 brings the selected model's Enki and Cyprus evidence together beside a
five-sentence recommendation prompt. Participants save and share the evidence ZIP
and their notebook, then compare validation with the reserved Enki test results at
the saved review budget. A one-sentence debrief records what the test supports and
what remains untested, preserving the original recommendation.

After updating an older notebook, restart its kernel and run from setup. Both DINO
caches were regenerated; stale preprocessing versions or changed image pixels are
rejected. Local `*-Copy*.ipynb` files are personal snapshots and are not regenerated.

## Setup help

- For unreliable internet, obtain the prepared `enki_workshop_sample.zip` from the
  facilitator and place it in `archaeology_workshop/` before loading images. Keep its
  filename; the full source archive is not interchangeable with this sample ZIP.
- After a reset or an undefined-name error, rerun from setup in order. Save written
  answers first. Colab runtime files are temporary, so download the ZIP and notebook.
- A skipped lab message means its flag is still `False`. Run the supplied function
  as given, run its settings cell with the flag enabled, then run the supplied plotting
  cell. After changing a setting, rerun settings and plotting; no earlier sections need rerunning.

## Submission files

Section 6 exports `team_submission.zip`. Rerunning export replaces it with the
current results and recommendation. Save the notebook separately to retain exercise
notes, figures and the final debrief; these are not all included in the ZIP.

| File in the ZIP | Contents |
|---|---|
| `submission.csv` | Test image IDs, scores and ranks, without labels. |
| `configuration.json` | Selected model, settings, review budgets and provenance. |
| `validation_results.csv` | Both models' validation results used for selection. |
| `cyprus_review.csv` | Cyprus scores and location metadata. |
| `data_audit.json` | Duplicate audit, original image sizes and common preprocessing. |
| `validation_diagnostics.csv` | Size-only reference and validation AP within original dimensions. |
| `transfer_diagnostics.csv` | Both models' score distributions on Enki validation and Cyprus. |
| `cyprus_catalogue_metrics.csv` | Catalogue agreement, shortlist recovery and random references. |
| `cyprus_pairs.csv` | Scores, margins and outcomes for each pair and model. |
| `recommendation.md` | The written recommendation made before the test reveal. |

## Data and model details

- **Enki:** [Rosati (2025)](https://zenodo.org/records/14950565), a 600-image sample from
  `Normalized_Tell_combined_350.zip` (whole-archive MD5 `1d4ea398ea5669f9bca4539416847206`).
  Member CRCs and decoded-pixel hashes verify the partial download. Positive examples
  use five RGB categories; masked and black-and-white positive folders are excluded.
  The source declares CC BY-SA 4.0 and separate Google/Bing imagery conditions; retain
  its notices. The balanced sample does not estimate tell prevalence in a real survey.
- **Cyprus catalogue:** [Crawford & Vella (2022)](https://doi.org/10.5334/joad.96), CC BY 4.0;
  [pinned revision](https://github.com/K-A-Crawford/Cyprus_Settlements/tree/70a5c3d399a2451e1d8dd5eb947eb80055276b44).
  The bundle contains 8,766 occupation rows describing 1,559 settlement IDs.
- **DINOv2:** [`facebook/dinov2-small`](https://github.com/facebookresearch/dinov2),
  revision `ed25f3a31f01632728cabb09d1542f84ab7b0056`. Inputs are resized bicubically
  to 64 × 64, enlarged to 224 × 224, then ImageNet-normalised. The frozen CLS features
  are L2-normalised. Caches bind image IDs to pixel hashes, weights and preprocessing.
- **Cyprus imagery:** modified Copernicus Sentinel data (2024), via
  [Microsoft Planetary Computer](https://planetarycomputer.microsoft.com/dataset/sentinel-2-l2a).
  Scene `S2A_MSIL2A_20240805T082601_R021_T36SVD_20240805T144559` supplies 10 m visible
  bands. Nearby centres are 2 km east and at least 750 m from catalogue points;
  chips have at least 90% clear-land pixels. Full acquisition metadata is bundled
  in `assets/cyprus/provenance.json`.

Both logistic regressions use `C=1`, `class_weight="balanced"`, `max_iter=2000` and
seed `42`. Smaller `C` imposes a stronger penalty on large weights. The class weighting
has no effect on class totals in the balanced training sample. See the
[scaler](https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.StandardScaler.html)
and [classifier](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html)
references for their parameters.

### Technical audit: original image sizes

The retained sample has 300 tell images originally at 350 × 350 pixels, 249 negatives
at 64 × 64 and 44 negatives at 350 × 350. Original dimensions are therefore associated
with the supplied label. The audit tests a simple rule that uses only width and height:
learn the tell fraction for each original size from training images, then predict
tell when that fraction is at least one-half. On the default validation set it gets
110 of 119 labels correct (92.4%), without inspecting any pixels. This is the accuracy
of that metadata rule, not the accuracy of either image classifier. It demonstrates
a collection problem; it does not prove which cues the fitted image classifiers use.

`validation_diagnostics.csv` also evaluates each image classifier separately on images
that had the same original dimensions. The 350 × 350 validation group contains 60 tells
and nine negatives; the 64 × 64 group contains 50 negatives and no tells. This check
helps separate ranking within a size group from separating the size groups themselves.
It cannot rule out other collection differences, and it is not a geographic evaluation.

For this optional audit, **average precision (AP)** summarises the whole ranking, rather
than just its first 50 images. Higher values mean tell-labelled images tend to appear
earlier; 1 is a perfect ranking. For untied labels `[1, 1, 0, 1, 0]`,
average precision at the positive ranks: `(1/1 + 2/2 + 3/4) / 3 ≈ 0.917`.
The workshop uses scikit-learn's [AP implementation](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.average_precision_score.html),
including its handling of ties. The `positive_fraction` column is simply tells divided
by all images in the group: `60/69 ≈ 0.870` in the 350-pixel group. This is a population
reference for a ranking with no useful information, not the exact expected AP of every
finite random sample. Default within-group AP is 0.996 for DINOv2 and 0.960 for
colour/texture. AP is not reported for groups containing only one label because they
cannot test separation of tells from negatives. These technical measures are not
required for the participant exercises or used to break model-selection ties.

`transfer_diagnostics.csv` preserves both models' score distributions on Enki and Cyprus.
It may reveal changes in the outputs, but cannot measure Cyprus detection accuracy
without suitable image labels. These tables are computed during export, away from
the core learning sequence.

## Development

Run the quick checks from the workshop directory:

```sh
uv run --locked --extra dev python check_notebooks.py
```

`uv run --locked --extra dev python build_notebooks.py` regenerates both notebooks
from the shared runtime, reusing the frozen assets in the participant notebook.
It works from a fresh checkout without `.work/`. Use `--assets-dir PATH` only when
replacing the bundle with prepared data in the unpacked assets layout.

Regeneration clears outputs. `uv run --locked --extra dev python check_notebooks.py --execute`
runs both standalone notebooks in isolated working directories using the current Python
environment. It reuses `archaeology_workshop/enki_workshop_sample.zip` or
`.work/data/enki_workshop_sample.zip` when available, otherwise the notebooks download
the sample. It saves outputs in the instructor notebook and writes
`.work/participant_validation.ipynb` separately.

Optional labs appear before section 6. Choose one before the final test reveal.
All labs supply complete working functions; the tasks are experiments and interpretation,
not syntax completion. Lab A explains PCA as a compressed picture of feature values.
In lab B, learners predict the effect of more training examples, compare 20/50-image
training sets with the 200-image classifier, inspect tells entering and leaving the
50-image shortlist, and record three controlled runs. Lab C explains the difference
between a classifier score and an image-similarity score, then asks learners to inspect
named Cyprus/Enki matches and compare neighbour-label counts. Both labs have
worked instructor results. Timing and teaching notes are in the guide.
Preparing new DINOv2 features requires `uv run --extra model ...`; the
`runtime.make_cyprus_bundle` acquisition utility requires `uv run --extra refresh ...`.
These are preparation tasks with additional network/model downloads, outside the main lesson.

To rebuild the frozen features reproducibly from the committed bundle and the original
600-image sample, run from this directory:

```sh
uv run --locked --extra model --extra dev python prepare_workshop.py
uv run --locked --extra dev python build_notebooks.py --assets-dir .work/prepared-assets
uv run --locked --extra dev python check_notebooks.py --execute
```

Preparation downloads the pinned DINOv2-small weights once and rebuilds both regions
in one pass. It reuses the verified local sample ZIP, downloading the indexed ranges
only if needed. `--archive PATH` selects another copy of that sample;
`--local-model PATH` permits an offline copy of the exact pinned weights.
No private ZIP index or new satellite acquisition is needed. The notebook builder
embeds the updated features, original Cyprus chips and current helper source.
