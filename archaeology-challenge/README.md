# Archaeology challenge workshop

Two standalone notebooks for “From Tells to Cyprus”, an archaeological survey
prioritisation workshop. The six sections cover image inspection, shortlist metrics,
a two-model comparison, mistakes, Cyprus transfer and one written recommendation.
The core path is estimated at about 3½ hours; the four-hour plan adds one optional
lab or catch-up. The instructor copy adds worked exercises and the final test results. See
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

Both feature pipelines now start from the same 64 × 64 RGB input, including Cyprus.
This reduces the original size/detail confound (all Enki tells are 350 × 350, while
most negatives are 64 × 64). The notebooks expose a size-only reference and validation
results within original-size groups. Matching pixel dimensions does not match ground
resolution or remove all acquisition differences. Cyprus has no image-visibility labels;
its scores remain an uncalibrated transfer exercise. Both models' score distributions
and the data audit are included in the exported evidence.

Section 5.3 now asks participants to calculate three Cyprus catalogue measures:
paired ranking agreement (with half credit for ties), the catalogue fraction in a
four-image shortlist, and recovery of the six catalogue locations. It includes an
automatic answer check, random references, tie diagnostics and worked instructor
answers. These measure catalogue agreement, not accuracy for visible archaeology.
The exports include `cyprus_catalogue_metrics.csv` and `cyprus_pairs.csv`.

After updating an older notebook, restart its kernel and run from setup. Both DINO
caches were regenerated; stale preprocessing versions or changed image pixels are
rejected. Local `*-Copy*.ipynb` files are personal snapshots and are not regenerated.

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

Optional PCA, label-efficiency and Cyprus-study labs appear before section 6 in the
notebooks. Choose one before the final test reveal; timing assumptions and facilitation
notes are in the facilitator guide.
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
