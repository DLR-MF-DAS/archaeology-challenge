# Archaeology challenge workshop

Two standalone notebooks for “From Tells to Cyprus”, an archaeological survey
prioritisation workshop. The six sections cover image inspection, shortlist metrics,
a two-model comparison, mistakes, Cyprus transfer and one written recommendation.
The instructor copy adds discussion notes and the final test results. See
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

Optional label-efficiency and catalogue-audit recipes are in the facilitator guide.
Preparing new DINOv2 features requires `uv run --extra model ...`; the
`runtime.make_cyprus_bundle` acquisition utility requires `uv run --extra refresh ...`.
These are preparation tasks with additional network/model downloads, outside the main lesson.
