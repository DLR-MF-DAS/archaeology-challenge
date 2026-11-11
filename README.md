# Archaeology challenge workshop

This project provides the shared Python helpers and two standalone notebooks for
“From Tells to Cyprus”, an archaeological survey prioritisation workshop. See
[START_HERE.txt](START_HERE.txt) for the workshop instructions, data provenance,
licensing notices and scientific limitations. The [facilitator guide](FACILITATOR_GUIDE.md)
contains a teaching plan, discussion feedback and the research informing the design.

## Install and run

Install [uv](https://docs.astral.sh/uv/getting-started/installation/) and run
these commands from the checkout (Python 3.10 or newer):

```sh
uv sync --locked
ARCHAEOLOGY_SKIP_INSTALL=1 uv run --locked jupyter notebook archaeology_participants.ipynb
```

`uv` creates the project environment and installs the locked dependencies.
The environment variable tells the notebook setup cell to use those dependencies
without running its standalone installation step. On Windows, set it in the
shell before launching Jupyter.
The core installation supports the cached-feature workflow. To recompute
DINOv2 embeddings, run with `uv run --extra model ...`; the instructor live
Sentinel-2 refresh utility requires `uv run --extra refresh ...`. These optional
workflows also require external data or model downloads. The notebooks remain
self-contained and can also be used in Colab without this project environment.

## Development

Run the quick checks from the repository root:

```sh
uv run --locked --extra dev python check_notebooks.py
```

`uv run --locked --extra dev python build_notebooks.py` regenerates both
notebooks from the shared runtime and prepared assets in `.work/`. Those caches
are not included in the package; see `START_HERE.txt` for data preparation.
Regeneration clears saved outputs. `uv run --locked --extra dev python check_notebooks.py --execute` runs both notebook paths against the prepared
sample and writes instructor outputs, so it requires local assets and a
suitable Jupyter kernel.
