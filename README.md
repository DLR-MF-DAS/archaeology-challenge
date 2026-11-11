# Archaeology challenge workshop

This project provides the shared Python helpers and two standalone notebooks for
“From Tells to Cyprus”, an archaeological survey prioritisation workshop. See
[START_HERE.txt](START_HERE.txt) for the workshop instructions, data provenance,
licensing notices and scientific limitations. The [facilitator guide](FACILITATOR_GUIDE.md)
contains a teaching plan, discussion feedback and the research informing the design.

## Install

From a checkout, using Python 3.10 or newer:

```sh
python -m pip install -e '.[dev]'
```

The core installation supports the cached-feature workflow. Install `.[model]`
to recompute DINOv2 embeddings, or `.[refresh]` for the instructor's live
Sentinel-2 refresh utility. These optional workflows also require external data
or model downloads. The notebooks remain self-contained and do not require an
installed copy of this package.

Import shared helpers with:

```python
from archaeology_challenge import runtime
```

## Development

Run commands from the repository root. The quick checks validate scientific
helpers and the generated notebook structure:

```sh
python check_notebooks.py
```

`python build_notebooks.py` regenerates both notebooks from the shared runtime
and prepared assets in `.work/`. Those caches are not included in the package;
see `START_HERE.txt` for the data-preparation context. Regeneration clears saved
outputs. `python check_notebooks.py --execute` runs both notebook paths against
the prepared sample and writes instructor outputs, so it requires the local
assets and a suitable Jupyter kernel.
