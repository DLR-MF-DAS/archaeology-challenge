"""Rebuild both regions' pinned features from the standalone notebook assets.

Run with the model and dev extras. No private ZIP index or live Sentinel download
is required; normal notebook participants only load the resulting caches.
"""
import argparse
import base64
import io
import json
import os
from pathlib import Path
import sys
import zipfile

HERE = Path(__file__).resolve().parent


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=HERE / ".work/prepared-assets")
    parser.add_argument("--archive", type=Path,
                        default=HERE / "archaeology_workshop/enki_workshop_sample.zip")
    parser.add_argument("--local-model", type=Path, help="Local copy of the pinned DINOv2 revision")
    args = parser.parse_args()
    os.environ.setdefault("ARCHAEOLOGY_WORKSHOP_ROOT", str(HERE / ".work/preparation"))
    os.environ.setdefault("MPLCONFIGDIR", str(HERE / ".work/matplotlib"))
    sys.path.insert(0, str(HERE / "src"))
    import pandas as pd
    from archaeology_challenge import runtime as w
    from build_notebooks import embedded_assets

    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    # Recover original imagery and sample membership from the committed bundle.
    with zipfile.ZipFile(io.BytesIO(base64.b64decode(embedded_assets()))) as bundle:
        for name in bundle.namelist():
            target = output / name
            if not target.resolve().is_relative_to(output):
                raise ValueError("Unsafe bundled asset path")
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(bundle.read(name))

    archive = w.download_enki_sample(output / "sample_manifest.json", args.archive)
    frame = w.index_enki(archive, w.ROOT / "images")
    _, _, audit = w.deduplicate_images(frame)
    cyprus = pd.read_csv(output / "cyprus/manifest.csv")
    cyprus["path"] = [str(output / "cyprus" / name) for name in cyprus.filename]
    vectors = w.extract_dino(frame.path.tolist() + cyprus.path.tolist(), local_model=args.local_model)
    w.save_embedding_cache(frame, output / "enki_embeddings.npz", vectors[:len(frame)])
    w.save_embedding_cache(cyprus, output / "cyprus/embeddings.npz", vectors[len(frame):])
    w.load_embedding_cache(frame, output / "enki_embeddings.npz")
    w.load_cyprus(output)
    audit.update(image_preprocessing=w.IMAGE_PREPROCESS_ID, dino_preprocessing=w.PREPROCESS_ID,
                 model_id=w.MODEL_ID, revision=w.MODEL_REVISION,
                 enki_images=len(frame), cyprus_images=len(cyprus),
                 native_sizes=frame.groupby(["label", "width", "height"]).size()
                 .rename("images").reset_index().to_dict("records"))
    (output / "preparation.json").write_text(json.dumps(audit, indent=2) + "\n")
    print(json.dumps(audit, indent=2), flush=True)
    print(f"Prepared {output}. Rebuild with: python build_notebooks.py --assets-dir {output}")


if __name__ == "__main__":
    main()
