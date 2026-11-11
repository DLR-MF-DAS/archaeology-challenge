"""Prepare real-image caches once. Normal Colab participants do not run this script."""
import os

from pathlib import Path
import json
import numpy as np


def main():
    os.environ.setdefault("ARCHAEOLOGY_WORKSHOP_ROOT", ".work/data")
    os.environ.setdefault("MPLCONFIGDIR", ".work/matplotlib")
    from archaeology_challenge import runtime as w

    archive = w.download_enki_sample('.work/sample_manifest.json', w.ROOT / 'enki_workshop_sample.zip')
    frame = w.index_enki(archive, w.ROOT / "images")
    frame.to_csv(".work/enki_index.csv", index=False)
    clean, audit, report = w.deduplicate_images(frame)
    audit.to_csv(".work/enki_audit.csv", index=False)
    Path(".work/audit.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2), flush=True)
    print(frame.groupby(["label", "category"]).size().to_string(), flush=True)
    cache_path = Path(".work/sample_embeddings.npz")
    if cache_path.exists():
        embeddings = w.load_embedding_cache(frame, cache_path)
    else:
        embeddings = w.extract_dino(frame.path.tolist(), local_model=".work/model")
        np.savez_compressed(cache_path, tile_ids=frame.tile_id.to_numpy(dtype=str),
                            pixel_sha256=frame.pixel_sha256.to_numpy(dtype=str), embeddings=embeddings,
                            model_id=w.MODEL_ID, revision=w.MODEL_REVISION, preprocessing=w.PREPROCESS_ID)
    print("Prepared embeddings:", embeddings.shape, flush=True)


if __name__ == "__main__":
    main()
