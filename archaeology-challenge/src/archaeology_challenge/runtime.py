"""Shared source embedded into both standalone Colab notebooks by build_notebooks.py."""
from pathlib import Path
import hashlib
import io
import json
import os
import time
import zipfile
import struct
import zlib

import imagehash
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import requests
from PIL import Image
from sklearn.base import clone
from sklearn.metrics import average_precision_score
from sklearn.model_selection import train_test_split
from tqdm.auto import tqdm

SEED = 42
ENKI_URL = "https://zenodo.org/records/14950565/files/Val_norm.zip?download=1"
ENKI_MD5 = "fbd56ed8039b946eadb6481badf80c95"
ENKI_TRAIN_URL = "https://zenodo.org/records/14950565/files/Normalized_Tell_combined_350.zip?download=1"
CYPRUS_REVISION = "70a5c3d399a2451e1d8dd5eb947eb80055276b44"
CYPRUS_URL = f"https://raw.githubusercontent.com/K-A-Crawford/Cyprus_Settlements/{CYPRUS_REVISION}/Cyprus_Settlements.csv"
MODEL_ID = "facebook/dinov2-small"
MODEL_REVISION = "ed25f3a31f01632728cabb09d1542f84ab7b0056"
COMMON_IMAGE_SIZE = 64
IMAGE_PREPROCESS_ID = "rgb-full-frame-bicubic-64-v2"
PREPROCESS_ID = IMAGE_PREPROCESS_ID + "-bicubic-224-imagenet-mean-std-cls-l2"
CYPRUS_ITEM = "S2A_MSIL2A_20240805T082601_R021_T36SVD_20240805T144559"
ROOT = Path(os.environ.get("ARCHAEOLOGY_WORKSHOP_ROOT", "archaeology_workshop"))


def file_digest(path, algorithm="sha256"):
    h = hashlib.new(algorithm)
    with open(path, "rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def pixel_digest(image):
    """Hash decoded RGB pixels, including dimensions, independently of the file encoding."""
    rgb = image.convert("RGB")
    return hashlib.sha256(str(rgb.size).encode() + rgb.tobytes()).hexdigest()


def image_pixel_digest(path):
    with Image.open(path) as image:
        return pixel_digest(image)


def image_input(path):
    """Common pixel grid for both feature branches and both regions.

    Downsample before DINO's required upsampling so native 350px positives cannot
    retain detail unavailable in the 64px negatives. This does not match ground
    sampling distance or remove every acquisition/appearance confound.
    """
    with Image.open(path) as image:
        return image.convert("RGB").resize(
            (COMMON_IMAGE_SIZE, COMMON_IMAGE_SIZE), Image.Resampling.BICUBIC,
        )


def dino_input(path):
    """Enlarge the shared low-resolution input; do not bypass it for large originals."""
    return np.asarray(image_input(path).resize((224, 224), Image.Resampling.BICUBIC),
                      dtype=np.float32).copy() / 255


def download(url, destination, md5=None):
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists() and (md5 is None or file_digest(destination, "md5") == md5):
        return destination
    part = destination.with_suffix(destination.suffix + ".part")
    for attempt in range(3):
        try:
            with requests.get(url, stream=True, timeout=(30, 120)) as response:
                response.raise_for_status()
                total = int(response.headers.get("content-length", 0)) or None
                with open(part, "wb") as out, tqdm(total=total, unit="B", unit_scale=True,
                                                    desc=destination.name) as progress:
                    for block in response.iter_content(1024 * 1024):
                        out.write(block)
                        progress.update(len(block))
            if md5 is not None and file_digest(part, "md5") != md5:
                raise ValueError("Checksum mismatch: the download is incomplete or has changed.")
            part.replace(destination)
            return destination
        except (requests.RequestException, ValueError):
            if attempt == 2:
                raise
            time.sleep(2 ** attempt)


def download_enki_sample(manifest_path, destination):
    """Fetch indexed byte ranges, validate ZIP-member CRCs, build a small local ZIP.

    The embedded manifest was prepared from the original ZIP central directory. If a server
    ignores byte ranges, abort rather than accidentally downloading the 1.7 GB archive.
    """
    manifest = json.loads(Path(manifest_path).read_text())
    rows = manifest["members"]
    expected = {r["member"]: r for r in rows}
    destination = Path(destination)
    if destination.exists():
        with zipfile.ZipFile(destination) as z:
            if set(z.namelist()) == set(expected) and all(
                z.getinfo(name).CRC == row["crc"] for name, row in expected.items()
            ) and z.testzip() is None:
                return destination
    cache_dir = destination.parent / "range_cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    partial = destination.with_suffix(".part")
    groups = sorted(set(r["batch"] for r in rows))
    with requests.Session() as session, zipfile.ZipFile(partial, "w", zipfile.ZIP_DEFLATED) as output:
        for batch in tqdm(groups, desc="Download compact balanced Enki sample"):
            group = [r for r in rows if r["batch"] == batch]
            start = min(r["offset"] for r in group)
            end = max(r["span_end"] for r in group)
            cached = cache_dir / f"{start}-{end}.bin"
            if cached.exists() and cached.stat().st_size == end - start + 1:
                data = cached.read_bytes()
            else:
                for attempt in range(4):
                    try:
                        with session.get(ENKI_TRAIN_URL, headers={"Range": f"bytes={start}-{end}"},
                                         stream=True, timeout=(30, 120)) as response:
                            if response.status_code == 429:
                                delay = min(60, int(response.headers.get("Retry-After", "30")))
                                time.sleep(delay)
                                continue
                            response.raise_for_status()
                            if response.status_code != 206:
                                raise RuntimeError("Server did not honour byte ranges. Use the facilitator's enki_workshop_sample.zip instead.")
                            actual_range = response.headers.get("Content-Range", "")
                            if not actual_range.startswith(f"bytes {start}-{end}/"):
                                raise RuntimeError(f"Unexpected byte range: {actual_range}")
                            data = response.content
                        if len(data) != end - start + 1:
                            raise ValueError("Truncated range download")
                        cached.write_bytes(data)
                        break
                    except requests.RequestException:
                        if attempt == 3:
                            raise
                        time.sleep(2 ** attempt)
                else:
                    raise RuntimeError("Zenodo rate limit persisted; retry later or use the facilitator's sample ZIP.")
            for row in group:
                position = row["offset"] - start
                assert data[position:position + 4] == b"PK\x03\x04"
                name_size, extra_size = struct.unpack_from("<HH", data, position + 26)
                payload_start = position + 30 + name_size + extra_size
                compressed = data[payload_start:payload_start + row["compressed_size"]]
                if row["method"] == 8:
                    payload = zlib.decompress(compressed, -15)
                elif row["method"] == 0:
                    payload = compressed
                else:
                    raise ValueError("Unsupported ZIP compression method")
                assert len(payload) == row["file_size"]
                assert zlib.crc32(payload) & 0xFFFFFFFF == row["crc"], "Archive member checksum mismatch"
                info = zipfile.ZipInfo(row["member"], date_time=(2025, 3, 1, 0, 0, 0))
                info.compress_type = zipfile.ZIP_DEFLATED
                output.writestr(info, payload)
    partial.replace(destination)
    return destination


def index_enki(archive, image_dir):
    """Read only image members, infer classes from actual path components, anonymise paths."""
    image_dir = Path(image_dir)
    image_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    with zipfile.ZipFile(archive) as source:
        members = sorted(source.namelist())
        for name in tqdm(members, desc="Index and audit images"):
            parts = Path(name).parts
            if "__MACOSX" in parts or any(p.startswith(".") for p in parts):
                continue
            if Path(name).suffix.lower() not in {".png", ".jpg", ".jpeg", ".tif", ".tiff"}:
                continue
            classes = set(p.lower() for p in parts) & {"positive", "tell", "negative"}
            if len(classes) != 1:
                raise ValueError(f"Unrecognised class directory: {name}")
            label = int(bool(classes & {"positive", "tell"}))
            tile_id = hashlib.sha256(name.encode()).hexdigest()[:16]
            target = image_dir / f"{tile_id}.png"
            with Image.open(io.BytesIO(source.read(name))) as original:
                rgb = original.convert("RGB")
                pixel_hash = pixel_digest(rgb)
                phash = str(imagehash.phash(rgb))
                # Recover from stale or locally edited extracted images, not just a missing file.
                if not target.exists() or image_pixel_digest(target) != pixel_hash:
                    rgb.save(target)
                width, height = rgb.size
            rows.append(dict(tile_id=tile_id, member=name, path=str(target), label=label,
                             category=parts[-2], pixel_sha256=pixel_hash, phash=phash,
                             width=width, height=height))
    frame = pd.DataFrame(rows).sort_values("tile_id").reset_index(drop=True)
    assert frame.tile_id.is_unique and set(frame.label) == {0, 1}
    return frame


def deduplicate_images(frame, max_hamming=4):
    """Connected components of close pHashes; exclude mixed-label components; keep one/image family.

    This is an image-similarity audit, NOT a verified geographic/site grouping.
    """
    hashes = [int(x, 16) for x in frame.phash]
    parent = list(range(len(frame)))

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    for i, value in enumerate(hashes):
        for j in range(i):
            if (value ^ hashes[j]).bit_count() <= max_hamming:
                parent[find(i)] = find(j)
    audit = frame.copy()
    audit["image_family"] = [f"family_{find(i):04d}" for i in range(len(frame))]
    conflicts = audit.groupby("image_family").label.nunique()
    bad = set(conflicts[conflicts > 1].index)
    clean = (audit.loc[~audit.image_family.isin(bad)]
             .drop_duplicates("image_family").reset_index(drop=True))
    report = {
        "raw_images": len(frame), "exact_duplicate_images": int(frame.pixel_sha256.duplicated().sum()),
        "similarity_families": audit.image_family.nunique(), "mixed_label_families_excluded": len(bad),
        "retained_images": len(clean), "phash_max_hamming": max_hamming,
        "geographic_independence_verified": False,
    }
    return clean, audit, report


def workshop_split(clean, per_class=100):
    # One representative per similarity family has already been selected.
    trainval, test = train_test_split(clean, test_size=0.2, stratify=clean.label, random_state=SEED)
    train_pool, val = train_test_split(trainval, test_size=0.25,
                                     stratify=trainval.label, random_state=SEED)
    train = pd.concat([g.sample(min(per_class, len(g)), random_state=SEED)
                       for _, g in train_pool.groupby("label")])
    splits = {"train": train, "validation": val, "test": test}
    for name, part in splits.items():
        splits[name] = part.sort_values("tile_id").reset_index(drop=True)
    assert not (set(train.image_family) & set(val.image_family))
    assert not (set(train.image_family) & set(test.image_family))
    assert not (set(val.image_family) & set(test.image_family))
    return splits


def contact_sheet(frame, scores=None, n=12, title="Image tiles", show_labels=True):
    subset = frame.head(n).reset_index(drop=True)
    if subset.empty:
        print(f"{title}: none.")
        return
    fig, axes = plt.subplots(int(np.ceil(len(subset) / 4)), 4,
                             figsize=(12, 3.1 * int(np.ceil(len(subset) / 4))), squeeze=False)
    for ax in axes.flat:
        ax.axis("off")
    for i, row in subset.iterrows():
        ax = axes.flat[i]
        with Image.open(row.path) as im:
            ax.imshow(im)
        label = f"label {row.label}" if show_labels and "label" in row else ""
        if "kind" in row:
            label = "catalogued location" if row.kind == "catalogued_location" else "unlabelled location"
        score = f"score {float(scores[i]):.3f}" if scores is not None else ""
        details = " · ".join(part for part in (label, score) if part)
        ax.set_title(f"{row.tile_id}\n{details}", fontsize=9)
    fig.suptitle(title)
    plt.tight_layout()
    plt.show()


def handcrafted_features(paths):
    """Colour distributions, global moments and simple gradient histograms (no pretrained network)."""
    features = []
    for path in tqdm(paths, desc="Colour and texture features"):
        rgb = np.asarray(image_input(path), dtype=np.float32) / 255
        colour = np.concatenate([np.histogram(rgb[:, :, c], bins=16, range=(0, 1), density=False)[0]
                                 / (64 * 64) for c in range(3)])
        grey = rgb.mean(axis=2)
        dy, dx = np.gradient(grey)
        magnitude = np.sqrt(dx ** 2 + dy ** 2)
        angle = np.arctan2(dy, dx)
        orientation = np.histogram(angle, bins=12, range=(-np.pi, np.pi), weights=magnitude)[0]
        orientation /= orientation.sum() + 1e-8
        moments = np.r_[rgb.mean(axis=(0, 1)), rgb.std(axis=(0, 1)), magnitude.mean(), magnitude.std()]
        features.append(np.r_[colour, moments, orientation])
    return np.asarray(features, dtype=np.float32)


def extract_dino(paths, local_model=None, batch_size=16):
    """Frozen CLS embeddings, shared 64px input then 224px, ImageNet and L2 normalisation.

    Model weights and preprocessing are pinned. The same function is used for Enki and Cyprus.
    Imports are lazy so the bundled embedding path also works without torch/transformers.
    """
    import torch
    from transformers import AutoModel

    device = "cuda" if torch.cuda.is_available() else (
        "mps" if torch.backends.mps.is_available() else "cpu")
    torch.manual_seed(SEED)
    torch.set_num_threads(min(4, os.cpu_count() or 1))
    if local_model:
        model = AutoModel.from_pretrained(str(local_model), local_files_only=True)
    else:
        model = AutoModel.from_pretrained(MODEL_ID, revision=MODEL_REVISION,
                                          cache_dir=str(ROOT / "model_cache"))
    model = model.eval().to(device)
    mean = torch.tensor([0.485, 0.456, 0.406])[:, None, None]
    std = torch.tensor([0.229, 0.224, 0.225])[:, None, None]
    vectors = []
    started = time.perf_counter()
    with torch.inference_mode():
        for start in tqdm(range(0, len(paths), batch_size), desc=f"DINOv2 ({device})"):
            batch = []
            for path in paths[start:start + batch_size]:
                arr = dino_input(path)
                batch.append((torch.from_numpy(arr).permute(2, 0, 1) - mean) / std)
            tensor = torch.stack(batch).to(device)
            feature = model(pixel_values=tensor).last_hidden_state[:, 0]
            feature = torch.nn.functional.normalize(feature.float(), dim=1)
            vectors.append(feature.cpu().numpy())
    print(f"Encoded {len(paths)} images in {time.perf_counter() - started:.1f}s on {device}.")
    return np.concatenate(vectors).astype(np.float32)


def save_embedding_cache(frame, cache_path, embeddings):
    """Write the same versioned, pixel-bound cache format for Enki and Cyprus."""
    vectors = np.asarray(embeddings, dtype=np.float32)
    if not frame.tile_id.is_unique or vectors.shape != (len(frame), 384) or not np.isfinite(vectors).all():
        raise ValueError("Expected unique image IDs and one finite 384-dimensional vector per image.")
    hashes = [image_pixel_digest(path) for path in frame.path]
    np.savez_compressed(cache_path, tile_ids=frame.tile_id.to_numpy(dtype=str),
                        pixel_sha256=np.asarray(hashes), embeddings=vectors,
                        model_id=MODEL_ID, revision=MODEL_REVISION, preprocessing=PREPROCESS_ID)


def load_embedding_cache(frame, cache_path):
    with np.load(cache_path, allow_pickle=False) as cache:
        required = {"model_id", "revision", "preprocessing", "tile_ids", "pixel_sha256", "embeddings"}
        if not required.issubset(cache.files) or any(
            str(cache[key]) != value for key, value in (
                ("model_id", MODEL_ID), ("revision", MODEL_REVISION), ("preprocessing", PREPROCESS_ID))
        ):
            raise ValueError("Stale feature cache: rebuild both regions with prepare_workshop.py.")
        ids = cache["tile_ids"].astype(str)
        vectors = cache["embeddings"]
        if (not frame.tile_id.is_unique or len(set(ids)) != len(ids)
                or vectors.shape != (len(ids), 384) or cache["pixel_sha256"].shape != (len(ids),)
                or not np.isfinite(vectors).all()):
            raise ValueError("Invalid feature cache: duplicate IDs, invalid dimensions or nonfinite vectors.")
        positions = {key: i for i, key in enumerate(ids)}
        if not set(frame.tile_id).issubset(positions):
            raise ValueError("Feature cache is missing image IDs.")
        indices = [positions[key] for key in frame.tile_id]
        expected = cache["pixel_sha256"][indices].astype(str)
        if not np.array_equal(expected, frame.pixel_sha256.to_numpy()):
            raise ValueError("Image/cache mismatch: pixels changed; rebuild the features.")
        return vectors[indices].astype(np.float32)


def source_size_scores(train, frame):
    """Metadata-only confound probe; learn size/label associations on training rows only."""
    frequencies = train.groupby(["width", "height"]).label.mean()
    prior = float(train.label.mean())
    return np.asarray([frequencies.get((row.width, row.height), prior)
                       for row in frame.itertuples()], dtype=float)


def validation_diagnostics(train, validation, scores, k=50):
    """Expose the size shortcut and evaluate each image model within native-size strata.

    AP is undefined for single-class strata. Positive fraction is the no-skill
    population reference for AP, not an exact finite-sample expected random AP.
    """
    probe = source_size_scores(train, validation)
    rows = [dict(model="source_size_only", subset="all", images=len(validation),
                 positives=int(validation.label.sum()), positive_fraction=float(validation.label.mean()),
                 accuracy=float(((probe >= .5) == validation.label).mean()),
                 **ranking_metrics(validation.label, probe, k))]
    for name, values in scores.items():
        for (width, height), indices in validation.groupby(["width", "height"]).indices.items():
            labels = validation.iloc[indices].label.to_numpy()
            both_classes = len(np.unique(labels)) == 2
            rows.append(dict(model=name, subset=f"native_{width}x{height}", images=len(labels),
                             positives=int(labels.sum()), positive_fraction=float(labels.mean()),
                             AP=float(average_precision_score(labels, np.asarray(values)[indices]))
                             if both_classes else float("nan")))
    return pd.DataFrame(rows)


def transfer_summary(validation_labels, validation_scores, target_scores):
    """Describe score distributions without assigning target labels or claiming calibration."""
    rows = []
    labels = np.asarray(validation_labels)
    for name, values in validation_scores.items():
        for group, scores in (
            ("Enki validation negatives", np.asarray(values)[labels == 0]),
            ("Enki validation tells", np.asarray(values)[labels == 1]),
            ("Cyprus (unlabelled)", np.asarray(target_scores[name])),
        ):
            rows.append(dict(model=name, group=group, images=len(scores),
                             minimum=float(np.min(scores)), median=float(np.median(scores)),
                             maximum=float(np.max(scores))))
    return pd.DataFrame(rows)


def catalogue_agreement(frame, score_column="score", k=4):
    """Measure catalogue recovery and paired ordering, not archaeological accuracy.

    Each pair must contain exactly one catalogued and one unlabelled location.
    Exact score ties earn half credit in paired agreement. Shortlist ties use
    tile-ID order; report the possible hit range when the cutoff crosses a tie.
    No unlabelled location is asserted to be an archaeological negative.
    """
    required = {"tile_id", "paired_site", "kind", score_column}
    if not required.issubset(frame.columns) or frame.empty:
        raise ValueError("Supply nonempty scored catalogue pairs with image IDs.")
    if not isinstance(k, (int, np.integer)) or isinstance(k, bool) or k < 1:
        raise ValueError("The Cyprus review budget must be a positive integer.")
    if frame[list(required)].isna().any().any() or not frame.tile_id.is_unique:
        raise ValueError("Catalogue pair IDs, kinds and scores must be present; image IDs must be unique.")
    kinds = {"catalogued_location", "unlabelled_background"}
    if set(frame.kind) != kinds or frame.duplicated(["paired_site", "kind"]).any():
        raise ValueError("Each catalogue pair needs exactly one image of each kind.")
    counts = frame.groupby("paired_site").size()
    if not counts.eq(2).all():
        raise ValueError("Each catalogue pair needs exactly one image of each kind.")
    ordered = frame.sort_values("tile_id", kind="stable").reset_index(drop=True)
    scores = ordered[score_column].to_numpy(dtype=float)
    order = ranking_indices(scores)
    pairs = ordered.pivot(index="paired_site", columns="kind", values=score_column).rename(
        columns={"catalogued_location": "catalogued_score", "unlabelled_background": "unlabelled_score"},
    ).reset_index()
    pairs.columns.name = None
    pairs["margin"] = pairs.catalogued_score - pairs.unlabelled_score
    pairs["outcome"] = np.select(
        [pairs.margin.gt(0), pairs.margin.lt(0)],
        ["catalogued higher", "unlabelled higher"], default="tie",
    )
    wins, ties, losses = (int((pairs.margin > 0).sum()), int((pairs.margin == 0).sum()),
                          int((pairs.margin < 0).sum()))
    known = ordered.kind.eq("catalogued_location").to_numpy()
    reviewed = min(k, len(ordered))
    hits = int(known[order[:reviewed]].sum())
    cutoff = scores[order[reviewed - 1]]
    above, tied = scores > cutoff, scores == cutoff
    remaining_slots = reviewed - int(above.sum())
    known_above, known_tied = int(known[above].sum()), int(known[tied].sum())
    min_hits = known_above + max(0, remaining_slots - (int(tied.sum()) - known_tied))
    max_hits = known_above + min(remaining_slots, known_tied)
    return pairs, {
        "pairs": len(pairs), "wins": wins, "ties": ties, "losses": losses,
        "pair_agreement": (wins + .5 * ties) / len(pairs),
        "reviewed": reviewed, "catalogued_at_k": hits, "catalogued_total": int(known.sum()),
        "catalogue_fraction_at_k": hits / reviewed,
        "catalogue_recovery_at_k": hits / int(known.sum()),
        "random_pair_agreement": .5,
        "random_catalogue_fraction": float(known.mean()),
        "random_catalogue_recovery": reviewed / len(ordered),
        "cutoff_tie": bool(reviewed < len(ordered) and scores[order[reviewed]] == cutoff),
        "min_catalogued_at_k": min_hits, "max_catalogued_at_k": max_hits,
    }


def partition_features(frame, features, splits):
    positions = {key: i for i, key in enumerate(frame.tile_id)}
    return {name: features[[positions[k] for k in subset.tile_id]]
            for name, subset in splits.items()}


def ranking_indices(scores):
    values = np.asarray(scores, dtype=float)
    if values.ndim != 1 or not np.isfinite(values).all():
        raise ValueError("Scores must be a finite one-dimensional array.")
    # Stable tie-breaking follows the tile_id-sorted input manifest.
    return np.argsort(-values, kind="stable")


def ranking_metrics(y_true, scores, k=50):
    """Count positives in a fixed-budget shortlist; AP describes the full ranking."""
    labels = np.asarray(y_true)
    order = ranking_indices(scores)
    if labels.ndim != 1 or len(labels) != len(order) or len(labels) == 0:
        raise ValueError("Use nonempty, aligned one-dimensional labels and scores.")
    if not np.isin(labels, [0, 1]).all():
        raise ValueError("Labels must be 0 or 1.")
    if not isinstance(k, (int, np.integer)) or isinstance(k, bool) or k < 1:
        raise ValueError("The review budget must be a positive integer.")
    reviewed = min(k, len(labels))
    hits = int(labels[order[:reviewed]].sum())
    positives = int(labels.sum())
    return {
        "hits": hits,
        "reviewed": reviewed,
        "precision_at_k": hits / reviewed,
        "recall_at_k": hits / positives if positives else float("nan"),
        "AP": float(average_precision_score(labels, scores)) if positives else float("nan"),
    }


def summarise_sites(phases):
    """Keep one record per settlement, flagging conflicting coordinates and quality codes."""
    frame = phases.copy()
    frame["LocQual"] = frame.LocQual.fillna("").astype(str).str.strip()
    for column in ("Long_Cord", "Lat_Cord"):
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    locations = frame[["ID", "Long_Cord", "Lat_Cord"]].drop_duplicates().groupby("ID").size()
    sites = frame.groupby("ID", sort=True).agg(
        Toponym=("Toponym", "first"), lon=("Long_Cord", "first"), lat=("Lat_Cord", "first"),
        LocQual=("LocQual", lambda s: "|".join(sorted(set(s)))), occupation_rows=("Period", "size"),
        periods=("Period", "nunique"),
        quality_documented=("LocQual", lambda s: set(s).issubset(set("ABCDE"))),
        quality_conflict=("LocQual", lambda s: s.nunique() > 1),
    ).reset_index()
    sites["coordinate_conflict"] = sites.ID.map(locations).gt(1)
    sites.loc[sites.coordinate_conflict, ["lon", "lat"]] = np.nan
    return sites


def load_cyprus(assets):
    """Read the bundled imagery and align the frozen features with the image manifest."""
    directory = Path(assets) / "cyprus"
    frame = pd.read_csv(directory / "manifest.csv")
    frame["path"] = [str(directory / name) for name in frame.filename]
    frame["pixel_sha256"] = [image_pixel_digest(path) for path in frame.path]
    vectors = load_embedding_cache(frame, directory / "embeddings.npz")
    return frame, vectors


def export_results(root, validation_results, cyprus, recommendation, configuration,
                   data_audit=None, validation_audit=None, transfer_audit=None,
                   cyprus_metrics=None, cyprus_pairs=None):
    """Package the ranked submission, evidence and one plain-text recommendation."""
    root = Path(root)
    configuration = {
        **configuration,
        "seed": SEED,
        "dataset": "Enki Normalized_Tell_combined_350.zip, curated 600-image workshop sample",
        "geographic_independence_verified": False,
        "encoder": MODEL_ID if configuration["selected_model"] == "dinov2" else None,
        "encoder_revision": MODEL_REVISION if configuration["selected_model"] == "dinov2" else None,
        "preprocessing": PREPROCESS_ID if configuration["selected_model"] == "dinov2" else None,
        "image_preprocessing": IMAGE_PREPROCESS_ID,
        "common_image_size": COMMON_IMAGE_SIZE,
        "cyprus_scores_calibrated": False,
        "cyprus_image_visibility_labels_available": False,
    }
    validation_results.to_csv(root / "validation_results.csv", index=False)
    cyprus.drop(columns=["path"], errors="ignore").to_csv(root / "cyprus_review.csv", index=False)
    (root / "configuration.json").write_text(json.dumps(configuration, indent=2), encoding="utf-8")
    report = recommendation.strip() or "Recommendation not yet written."
    (root / "recommendation.md").write_text(report + "\n", encoding="utf-8")
    names = ["submission.csv", "configuration.json", "validation_results.csv",
             "cyprus_review.csv", "recommendation.md"]
    if data_audit is not None:
        (root / "data_audit.json").write_text(json.dumps(data_audit, indent=2), encoding="utf-8")
        names.append("data_audit.json")
    for name, frame in (("validation_diagnostics.csv", validation_audit),
                        ("transfer_diagnostics.csv", transfer_audit),
                        ("cyprus_catalogue_metrics.csv", cyprus_metrics),
                        ("cyprus_pairs.csv", cyprus_pairs)):
        if frame is not None:
            frame.to_csv(root / name, index=False)
            names.append(name)
    bundle = root / "team_submission.zip"
    with zipfile.ZipFile(bundle, "w", zipfile.ZIP_DEFLATED) as output:
        for name in names:
            output.write(root / name, name)
    if not recommendation.strip():
        print("Draft saved. Write your recommendation and rerun the export cell before sharing.")
    return bundle


def nearest_examples(query_vector, train_vectors, train_frame, k=4):
    similarity = train_vectors @ query_vector
    order = ranking_indices(similarity)[:k]
    return train_frame.iloc[order].reset_index(drop=True), similarity[order]


def learning_curve_rows(estimator, X, y, X_val, y_val, metric_fn, k=50):
    rows = []
    for budget in (10, 25, 50, 100):
        for seed in (0, 1, 2):
            rng = np.random.default_rng(seed)
            if any((y == label).sum() < budget for label in (0, 1)):
                continue
            indices = np.concatenate([rng.choice(np.flatnonzero(y == label), budget, replace=False)
                                      for label in (0, 1)])
            fitted = clone(estimator).fit(X[indices], y[indices])
            scores = fitted.predict_proba(X_val)[:, 1]
            rows.append(dict(labels_per_class=budget, seed=seed, **metric_fn(y_val, scores, k)))
    return pd.DataFrame(rows)


def write_submission(frame, scores, destination):
    scores = np.asarray(scores, dtype=float)
    if len(scores) != len(frame):
        raise ValueError("Exactly one score per test tile is required.")
    order = ranking_indices(scores)
    output = pd.DataFrame({"tile_id": frame.tile_id.to_numpy()[order], "score": scores[order],
                           "rank": np.arange(1, len(scores) + 1)})
    assert output.tile_id.is_unique
    output.to_csv(destination, index=False)
    return output


def score_submission(submission, truth, metric_fn, k=50):
    if not submission.tile_id.is_unique or set(submission.tile_id) != set(truth.tile_id):
        raise ValueError("Submission must contain every test tile exactly once, with no extra IDs.")
    aligned = truth[["tile_id", "label"]].merge(submission[["tile_id", "score"]],
                                               on="tile_id", how="left", validate="one_to_one")
    return metric_fn(aligned.label.to_numpy(), aligned.score.to_numpy(), k)


def make_cyprus_bundle(phases, output_dir, local_model=None):
    """Instructor refresh utility: public Sentinel-2 COG windows; no paid account required.

    Normal participant runs use the verified, embedded bundle instead of live STAC requests.
    """
    import rasterio
    from rasterio.warp import transform as transform_points
    from rasterio.windows import Window
    import planetary_computer

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    item_url = f"https://planetarycomputer.microsoft.com/api/stac/v1/collections/sentinel-2-l2a/items/{CYPRUS_ITEM}"
    response = requests.get(item_url, timeout=60)
    response.raise_for_status()
    item = response.json()
    points = phases.drop_duplicates("ID").copy()
    for c in ("Long_Cord", "Lat_Cord", "Longitude", "Latitude"):
        points[c] = pd.to_numeric(points[c], errors="coerce")
    candidates = points.loc[(points.LocQual.str.strip() == "A") &
                            points.Long_Cord.between(32.4, 32.85) &
                            points.Lat_Cord.between(34.72, 35.12)]
    candidates = candidates.sort_values("ID").sample(frac=1, random_state=SEED)
    rows = []
    known_xy = points[["Longitude", "Latitude"]].dropna().to_numpy()
    with rasterio.Env(GDAL_DISABLE_READDIR_ON_OPEN="EMPTY_DIR", GDAL_HTTP_MAX_RETRY="3",
                       GDAL_HTTP_RETRY_DELAY="2"):
        with rasterio.open(planetary_computer.sign(item["assets"]["visual"]["href"])) as rgb, \
             rasterio.open(planetary_computer.sign(item["assets"]["SCL"]["href"])) as scl:
            assert rgb.crs.to_epsg() == 32636
            for _, site in candidates.iterrows():
                if len(rows) >= 12:
                    break
                x, y = transform_points("EPSG:4326", rgb.crs, [site.Long_Cord], [site.Lat_Cord])
                pair = []
                for kind, px, py in (("catalogued_location", x[0], y[0]),
                                     ("unlabelled_background", x[0] + 2000, y[0])):
                    if kind == "unlabelled_background" and np.linalg.norm(known_xy - [px, py], axis=1).min() < 750:
                        break
                    rr, cc = rgb.index(px, py)
                    if not (50 <= rr < rgb.height - 50 and 50 <= cc < rgb.width - 50):
                        break
                    # 1 km x 1 km at 10 m/pixel; SCL has 20 m pixels.
                    image = rgb.read([1, 2, 3], window=Window(cc - 50, rr - 50, 100, 100))
                    sr, sc = scl.index(px, py)
                    quality = scl.read(1, window=Window(sc - 25, sr - 25, 50, 50))
                    clear_land = float(np.isin(quality, [4, 5, 7]).mean())
                    if clear_land < 0.9 or (image.sum(axis=0) == 0).mean() > 0.05:
                        break
                    tile_id = f"{site.ID}_{'site' if kind == 'catalogued_location' else 'background'}"
                    filename = tile_id + ".png"
                    Image.fromarray(image.transpose(1, 2, 0)).save(output_dir / filename)
                    lon, lat = transform_points(rgb.crs, "EPSG:4326", [px], [py])
                    pair.append(dict(tile_id=tile_id, filename=filename, kind=kind, paired_site=site.ID,
                                     lon=lon[0], lat=lat[0], LocQual=site.LocQual.strip(),
                                     clear_land_fraction=clear_land, gsd_m=10, footprint_m=1000,
                                     acquisition=item["properties"]["datetime"], item_id=CYPRUS_ITEM))
                if len(pair) == 2:
                    rows.extend(pair)
    manifest = pd.DataFrame(rows).sort_values("tile_id").reset_index(drop=True)
    if len(manifest) != 12:
        raise RuntimeError(f"Expected six usable pairs; found {len(manifest)} chips.")
    manifest.to_csv(output_dir / "manifest.csv", index=False)
    vectors = extract_dino([str(output_dir / f) for f in manifest.filename], local_model=local_model)
    cache_frame = manifest.assign(path=[str(output_dir / name) for name in manifest.filename])
    save_embedding_cache(cache_frame, output_dir / "embeddings.npz", vectors)
    metadata = dict(item_url=item_url, item_id=CYPRUS_ITEM, acquisition=item["properties"]["datetime"],
                    attribution="Contains modified Copernicus Sentinel data (2024), accessed through Microsoft Planetary Computer.",
                    interpretation="Landscape-context stress test, not archaeological detection ground truth.")
    (output_dir / "provenance.json").write_text(json.dumps(metadata, indent=2))
    return manifest
