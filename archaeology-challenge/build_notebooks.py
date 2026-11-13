"""Build standalone notebooks; reuse frozen assets from the checked-in participant copy."""
import argparse
import ast
import base64
import csv
import io
from pathlib import Path
import textwrap
import zipfile

import nbformat as nbf

HERE = Path(__file__).resolve().parent


def code(source, cell_id, **metadata):
    return nbf.v4.new_code_cell(
        textwrap.dedent(source).strip() + "\n", id=cell_id, metadata=metadata,
    )


def md(source, cell_id):
    return nbf.v4.new_markdown_cell(textwrap.dedent(source).strip(), id=cell_id)


def supplied(source, cell_id, title):
    """Fold provided plumbing while leaving its output visible."""
    return code(f"# @title Supplied · {title}\n" + textwrap.dedent(source).strip(), cell_id,
                tags=["supplied"], **{"jupyter": {"source_hidden": True},
                                     "colab": {"cellView": "form"}})


def embedded_assets(asset_dir=None):
    """Repack only the pinned data and current helper source, with stable ZIP metadata.

    A fresh checkout can rebuild without private preparation caches. To replace the data,
    pass a directory with the same layout as the unpacked notebook assets.
    """
    if asset_dir is None:
        notebook = nbf.read(HERE / "archaeology_participants.ipynb", as_version=4)
        payload = None
        for cell in notebook.cells:
            if cell.cell_type != "code" or "ASSET_B64 =" not in cell.source:
                continue
            for node in ast.parse(cell.source).body:
                if isinstance(node, ast.Assign) and any(
                    isinstance(target, ast.Name) and target.id == "ASSET_B64"
                    for target in node.targets
                ):
                    payload = ast.literal_eval(node.value)
        if payload is None:
            raise ValueError("No bundled assets found; supply --assets-dir with prepared data.")
        with zipfile.ZipFile(io.BytesIO(base64.b64decode(payload))) as bundle:
            files = {name: bundle.read(name) for name in bundle.namelist()}
    else:
        directory = Path(asset_dir)
        files = {path.relative_to(directory).as_posix(): path.read_bytes()
                 for path in directory.rglob("*") if path.is_file()}

    manifest = csv.DictReader(io.StringIO(files["cyprus/manifest.csv"].decode()))
    names = ["enki_embeddings.npz", "sample_manifest.json", "Cyprus_Settlements.csv",
             "cyprus/manifest.csv", "cyprus/embeddings.npz", "cyprus/provenance.json"]
    names += ["cyprus/" + row["filename"] for row in manifest]
    files["workshop_helpers.py"] = (HERE / "src/archaeology_challenge/runtime.py").read_bytes()
    names.append("workshop_helpers.py")
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", zipfile.ZIP_DEFLATED) as bundle:
        for name in sorted(names):
            info = zipfile.ZipInfo(name, date_time=(2025, 3, 1, 0, 0, 0))
            info.compress_type = zipfile.ZIP_DEFLATED
            bundle.writestr(info, files[name])
    return base64.b64encode(buffer.getvalue()).decode()


def notebook(instructor, payload):
    role = "Instructor copy" if instructor else "Participant notebook"
    cells = [md('''
    # From tells to Cyprus
    **Participant notebook · four-hour workshop**

    Help a survey team choose which **50 images** to review first. Compare two models
    on labelled tell images, inspect their mistakes, then try Cyprus imagery.
    A *tell* is a mound formed by repeated human occupation.

    1. Run the six sections in order. Complete the `...` expressions and keep brief notes.
    2. Choose one optional lab before section 6, if time permits.
    3. Write your recommendation, then share the ZIP and saved notebook before the test reveal.

    Run cells marked **Supplied** as they are; their code is folded. Exercise code and
    settings remain visible. Use Jupyter or Colab; setup needs internet on first use
    and downloads about 73 MB of images. No GPU is needed.
    ''' if not instructor else '''
    # From tells to Cyprus
    **Instructor copy · four-hour workshop**

    Help a survey team choose which **50 images** to review first. Compare two models
    on labelled tell images, inspect their mistakes, then try Cyprus imagery.
    A *tell* is a mound formed by repeated human occupation.

    1. Run the six sections in order. Complete the `...` expressions and keep brief notes.
    2. Choose one optional lab before section 6, if time permits.
    3. Write your recommendation, then share the ZIP and saved notebook before the test reveal.

    Run cells marked **Supplied** as they are; their code is folded. Exercise code and
    settings remain visible. Use Jupyter or Colab; setup needs internet on first use
    and downloads about 73 MB of images. No GPU is needed.
    ''', "introduction")]
    if instructor:
        cells += [md('''
    This copy includes solutions, lab outputs and the final test result. Teaching prompts,
    timing and troubleshooting are in the [facilitator guide](https://github.com/DLR-MF-DAS/hackathEOn/blob/main/archaeology-challenge/FACILITATOR_GUIDE.md).
    ''', "instructor-introduction")]

    cells += [md('''
    ## 1. Look at the images

    Run setup, then load the images. Enki's source labels are **tell (`1`)** and
    **negative (`0`)**.
    ''', "setup-introduction"), code(f'''
    # @title Supplied · setup (run once)
    import base64, importlib.util, io, os, shutil, subprocess, sys, zipfile
    from pathlib import Path

    requirements = {{"numpy": "numpy>=1.24,<3", "pandas": "pandas>=2.1,<4",
                    "matplotlib": "matplotlib>=3.8,<4", "sklearn": "scikit-learn>=1.4,<2",
                    "PIL": "pillow>=10,<14", "imagehash": "ImageHash==4.3.2",
                    "requests": "requests>=2.31,<3", "tqdm": "tqdm>=4.66,<5"}}
    missing = [package for module, package in requirements.items()
               if importlib.util.find_spec(module) is None]
    if missing and not os.environ.get("ARCHAEOLOGY_SKIP_INSTALL"):
        installer = ([sys.executable, "-m", "pip", "install", "-q"]
                     if importlib.util.find_spec("pip") else
                     [shutil.which("uv") or "uv", "pip", "install", "--python", sys.executable, "-q"])
        subprocess.check_call(installer + missing)

    ROOT = Path(os.environ.get("ARCHAEOLOGY_WORKSHOP_ROOT", "archaeology_workshop"))
    assets = ROOT / "assets"
    assets.mkdir(parents=True, exist_ok=True)
    ASSET_B64 = "{payload}"
    with zipfile.ZipFile(io.BytesIO(base64.b64decode(ASSET_B64))) as bundle:
        for name in bundle.namelist():
            if not (assets / name).resolve().is_relative_to(assets.resolve()):
                raise ValueError("Unsafe asset path")
        bundle.extractall(assets)
    del ASSET_B64
    spec = importlib.util.spec_from_file_location("workshop_helpers", assets / "workshop_helpers.py")
    workshop = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(workshop)
    print("Ready. Run the next cell to load the images.")
    ''', "setup", tags=["infrastructure"],
        **{"jupyter": {"source_hidden": True}, "colab": {"cellView": "form"}}), md('''
    We use a balanced 600-image sample from Enki's training archive. Removing similar
    images and conflicting labels leaves 593 images, split as follows:

    | Set | Images | Use |
    |---|---:|---|
    | Training | 200 | Fit the models: 100 tells and 100 negatives. |
    | Validation | 119 | Compare models and choose one. |
    | Test | 119 | Check the saved choice at the end. |
    | Unused | 155 | Keep the training exercise small. |

    All tells were originally 350 × 350 pixels; most negatives were 64 × 64.
    Both models receive 64 × 64 images to reduce this shortcut. Section 3 checks the
    remaining acquisition bias. This teaching split has no verified geographic separation.
    ''', "images-introduction"), supplied('''
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    from IPython.display import display, FileLink
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    archive = workshop.download_enki_sample(assets / "sample_manifest.json", ROOT / "enki_workshop_sample.zip")
    images = workshop.index_enki(archive, ROOT / "images")
    images, duplicate_audit, audit = workshop.deduplicate_images(images)
    splits = workshop.workshop_split(images, per_class=100)
    train, validation, test = (splits[name] for name in ("train", "validation", "test"))
    print(f"{audit['raw_images']} images → {len(images)} after filtering.")
    print(f"Training: {len(train)}; validation: {len(validation)}; test: {len(test)}.")
    size_counts = images.groupby(["label", "width", "height"]).size().rename("images").reset_index()
    audit["native_size_counts"] = size_counts.to_dict("records")
    audit["image_preprocessing"] = workshop.IMAGE_PREPROCESS_ID
    workshop.contact_sheet(train.sample(12, random_state=42), title="Training images: 1 = tell, 0 = negative")
    ''', "images", title='load and preview images'), md('''
    **Inspect:** choose one tell and one negative from the gallery. Record their IDs and
    one visible difference. Could it describe tell shape, or how the imagery was collected?
    ''', "image-question")]

    cells += [md('''
    ## 2. What makes a useful shortlist?

    - **Precision** = positives in the shortlist / images reviewed.
    - **Recall** = positives in the shortlist / all available positives.

    Practise on this invented ranking of five labels: `1` means positive, `0` negative.
    The list is already ordered by an imagined model's score.
    ''', "shortlist-introduction"), code('''
    # Invented ranking: 1 = positive, 0 = negative.
    ranked_labels = [1, 1, 0, 1, 0]
    ''', "shortlist-labels"), md('''
    For `k = 2`, the first two entries are `[1, 1]`: two positives in two reviews,
    out of three positives in the full list. Precision is `2/2 = 100%`; recall is
    `2/3 ≈ 67%`.

    Run the cell. `[:k]` selects the first `k` entries; `sum()` counts their positive labels.
    ''', "shortlist-predict"), code('''
    k = 2  # Number of list entries to include in the shortlist

    shortlist_labels = ranked_labels[:k]

    shortlist_size = len(shortlist_labels)
    positives_in_shortlist = sum(shortlist_labels)
    positives_in_full_list = sum(ranked_labels)

    precision = positives_in_shortlist / shortlist_size
    recall = positives_in_shortlist / positives_in_full_list

    print("Full supplied list:", ranked_labels)
    print(f"Shortlist (first {k} entries):", shortlist_labels)
    print(f"Precision: {positives_in_shortlist}/{shortlist_size} = {precision:.1%}")
    print(f"Recall: {positives_in_shortlist}/{positives_in_full_list} = {recall:.1%}")
    ''', "shortlist-example"), md('''
    **Try it:** predict precision and recall for `k = 4`, then change the value and rerun.
    Why do the fractions change differently? Section 3 uses the same calculation on
    real validation images at `k = 50`.
    ''', "shortlist-discussion")]

    cells += [md('''
    ## 3. Compare two models

    Which image representation puts more tells into a 50-image shortlist?

    ### 3.1. Turn images into features

    A **feature matrix** has one image per row and one numeric measurement per column.
    Labels are stored separately. We compare:

    | Representation | Features per image | Description |
    |---|---:|---|
    | `colour_texture` | 68 | Colour distributions, variation and edges: a simple baseline. |
    | `dinov2` | 384 | Visual features from a pretrained neural network. |

    DINOv2's features are already computed and bundled. Its weights stay frozen; we fit
    one logistic regression for each representation. Both use the same images and split.
    Run the next cell to prepare the matrices.
    ''', "models-introduction"), code('''
    # Compute simple measurements and load the previously computed DINOv2 vectors.
    colour_features = workshop.handcrafted_features(images.path)
    dino_features = workshop.load_embedding_cache(images, assets / "enki_embeddings.npz")

    # Reuse the same image groups for both representations.
    feature_sets = {
        "colour_texture": workshop.partition_features(images, colour_features, splits),
        "dinov2": workshop.partition_features(images, dino_features, splits),
    }

    feature_summary = []
    for name, features in feature_sets.items():
        feature_summary.append({
            "representation": name,
            "training_images": features["train"].shape[0],
            "validation_images": features["validation"].shape[0],
            "features_per_image": features["train"].shape[1],
        })
    display(pd.DataFrame(feature_summary))
    ''', "features"), md('''
    ### 3.2. Fit the classifiers

    Each pipeline learns two steps from **training data only**:

    1. `StandardScaler` centres and scales each feature using its training mean and spread.
    2. Logistic regression learns feature weights to predict the source label.

    The fitted pipeline reuses those transformations for later images. Run `.fit(...)`
    below to train both models.
    ''', "training-explanation"), code('''
    models = {}

    for name, features in feature_sets.items():
        model = make_pipeline(
            StandardScaler(),
            LogisticRegression(C=1, class_weight="balanced", max_iter=2000, random_state=42),
        )
        model.fit(features["train"], train.label)
        models[name] = model
        print(f"Fitted {name}: {len(train)} training images, "
              f"{features['train'].shape[1]} features per image.")
    ''', "train-models"), md('''
    ### 3.3. Rank the validation images

    `predict_proba(...)[:, 1]` gives each image's score for label `1`.
    Higher scores are reviewed first. Run the cell to rank validation images and preview
    each model's top four.
    ''', "validation-explanation"), code('''
    validation_scores = {}

    for name, model in models.items():
        probabilities = model.predict_proba(feature_sets[name]["validation"])
        validation_scores[name] = probabilities[:, 1]  # Column for source label 1

    # Show each model's highest-scoring images, keeping both full rankings.
    validation_rankings = {}
    for name, scores in validation_scores.items():
        ranking = validation[["tile_id", "label", "path"]].copy()
        ranking["score"] = scores
        ranking = ranking.sort_values("score", ascending=False, kind="stable").reset_index(drop=True)
        ranking.insert(0, "rank", np.arange(1, len(ranking) + 1))
        validation_rankings[name] = ranking
        preview = ranking.head(4)

        fig, axes = plt.subplots(1, 4, figsize=(12, 3.5), layout="constrained")
        for ax, row in zip(axes, preview.itertuples(index=False)):
            ax.imshow(plt.imread(row.path))
            ax.set_title(f"Rank {row.rank} · score {row.score:.6f}\\n{row.tile_id} · label {row.label}", fontsize=9)
            ax.axis("off")
        fig.suptitle(f"{name}: highest-scoring validation images")
        plt.show()

    ''', "score-validation"), md('''
    ### 3.4. How many tells does the colour/texture model find in 50 reviews?

    The `colour_texture` model has ordered the validation images by score, highest first.
    Suppose you can review **50 images**. Take the first 50 and use their source labels
    to count how many are tells. Run the cell to display the counts.
    ''', "comparison-explanation"), code('''
    shortlist = validation_rankings["colour_texture"].head(50)
    hits = int(shortlist["label"].sum())
    reviewed = len(shortlist)
    total_positives = int(validation.label.sum())

    print(f"{reviewed} images in the review list (reviewed)")
    print(f"{hits} of them have the source label 'tell' (hits)")
    print(f"{total_positives} tell-labelled images in the complete validation set (total_positives)")
    ''', "shortlist-counts"), md('''
    **Calculate precision and recall for this review list:**

    - **Precision:** of the images you would review, what fraction are tells?
      Divide `hits` by `reviewed`.
    - **Recall:** of all the tells available, what fraction did this review list find?
      Divide `hits` by `total_positives`.

    In the next code cell, replace `...` on the `precision = ...` and `recall = ...`
    lines with those divisions. Use the variable names and Python's `/` operator.
    Run the cell to see your percentages and check whether both calculations are correct.
    ''', "shortlist-exercise-instructions"), code('''
    precision = ...  # Tells found / images reviewed.
    recall = ...     # Tells found / all tells available.

    # Supplied feedback for your two answers.
    if precision is Ellipsis or recall is Ellipsis:
        print("Replace ... on both calculation lines, then rerun this cell.")
    else:
        print(f"Precision: {precision:.1%}")
        print(f"Recall: {recall:.1%}")
        if np.allclose([precision, recall], [hits / reviewed, hits / total_positives], rtol=0, atol=1e-9):
            print("Both calculations are correct.")
        else:
            print("Check the denominators: precision uses images reviewed; recall uses all tells available.")
    '''.replace("precision = ...", "precision = hits / reviewed" if instructor else "precision = ...")
       .replace("recall = ...", "recall = hits / total_positives" if instructor else "recall = ..."),
        "shortlist-metrics-exercise"), md('''
    **Now compare both models.** Run the next cell. Which model finds more tells in
    the same 50 reviews?

    | Column | Meaning |
    |---|---|
    | `hits` | Tell-labelled images found. |
    | `reviewed` | Images in the review list. |
    | `precision_at_k` | Percentage of reviewed images that are tells. |
    | `recall_at_k` | Percentage of available tells found. |
    | `AP` | Average precision: performance across the full ranking. |

    The code selects higher recall at 50 reviews; higher AP breaks a tie.
    ''', "model-comparison-instructions"), code('''
    REVIEW_BUDGET = 50
    validation_size = len(validation)
    validation_positives = int(validation.label.sum())
    print(f"Validation pool: {validation_size} images, {validation_positives} source positives.")
    print(f"Shortlist budget: {REVIEW_BUDGET} images per model.")

    rows = []
    for name, scores in validation_scores.items():
        metrics = workshop.ranking_metrics(validation.label, scores, k=REVIEW_BUDGET)
        row = {"model": name}
        row.update(metrics)
        rows.append(row)

    # Higher recall wins; higher AP breaks a recall tie.
    results = pd.DataFrame(rows).sort_values(
        ["recall_at_k", "AP", "model"], ascending=[False, False, True],
    ).reset_index(drop=True)
    selected = results.iloc[0]["model"]
    comparison_display = results.copy()
    for column in ("precision_at_k", "recall_at_k"):
        comparison_display[column] = results[column].map(lambda value: f"{value:.1%}")
    display(comparison_display.round(3))

    winner, runner_up = results.iloc[0], results.iloc[1]
    extra_hits = int(winner["hits"] - runner_up["hits"])
    print(f"Selected using validation: {selected}")
    if extra_hits > 0:
        print(f"Its shortlist contains {extra_hits} more source positives than {runner_up['model']}, "
              "using the same number of slots.")
    elif winner["AP"] > runner_up["AP"]:
        print("Both shortlists contain the same number of positives; higher AP broke the tie.")
    else:
        print("Recall and AP tie exactly; model-name order broke the tie, not stronger evidence.")
    ''', "compare-models"), md('''
    ### 3.5. Compare with random selection and the budget limit

    Randomly selecting 50 out of 119 images gives each positive a `50/119` chance of
    being included: the expected recall. Even a perfect ordering can recover at most
    50 of the 60 positives, giving a recall ceiling of `50/60`.

    The cell below calculates both references from the current counts.
    ''', "comparison-references"), code('''
    shortlist_size = min(REVIEW_BUDGET, validation_size)
    random_recall = shortlist_size / validation_size
    random_hits = random_recall * validation_positives
    maximum_hits = min(shortlist_size, validation_positives)
    recall_ceiling = maximum_hits / validation_positives

    print(f"Random selection: {random_hits:.1f} source positives on average in {shortlist_size} slots; "
          f"expected recall = {shortlist_size}/{validation_size} = {random_recall:.1%}.")
    print(f"Best possible: {maximum_hits} source positives in {shortlist_size} slots; "
          f"recall ceiling = {maximum_hits}/{validation_positives} = {recall_ceiling:.1%}.")
    print(f"Selected model: {int(winner['hits'])} source positives in the shortlist, "
          f"{validation_positives - int(winner['hits'])} outside it.")
    ''', "validation-benchmarks"), md('''
    ### 3.6. Check the image-size shortcut

    A reference using **only original width and height**, learned from training labels,
    shows how much acquisition differences predict the label.

    The table also compares model AP within each original size. Compare AP with the
    positive fraction; AP is omitted for groups containing only one label. Common resizing
    reduces the size shortcut but does not remove every acquisition difference.
    ''', "resolution-audit-explanation"), supplied('''
    validation_audit = workshop.validation_diagnostics(
        train, validation, validation_scores, k=REVIEW_BUDGET)
    size_probe = validation_audit.loc[validation_audit.model.eq("source_size_only")]
    print(f"Original-size-only reference: {size_probe.iloc[0].accuracy:.1%} validation label accuracy.")
    display(validation_audit.loc[validation_audit.model.ne("source_size_only"),
        ["model", "subset", "images", "positives", "positive_fraction", "AP"]].round(3))
    ''', "resolution-audit", title='check acquisition bias'), md('''
    **Explain your choice:** cite hits, precision and recall. How does it compare with
    random selection and the recall ceiling? Keep the selected model for the next sections.
    ''', "model-question")]

    cells += [md('''
    ## 4. Inspect the shortlist

    - **False alarm:** a source-negative image among ranks 1–50.
    - **Miss:** a source-positive image below rank 50.

    Run the cell to see up to four examples of each. Some positives must remain outside
    when there are more positives than review slots.
    ''', "mistakes-introduction"), supplied('''
    review = validation_rankings[selected].copy()
    shortlist = review.head(REVIEW_BUDGET)
    false_alarms = shortlist.query("label == 0")
    misses = review.iloc[REVIEW_BUDGET:].query("label == 1")

    print(f"{selected}: {len(shortlist)} images in the shortlist")
    print(f"Inside: {int(shortlist.label.sum())} source positives, {len(false_alarms)} source negatives.")
    print(f"Outside: {len(misses)} source positives missed at this budget.")

    for title, group in (
        ("False alarms: source negatives inside the shortlist", false_alarms),
        ("Misses: source positives outside the shortlist", misses),
    ):
        print(f"{title}: {len(group)} images")
        if group.empty:
            print("No examples in this group at this budget.")
            continue

        examples = group.head(4)
        fig, axes = plt.subplots(1, 4, figsize=(12, 3.5), layout="constrained")
        for ax in axes:
            ax.axis("off")
        for ax, row in zip(axes, examples.itertuples(index=False)):
            ax.imshow(plt.imread(row.path))
            ax.set_title(f"Rank {row.rank} · score {row.score:.6f}\\n{row.tile_id}", fontsize=9)
        fig.suptitle(f"{selected} — {title}")
        plt.show()
    ''', "mistakes", title='inspect false alarms and misses'), md('''
    **Keep one observation:** record a displayed tile's ID, rank and visible feature.
    Suggest a comparison that could test why it received that score.
    ''', "mistake-question"), md('''
    ### Try ten more reviews

    Predict how many additional positives ten slots will recover. Change
    `inspection_budget = 50` to `60` and run the cell. Report extra positives found,
    extra negatives reviewed and positives still missed.

    **Extension:** find the smallest budget that recovers every validation positive.
    Keep `REVIEW_BUDGET = 50` for the original comparison and report.
    ''', "inspection-budget-exercise"), code('''
    inspection_budget = 50  # Change to 60 to inspect ten more images.

    added = review.iloc[REVIEW_BUDGET:inspection_budget]
    remaining_misses = review.iloc[inspection_budget:].query("label == 1")
    new_false_alarms = added.query("label == 0")

    print(f"Extra images reviewed: {len(added)}")
    print(f"Extra source positives recovered: {int(added.label.sum())}")
    print(f"Extra background-labelled images: {len(new_false_alarms)}")
    print(f"Source positives still outside: {len(remaining_misses)}")
    display(added[["rank", "tile_id", "label", "score"]].round(6))
    workshop.contact_sheet(new_false_alarms, new_false_alarms.score.to_numpy(), n=4,
                           title=f"{selected}: newly included source negatives at {inspection_budget} reviews")
    ''' if not instructor else '''
    inspection_budget = 60  # Worked example: inspect ten more images.

    added = review.iloc[REVIEW_BUDGET:inspection_budget]
    remaining_misses = review.iloc[inspection_budget:].query("label == 1")
    new_false_alarms = added.query("label == 0")

    print(f"Extra images reviewed: {len(added)}")
    print(f"Extra source positives recovered: {int(added.label.sum())}")
    print(f"Extra background-labelled images: {len(new_false_alarms)}")
    print(f"Source positives still outside: {len(remaining_misses)}")
    display(added[["rank", "tile_id", "label", "score"]].round(6))
    workshop.contact_sheet(new_false_alarms, new_false_alarms.score.to_numpy(), n=4,
                           title=f"{selected}: newly included source negatives at {inspection_budget} reviews")
    ''', "inspection-budget")]

    cells += [md('''
    ## 5. Try the same model in Cyprus

    ### 5.1. Score a different landscape

    Reuse the fitted models on **12 Sentinel-2 images**: six catalogued settlement
    locations and six nearby unlabelled locations, paired 2 km apart. Each image covers
    1 km × 1 km at 10 m per pixel, acquired on 5 August 2024.

    | `kind` | Meaning |
    |---|---|
    | `catalogued_location` | Centred on a recorded settlement. |
    | `unlabelled_background` | Nearby location without a catalogue label. |

    **These are not visibility labels.** A catalogue point need not have visible remains;
    an unlabelled location is not a confirmed absence. We can measure catalogue agreement,
    but detection accuracy remains unknown. Resolution, landscape and target form also
    differ from Enki.

    Run the cell to compare score ranges and view the four highest-ranked Cyprus images.
    The scores have not been calibrated as Cyprus probabilities.
    ''', "cyprus-introduction"), code('''
    cyprus, cyprus_dino = workshop.load_cyprus(assets)
    cyprus_features = {
        "colour_texture": workshop.handcrafted_features(cyprus.path),
        "dinov2": cyprus_dino,
    }
    cyprus_scores = {}
    for name, model in models.items():
        cyprus_scores[name] = model.predict_proba(cyprus_features[name])[:, 1]
        cyprus[f"{name}_score"] = cyprus_scores[name]
    transfer_audit = workshop.transfer_summary(validation.label, validation_scores, cyprus_scores)
    display(transfer_audit.round(3))
    cyprus["score"] = cyprus_scores[selected]
    cyprus = cyprus.sort_values("score", ascending=False, kind="stable").reset_index(drop=True)
    cyprus.insert(0, "rank", range(1, len(cyprus) + 1))
    workshop.contact_sheet(cyprus, cyprus.score.to_numpy(), n=4, show_labels=False,
                           title=f"Cyprus: four highest scores from {selected}")
    ''', "cyprus"), md('''
    ### 5.2. Compare two pairs

    1. Run `pair_id = "CY5"`: catalogue location on the left, unlabelled neighbour on the right.
    2. Change it to `"CY479"` and rerun. Does the catalogue location rank first in both pairs?
    3. Record one visible difference and the evidence needed to judge either ordering as an error.
    ''', "cyprus-question"), code('''
    pair_id = "CY5"  # Then change this to "CY479" and rerun.
    pair = cyprus.loc[cyprus["paired_site"].eq(pair_id)].sort_values("kind")

    fig, axes = plt.subplots(1, 2, figsize=(8, 4), layout="constrained")
    for ax, row in zip(axes, pair.itertuples(index=False)):
        place = "Catalogued location" if row.kind == "catalogued_location" else "Unlabelled location"
        ax.imshow(plt.imread(row.path))
        ax.set_title(f"{row.tile_id}\\n{place}\\nRank {row.rank} · score {row.score:.6f}", fontsize=9)
        ax.axis("off")
    fig.suptitle(f"{selected}: pair {pair_id}")
    plt.show()
    ''', "cyprus-pair"), md('''
    **Your notes:** replace this text with two or three sentences naming the tiles,
    describing an observation and stating what evidence is missing.
    ''' if not instructor else '''
    **Worked observation:** `CY5_site` shows bright angular patches; its neighbour has
    pale winding lines through darker terrain. This describes the images, not the
    classifier's reasoning. Check the current ranks and repeat for `CY479`; suitable
    visibility labels are needed before calling either ordering a detection error.
    ''', "cyprus-pair-notes"), md('''
    ### 5.3. Calculate agreement with the Cyprus catalogue

    Use all six pairs and a **four-image shortlist**. A pair is a win if its catalogued
    image scores higher than its unlabelled neighbour, a loss if lower, and a tie if equal.
    Run the cell to obtain the counts for the selected model.
    ''', "cyprus-metrics-introduction"), code('''
    CYPRUS_BUDGET = 4
    cyprus_pairs, cyprus_counts = workshop.catalogue_agreement(cyprus, k=CYPRUS_BUDGET)
    display(cyprus_pairs.round(6))
    print(f"First {CYPRUS_BUDGET} images in the selected model's ranking:")
    display(cyprus.head(CYPRUS_BUDGET)[["rank", "tile_id", "kind", "score"]].round(6))

    pair_wins = cyprus_counts["wins"]
    pair_ties = cyprus_counts["ties"]
    pair_count = cyprus_counts["pairs"]
    cyprus_hits = cyprus_counts["catalogued_at_k"]
    cyprus_reviewed = cyprus_counts["reviewed"]
    catalogued_total = cyprus_counts["catalogued_total"]
    print(f"Pairs: {pair_wins} wins, {pair_ties} ties, {cyprus_counts['losses']} losses; "
          f"{pair_count} pairs in total.")
    print(f"Shortlist: {cyprus_hits} catalogued locations in {cyprus_reviewed} reviews; "
          f"{catalogued_total} catalogued locations in the full sample.")
    ''', "cyprus-counts"), md('''
    **Calculate how well the model ranks the catalogued Cyprus locations.** Replace
    `...` on each of the three lines below, using these counts:

    1. `pair_agreement`: how often does the catalogued image score higher than its neighbour?
       Add `pair_wins` and half of `pair_ties`, then divide by `pair_count`.
    2. `catalogue_fraction`: what fraction of the reviewed images are catalogued locations?
       Divide `cyprus_hits` by `cyprus_reviewed`.
    3. `catalogue_recovery`: what fraction of all catalogue locations did the review list find?
       Divide `cyprus_hits` by `catalogued_total`.

    Run your calculation cell, then **Supplied · check your Cyprus calculations**.
    ''', "cyprus-calculation-instructions"), code('''
    pair_agreement = ...  # Include half credit for ties.
    catalogue_fraction = ...  # Fraction of reviewed images that are catalogued.
    catalogue_recovery = ...  # Fraction of all catalogued locations recovered.

    print("Pair agreement:", pair_agreement)
    print("Catalogue fraction in shortlist:", catalogue_fraction)
    print("Catalogue recovery:", catalogue_recovery)
    ''' if not instructor else '''
    pair_agreement = (pair_wins + 0.5 * pair_ties) / pair_count
    catalogue_fraction = cyprus_hits / cyprus_reviewed
    catalogue_recovery = cyprus_hits / catalogued_total

    print("Pair agreement:", pair_agreement)
    print("Catalogue fraction in shortlist:", catalogue_fraction)
    print("Catalogue recovery:", catalogue_recovery)
    ''', "cyprus-calculation"), md('''
    Compare with the printed random references. If scores tie across a shortlist cutoff,
    the check reports the possible range of catalogue counts under different tie orderings.
    ''', "cyprus-check-explanation"), supplied('''
    cyprus_metric_rows, cyprus_pair_rows = [], []
    for name in models:
        pairs_for_model, metrics = workshop.catalogue_agreement(
            cyprus, score_column=f"{name}_score", k=CYPRUS_BUDGET)
        cyprus_metric_rows.append({"model": name, **metrics})
        cyprus_pair_rows.append(pairs_for_model.assign(model=name))
    cyprus_metrics = pd.DataFrame(cyprus_metric_rows)
    cyprus_pair_results = pd.concat(cyprus_pair_rows, ignore_index=True)
    cyprus_table = pd.DataFrame({"Model": cyprus_metrics.model,
        "Pairs: win/tie/loss": [f"{row.wins}/{row.ties}/{row.losses}" for row in cyprus_metrics.itertuples()]})
    for column, title in (("pair_agreement", "Pair agreement"),
                          ("catalogue_fraction_at_k", "Catalogue fraction"),
                          ("catalogue_recovery_at_k", "Catalogue recovery")):
        cyprus_table[title] = cyprus_metrics[column].map(lambda value: f"{value:.1%}")
    display(cyprus_table)

    cyprus_reference = cyprus_metrics.set_index("model").loc[selected]
    print(f"{selected}: pair agreement = ({pair_wins} + 0.5 × {pair_ties})/{pair_count} "
          f"= {cyprus_reference.pair_agreement:.1%}.")
    print(f"Catalogue fraction = {cyprus_hits}/{cyprus_reviewed} "
          f"= {cyprus_reference.catalogue_fraction_at_k:.1%}; "
          f"catalogue recovery = {cyprus_hits}/{catalogued_total} "
          f"= {cyprus_reference.catalogue_recovery_at_k:.1%}.")
    print(f"Random references: pair agreement {cyprus_reference.random_pair_agreement:.1%}; "
          f"catalogue fraction {cyprus_reference.random_catalogue_fraction:.1%}; "
          f"catalogue recovery {cyprus_reference.random_catalogue_recovery:.1%}.")
    for row in cyprus_metrics.itertuples(index=False):
        if row.cutoff_tie:
            print(f"{row.model}: a score tie crosses the shortlist cutoff; "
                  f"{row.min_catalogued_at_k}–{row.max_catalogued_at_k} catalogue locations "
                  f"could occupy its {row.reviewed} slots under different tie orderings.")
    answers = (pair_agreement, catalogue_fraction, catalogue_recovery)
    expected = cyprus_reference[["pair_agreement", "catalogue_fraction_at_k", "catalogue_recovery_at_k"]]
    if any(answer is Ellipsis for answer in answers):
        print("Complete the three expressions, then rerun your calculation and this check.")
    elif np.allclose(answers, expected.to_numpy(dtype=float), rtol=0, atol=1e-9):
        print("All three calculations match the reference.")
    else:
        print("Recheck the counts, denominators and half credit for ties against the reference above.")
    print("Cyprus detection accuracy: unknown; independent image-visibility labels are missing.")
    ''', "cyprus-metrics-check", title='check your Cyprus calculations'), md('''
    **Conclude:** compare your percentages with random ordering. Keep one calculation for
    your recommendation and explain why it cannot establish detection accuracy. These six
    selected pairs, with some overlapping image footprints, are a small sample.
    ''' if not instructor else '''
    **Worked interpretation:** compare wins and ties as well as agreement percentages.
    Better catalogue recovery supports a statement about these sampled locations. The
    six pairs and missing visibility labels cannot establish detection accuracy in Cyprus.
    ''', "cyprus-metrics-discussion")]

    cells += [md('''
    ## Optional lab — choose one before section 6

    Allow about 30 minutes for **one lab**, or continue to section 6.

    | Lab | Task |
    |---|---|
    | A. Feature projection | Complete PCA and compare the feature spaces. |
    | B. Training data | Fit smaller models and inspect changed mistakes. |
    | C. Cyprus image search | Retrieve similar Enki images and inspect the matches. |

    Complete the chosen function, edit its short settings cell, then run the supplied
    plotting cell. The main model choice and 50-review budget stay fixed.
    ''', "optional-experiments"), md('''
    ### Lab A. See the training features in 2D or 3D

    **Question:** do the label groups occupy different parts of feature space?
    PCA creates axes that retain as much feature variation as possible. It uses no labels;
    we add those as plot colours afterwards.

    Complete three expressions:

    1. Select `feature_sets[name]["train"]`.
    2. Standardise it with `scaler.fit_transform(...)`.
    3. Project it with `pca.fit_transform(...)`.

    Run the function cell, set `RUN_PROJECTION = True` below, then run the supplied plot.
    Start with two components. Each dot is a training image; the reported percentage is
    feature variance retained, not classification accuracy.
    ''', "projection-instructions"), code('''
    from sklearn.decomposition import PCA


    def project_training_features(name, n_components=2):
        # TODO 1: select this representation's training feature matrix.
        X = ...

        scaler = StandardScaler()
        # TODO 2: fit the scaler on X and return the scaled training rows.
        X_scaled = ...

        pca = PCA(n_components=n_components, svd_solver="full")
        # TODO 3: fit PCA on X_scaled and return the projected coordinates.
        coordinates = ...

        return coordinates, pca.explained_variance_ratio_
    ''' if not instructor else '''
    from sklearn.decomposition import PCA


    def project_training_features(name, n_components=2):
        # Step 1: select this representation's training feature matrix.
        X = feature_sets[name]["train"]

        scaler = StandardScaler()
        # Step 2: fit the scaler on X and return the scaled training rows.
        X_scaled = scaler.fit_transform(X)

        pca = PCA(n_components=n_components, svd_solver="full")
        # Step 3: fit PCA on X_scaled and return the projected coordinates.
        coordinates = pca.fit_transform(X_scaled)

        return coordinates, pca.explained_variance_ratio_
    ''', "projection-exercise"), code(f'''
    RUN_PROJECTION = {instructor}  # Set True after completing and running the function above.
    N_COMPONENTS = 2         # Start with 2; change to 3 for the extension.
    ELEVATION, AZIMUTH = 20, -60  # Camera angles for the 3D view, in degrees.
    ''', "projection-settings"), supplied(f'''
    if not RUN_PROJECTION:
        print("Lab A skipped. Complete the function, then enable RUN_PROJECTION in its settings cell.")
    else:
        if N_COMPONENTS not in (2, 3):
            raise ValueError("Choose N_COMPONENTS = 2 or 3 for this plotting exercise.")

        fig = plt.figure(figsize=(13, 6))
        fig.subplots_adjust(left=0.07, right=0.94, bottom=0.16, top=0.84, wspace=0.28)
        labels = train.label.to_numpy()  # Same row order as the training feature matrices.
        class_styles = [
            (0, "Background (source label 0)", "#2874A6", "o"),
            (1, "Tell (source label 1)", "#D55E00", "^"),
        ]

        for panel, name in enumerate(("colour_texture", "dinov2"), start=1):
            coordinates, variance_ratio = project_training_features(name, N_COMPONENTS)
            ax = fig.add_subplot(1, 2, panel, projection="3d" if N_COMPONENTS == 3 else None)

            for label, description, colour, marker in class_styles:
                points = coordinates[labels == label]
                style = dict(label=description, color=colour, marker=marker,
                             s=32, alpha=0.75, edgecolors="white", linewidths=0.4)
                if N_COMPONENTS == 3:
                    ax.scatter(points[:, 0], points[:, 1], points[:, 2], depthshade=False, **style)
                else:
                    ax.scatter(points[:, 0], points[:, 1], **style)

            ax.set_xlabel(f"PC1 ({{variance_ratio[0]:.1%}} of variance)")
            ax.set_ylabel(f"PC2 ({{variance_ratio[1]:.1%}} of variance)")
            if N_COMPONENTS == 3:
                ax.set_zlabel(f"PC3 ({{variance_ratio[2]:.1%}} of variance)")
                ax.view_init(elev=ELEVATION, azim=AZIMUTH)
            ax.set_title(f"{{name}}: {{N_COMPONENTS}} PCs retain {{variance_ratio.sum():.1%}} of variance")
            ax.legend(loc="upper right", fontsize=8)
            print(f"{{name}}: coordinate shape {{coordinates.shape}}; "
                  f"variance retained {{variance_ratio.sum():.1%}}.")

        fig.suptitle(f"PCA of {{len(train)}} training images — source labels added for display")
        plt.show()
    ''', "projection-plot", title='draw the PCA plots'), md('''
    **Interpret:** describe overlap or separation and record the retained variance.
    Could images overlapping in two dimensions be separable using all features?
    Each panel has its own axes; training separation alone does not measure performance.

    **Extension:** set `N_COMPONENTS = 3`. Change `AZIMUTH` from `-60` to `30` to inspect
    another angle. Does the third component expose additional structure?
    ''', "projection-discussion"), md('''
    ### Lab B. Which mistakes does more training data fix?

    Fit on **20 training images** and compare with the existing 200-image model.
    Both rank the same validation images for **50 reviews**.

    Complete the three selections below. `rows` contains training row positions:

    - Training features: `feature_sets[name]["train"]` indexed by `rows`.
    - Matching labels: `train.label.to_numpy()` indexed by `rows`.
    - Prediction inputs: `feature_sets[name]["validation"]`.

    `clone` creates a fresh pipeline, so both scaling and classification learn from the
    smaller set. Run your function, then enable the experiment in its settings cell.
    Predict whether the larger model will retain every previously found tell.
    ''', "label-experiment-instructions"), code('''
    from sklearn.base import clone


    def small_training_scores(name, rows):
        X_small = ...       # Training features at rows.
        y_small = ...       # Training labels at the same rows.
        X_validation = ...  # Validation features, used only after fitting.
        if any(value is Ellipsis for value in (X_small, y_small, X_validation)):
            return None

        lab_model = clone(models[name])
        lab_model.fit(X_small, y_small)
        return lab_model.predict_proba(X_validation)[:, 1]
    '''.replace("X_small = ...", 'X_small = feature_sets[name]["train"][rows]' if instructor else "X_small = ...")
       .replace("y_small = ...", "y_small = train.label.to_numpy()[rows]" if instructor else "y_small = ...")
       .replace("X_validation = ...", 'X_validation = feature_sets[name]["validation"]' if instructor else "X_validation = ..."),
        "label-experiment-function"), code(f'''
    RUN_LABEL_EXPERIMENT = {instructor}  # Enable after completing the function above.
    B_MODEL = "dinov2"
    B_TRAINING_IMAGES = 20  # Total: half tells, half negatives. Then try 50.
    B_SEED = 0             # Then try 1 with 20 images to change which examples are used.
    ''', "label-experiment-settings"), supplied('''
    if not RUN_LABEL_EXPERIMENT:
        print("Lab B skipped. Complete its function and enable RUN_LABEL_EXPERIMENT to try it.")
    else:
        if B_MODEL not in models or B_TRAINING_IMAGES not in (20, 50, 100):
            raise ValueError("Choose dinov2 or colour_texture, and 20, 50 or 100 training images.")
        b_rng = np.random.default_rng(B_SEED)
        b_labels = train.label.to_numpy()
        b_rows = np.concatenate([
            b_rng.permutation(np.flatnonzero(b_labels == label))[:B_TRAINING_IMAGES // 2]
            for label in (0, 1)
        ])
        b_scores = small_training_scores(B_MODEL, b_rows)
        if b_scores is None:
            print("Complete all three expressions and rerun the function cell first.")
        else:
            b_full_scores = validation_scores[B_MODEL]
            b_results = pd.DataFrame([
                workshop.ranking_metrics(validation.label, scores, REVIEW_BUDGET)
                for scores in (b_scores, b_full_scores)
            ], index=[f"{B_TRAINING_IMAGES} training images", f"{len(train)} training images"])
            b_table = b_results[["hits", "reviewed"]].rename(columns={
                "hits": "Tells found", "reviewed": "Images reviewed"})
            b_table["Precision"] = b_results.precision_at_k.map(lambda value: f"{value:.1%}")
            b_table["Recall"] = b_results.recall_at_k.map(lambda value: f"{value:.1%}")
            print(f"{B_MODEL}; seed {B_SEED}; {int(validation.label.sum())} available validation tells.")
            display(b_table)

            b_changes = validation[["tile_id", "path", "label"]].copy()
            for column, scores in (("small_rank", b_scores), ("full_rank", b_full_scores)):
                ranks = np.empty(len(scores), dtype=int)
                ranks[workshop.ranking_indices(scores)] = np.arange(1, len(scores) + 1)
                b_changes[column] = ranks
            b_gained = b_changes.loc[b_changes.label.eq(1) &
                b_changes.small_rank.gt(REVIEW_BUDGET) & b_changes.full_rank.le(REVIEW_BUDGET)]
            b_lost = b_changes.loc[b_changes.label.eq(1) &
                b_changes.small_rank.le(REVIEW_BUDGET) & b_changes.full_rank.gt(REVIEW_BUDGET)]
            print(f"With {len(train)} training images: {len(b_gained)} previously missed tells enter, "
                  f"{len(b_lost)} previously found tells leave; net change {len(b_gained) - len(b_lost):+d}.")
            for title, changed in (("Newly found by the larger model", b_gained),
                                   ("Now missed by the larger model", b_lost)):
                print(f"{title}: {len(changed)} source-positive images; showing up to four.")
                if changed.empty:
                    continue
                fig, axes = plt.subplots(1, 4, figsize=(12, 3.5), layout="constrained")
                for ax in axes:
                    ax.axis("off")
                for ax, row in zip(axes, changed.sort_values("full_rank").head(4).itertuples(index=False)):
                    ax.imshow(plt.imread(row.path))
                    ax.set_title(f"{row.tile_id}\\nRank {row.small_rank} → {row.full_rank}", fontsize=9)
                fig.suptitle(f"{title} · {B_TRAINING_IMAGES} → {len(train)} training images")
                plt.show()
    ''', "label-experiment-run", title='compare training sets and changed images'), md('''
    **Investigate:** pick a tell entering or leaving the shortlist. Record its ID, two
    ranks and a visible feature. Does more training data improve every image or only
    the total? Use the entering/leaving counts.

    **Rerun:** try 50 training images with seed 0, then 20 with seed 1. The seed changes
    which examples are selected. Keep hit counts for all three runs and a short conclusion
    that cites your tile. Does the conclusion hold across runs?

    **Extension:** try `B_MODEL = "colour_texture"`. These experiments concern this
    validation sample; new regions need their own evaluation.
    ''', "label-experiment-discussion")]
    if instructor:
        cells += [supplied('''
    if RUN_LABEL_EXPERIMENT and b_scores is not None:
        b_worked_rows = []
        b_full_set = set(workshop.ranking_indices(validation_scores[B_MODEL])[:REVIEW_BUDGET])
        for size, seed in ((20, 0), (50, 0), (20, 1)):
            rng = np.random.default_rng(seed)
            rows = np.concatenate([rng.permutation(np.flatnonzero(train.label.to_numpy() == label))[:size // 2]
                                   for label in (0, 1)])
            scores = small_training_scores(B_MODEL, rows)
            small_set = set(workshop.ranking_indices(scores)[:REVIEW_BUDGET])
            gained = sum(int(validation.iloc[i].label) for i in b_full_set - small_set)
            lost = sum(int(validation.iloc[i].label) for i in small_set - b_full_set)
            b_worked_rows.append({"Training images": size, "Seed": seed,
                "Small-model hits": workshop.ranking_metrics(validation.label, scores, REVIEW_BUDGET)["hits"],
                "Full-model hits": int(b_results.iloc[1].hits), "Tells entering": gained, "Tells leaving": lost})
        display(pd.DataFrame(b_worked_rows))
        print("Compare net gains with the entering and leaving images. Equal hit counts need not mean "
              "the same shortlist; more training data need not improve every image's rank.")
    ''', "label-experiment-answer", title='worked comparison of the three runs')]

    cells += [md('''
    ### Lab C. What does Cyprus resemble in the training set?

    Retrieve the **four closest Enki training images** for each Cyprus image, then compare
    their labels with its classifier score.

    Complete two expressions:

    1. Compute `training_vectors @ query_vector`: one dot product per training image.
    2. Apply `np.argsort` to the negative similarities, then take `[:k]`.

    The cached vectors have length one, so the dot product is cosine similarity. Larger
    values mean closer feature directions. Labels do not choose the matches.
    Run your function, enable the supplied search and start with pair `CY754`.
    ''', "cyprus-neighbour-instructions"), code('''
    def nearest_training_indices(query_vector, training_vectors, k=4):
        similarities = ...  # One dot product per training image.
        if similarities is Ellipsis:
            return None
        indices = ...      # Indices of the k largest similarities.
        if indices is Ellipsis:
            return None
        return indices, similarities[indices]
    '''.replace("similarities = ...", "similarities = training_vectors @ query_vector" if instructor else "similarities = ...")
       .replace("indices = ...", 'indices = np.argsort(-similarities, kind="stable")[:k]' if instructor else "indices = ..."),
        "cyprus-neighbour-function"), code(f'''
    RUN_CYPRUS_NEIGHBOURS = {instructor}  # Enable after completing the function above.
    C_PAIR_IDS = {["CY754", "CY479"] if instructor else ["CY754"]}  # Pair IDs to investigate.
    C_NEIGHBOURS = 4
    ''', "cyprus-neighbour-settings"), supplied('''
    if not RUN_CYPRUS_NEIGHBOURS:
        print("Lab C skipped. Complete its search function and enable RUN_CYPRUS_NEIGHBOURS to try it.")
    else:
        if not isinstance(C_NEIGHBOURS, int) or not 1 <= C_NEIGHBOURS <= 8:
            raise ValueError("Choose between one and eight neighbours.")
        # Load IDs and vectors together: the section 5 table has since been sorted by score.
        c_images, c_vectors = workshop.load_cyprus(assets)
        c_positions = np.flatnonzero(c_images.paired_site.isin(C_PAIR_IDS))
        if not C_PAIR_IDS or len(c_positions) != 2 * len(C_PAIR_IDS):
            raise ValueError(f"Choose a pair from {sorted(c_images.paired_site.unique())}.")
        c_summary_rows = []
        for position in c_positions:
            query = c_images.iloc[position]
            matches = nearest_training_indices(c_vectors[position], feature_sets["dinov2"]["train"], C_NEIGHBOURS)
            if matches is None:
                print("Complete both expressions and rerun the function cell first.")
                break
            indices, similarities = matches
            neighbours = train.iloc[indices]
            query_score = models["dinov2"].predict_proba(c_vectors[position:position + 1])[0, 1]
            c_summary_rows.append({"Cyprus tile": query.tile_id, "DINOv2 classifier score": query_score,
                "Closest similarity": float(similarities[0]),
                "Tell-labelled neighbours": int(neighbours.label.sum()), "Neighbours": len(neighbours)})
            fig, axes = plt.subplots(1, len(neighbours) + 1, figsize=(3 * (len(neighbours) + 1), 3.5),
                                     layout="constrained")
            axes[0].imshow(plt.imread(query.path))
            axes[0].set_title(f"Cyprus: {query.tile_id}\\nClassifier score {query_score:.3f}", fontsize=9)
            for ax, row, similarity in zip(axes[1:], neighbours.itertuples(index=False), similarities):
                ax.imshow(plt.imread(row.path))
                label = "tell" if row.label == 1 else "negative"
                ax.set_title(f"{row.tile_id} · {label}\\nSimilarity {similarity:.3f}", fontsize=9)
            for ax in axes:
                ax.axis("off")
            fig.suptitle("Cyprus query → nearest Enki training images (source labels shown)")
            plt.show()
        if c_summary_rows:
            c_neighbour_summary = pd.DataFrame(c_summary_rows)
            display(c_neighbour_summary.round(3))
            print("Neighbour labels describe Enki images. They are not visibility labels for the Cyprus query.")
    ''', "cyprus-neighbour-run", title='show the Cyprus image matches'), md('''
    **Inspect:** name a Cyprus query and an Enki match. Describe a visible resemblance
    and difference. Do the neighbours' labels agree with a high or low classifier score?
    Logistic regression uses learned weights on scaled features; this search compares
    vector directions, so agreement is not guaranteed.

    **Rerun:** change `C_PAIR_IDS` to `["CY479"]`. Keep a short comparison of the two pairs
    with one named image match. What independent evidence would establish whether a Cyprus
    tile contains visible archaeology? Its neighbours' Enki labels cannot supply that answer.
    ''' if not instructor else '''
    **Worked interpretation:** `CY754_site` scores highly with one tell-labelled neighbour;
    its nearby query scores low with four. Both `CY479` queries have one tell-labelled
    neighbour despite different scores. The methods weight features differently.

    Ask for a named image resemblance and difference. Similarity neither explains the
    classifier's score nor supplies Cyprus visibility labels.
    ''', "cyprus-neighbour-discussion")]


    cells += [md('''
    ## 6. Recommend, save and check

    ### 6.1. Write a five-sentence recommendation

    Run the evidence summary, then write your paragraph below.
    ''', "report-introduction"), supplied('''
    report_validation = results.loc[results.model.eq(selected)].iloc[0]
    report_cyprus = cyprus_metrics.loc[cyprus_metrics.model.eq(selected)].iloc[0]
    print(f"Selected model: {selected}")
    print(f"Enki validation: {int(report_validation.hits)} of {int(validation.label.sum())} "
          f"source positives in {int(report_validation.reviewed)} reviews; "
          f"precision {report_validation.precision_at_k:.1%}, recall {report_validation.recall_at_k:.1%}.")
    print(f"Cyprus: {int(report_cyprus.catalogued_at_k)} of {int(report_cyprus.catalogued_total)} "
          f"catalogue locations in {int(report_cyprus.reviewed)} reviews; "
          f"catalogue recovery {report_cyprus.catalogue_recovery_at_k:.1%}. "
          "Detection accuracy remains unknown.")
    if report_cyprus.cutoff_tie:
        print(f"Cyprus tie warning: recovery can range from "
              f"{report_cyprus.min_catalogued_at_k / report_cyprus.catalogued_total:.1%} to "
              f"{report_cyprus.max_catalogued_at_k / report_cyprus.catalogued_total:.1%} "
              "depending on how tied scores are ordered.")
    ''', "report-evidence", title='collect evidence for the recommendation'), md('''
    Write **five sentences** between the triple quotes:

    1. **Decide:** name the model, proposed use and scope. Further evaluation or no deployment are valid choices.
    2. **Support:** cite Enki hits, reviews, available positives, precision and recall.
    3. **Observe:** name one tile and describe a visible feature.
    4. **Qualify:** report Cyprus catalogue recovery and its detection-accuracy limitation.
    5. **Act:** identify the next local imagery, expert labels and independent assessment needed.

    Reuse your notes. Read the paragraph to a partner, or reread it yourself, and revise
    any claim that goes beyond the evidence.
    ''', "report-prompts"), code('''
    # Write five sentences using the evidence and prompts above.
    recommendation = """

    """.strip()
    print(recommendation or "Draft: write your recommendation above, then rerun this cell.")
    ''' if not instructor else '''
    recommendation = (
        f"Evaluate {selected} further as an aid to expert review of imagery similar to Enki. "
        f"It retrieves {int(report_validation.hits)} of {int(validation.label.sum())} validation "
        f"source positives in {int(report_validation.reviewed)} reviews "
        f"(precision {report_validation.precision_at_k:.1%}, recall {report_validation.recall_at_k:.1%}). "
        "CY5_site shows bright angular patches, although this does not explain the model's score. "
        f"Cyprus catalogue recovery is {int(report_cyprus.catalogued_at_k)}/{int(report_cyprus.catalogued_total)} "
        f"({report_cyprus.catalogue_recovery_at_k:.1%}) in {int(report_cyprus.reviewed)} reviews; "
        "detection accuracy remains unknown because visibility labels are missing. "
        "Before using this approach in Cyprus, evaluate suitable local imagery with expert "
        "visibility labels from locations kept separate from model development."
    )
    print(recommendation)
    ''', "recommendation"), md('''
    ### 6.2. Save and share two files

    Run the export cell. It saves your paragraph, rankings, settings and audit tables;
    test labels are still unused. Then **download `team_submission.zip` and save your
    notebook (`.ipynb`)**. Share both before the reveal.

    A blank paragraph produces a draft. Finish it and rerun export before sharing;
    rerun export after any further edits too.
    ''', "export-introduction"), supplied('''
    test_scores = models[selected].predict_proba(feature_sets[selected]["test"])[:, 1]
    submission = workshop.write_submission(test, test_scores, ROOT / "submission.csv")
    bundle = workshop.export_results(ROOT, results, cyprus, recommendation, configuration={
        "selected_model": selected, "review_budget": REVIEW_BUDGET,
        "cyprus_review_budget": CYPRUS_BUDGET,
        "cyprus_metrics_target": "catalogue agreement; not image-visibility accuracy",
        "C": models[selected].named_steps["logisticregression"].C,
        "class_weight": "balanced", "scaler": "StandardScaler",
        "training_tiles": len(train), "test_tiles": len(test),
        "sample_manifest_sha256": workshop.file_digest(assets / "sample_manifest.json"),
    }, data_audit=audit, validation_audit=validation_audit, transfer_audit=transfer_audit,
       cyprus_metrics=cyprus_metrics, cyprus_pairs=cyprus_pair_results)
    # Keep the comparison tied to the choice and budget used for this export.
    submitted_model = selected
    submitted_budget = REVIEW_BUDGET
    submitted_validation = results.loc[results.model.eq(selected)].iloc[0].to_dict()
    submitted_validation["positives"] = int(validation.label.sum())
    display(FileLink(str(bundle)))
    print("Download the ZIP using this link or Colab's Files panel; also save your notebook.")
    ''', "export", title='save the submission ZIP'), md('''
    ### 6.3. Check the saved choice on Enki test images

    After sharing, set `REVEAL_TEST = True`, run the settings cell, then run the
    supplied comparison below. Compare validation and
    test results at the same review budget, then complete the one-sentence debrief.
    Keep your original recommendation. This check uses the same Enki source; it does
    not test a new region. Stop model selection after this reveal.
    ''' if not instructor else '''
    ### 6.3. Check the saved choice on Enki test images

    Reveal after participants share their files. This copy enables the test cell;
    the participant copy starts with `REVEAL_TEST = False`. Compare validation and
    test at the saved budget, then discuss the debrief. Keep the original recommendation
    and finish model selection; Cyprus detection accuracy remains unmeasured.
    ''', "reveal-introduction"), code(f'''
    REVEAL_TEST = {instructor}  # Enable after sharing the ZIP and notebook.
    ''', "reveal-settings"), supplied('''
    if REVEAL_TEST:
        final_result = workshop.score_submission(submission, test, workshop.ranking_metrics, submitted_budget)
        final_result["positives"] = int(test.label.sum())
        comparison = pd.DataFrame([submitted_validation, final_result], index=["Validation", "Test"])
        print(f"Exported model: {submitted_model}; review budget: {submitted_budget}")
        comparison_table = comparison[["hits", "reviewed", "positives"]].rename(columns={
            "hits": "Tells found", "reviewed": "Images reviewed", "positives": "Available tells"})
        comparison_table["Precision"] = comparison.precision_at_k.map(lambda value: f"{value:.1%}")
        comparison_table["Recall"] = comparison.recall_at_k.map(lambda value: f"{value:.1%}")
        display(comparison_table)
        print(f"Random selection on test: expected recall {min(submitted_budget, len(test)) / len(test):.1%}.")
    else:
        print("Test labels remain unused. Finish and share your recommendation before the reveal.")
    ''', "reveal", title='compare validation and test'), md('''
    **Debrief — complete one sentence here:** the test [supports / weakens] my
    recommendation because ..., while ... still needs to be tested.

    Save the notebook again to keep this reflection alongside your original report.
    ''' if not instructor else '''
    **Worked debrief for the default run:** the test supports further evaluation for
    expert review of Enki-like imagery because precision and recall remain close to
    validation, while performance at independent Cyprus locations remains untested.

    Ask participants to support their sentence with the comparison above, and save
    their reflection in the notebook alongside the original recommendation.
    ''', "debrief-notes")]

    cells += [md('''
    ### Sources

    - **Enki:** [Rosati (2025)](https://zenodo.org/records/14950565). CC BY-SA 4.0 with separate Google/Bing imagery conditions; retain the source notices.
    - **Cyprus catalogue:** [Crawford & Vella (2022)](https://doi.org/10.5334/joad.96), CC BY 4.0.
    - **Features:** [DINOv2](https://github.com/facebookresearch/dinov2).
    - **Cyprus imagery:** modified Copernicus Sentinel data (2024), via [Microsoft Planetary Computer](https://planetarycomputer.microsoft.com/dataset/sentinel-2-l2a).

    Versions, acquisition details and troubleshooting are in the
    [workshop README](https://github.com/DLR-MF-DAS/hackathEOn/blob/main/archaeology-challenge/README.md).
    ''', "credits")]
    nb = nbf.v4.new_notebook(cells=cells, metadata={
        "kernelspec": {"display_name": "Python 3 (ipykernel)", "language": "python", "name": "python3"},
        "language_info": {"name": "python"},
        "colab": {"name": "archaeology_instructor.ipynb" if instructor else "archaeology_participants.ipynb",
                  "provenance": [], "toc_visible": True},
    })
    return nb


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--assets-dir", type=Path, help="Replace frozen assets from an unpacked bundle directory")
    args = parser.parse_args()
    payload = embedded_assets(args.assets_dir)
    for instructor, filename in ((False, "archaeology_participants.ipynb"),
                                 (True, "archaeology_instructor.ipynb")):
        nb = notebook(instructor, payload)
        path = HERE / filename
        if path.exists():
            previous = nbf.read(path, as_version=4)
            for key in ("kernelspec", "language_info"):
                if key in previous.metadata:
                    nb.metadata[key] = previous.metadata[key]
        nbf.validate(nb)
        nbf.write(nb, path)
        print(filename, len(nb.cells), "cells")


if __name__ == "__main__":
    main()
