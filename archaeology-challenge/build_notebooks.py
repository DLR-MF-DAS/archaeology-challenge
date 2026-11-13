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
    cells = [md(f'''
    # From tells to Cyprus
    **{role} · four-hour workshop**

    A survey team has more images than it can inspect. Can a model help it choose
    **50 images to look at first**? You will teach two models using labelled examples,
    count how many tells their lists contain, and see what happens on images from Cyprus.
    A **tell** is a mound formed by repeated human occupation.

    You need basic Python: variables, lists and division. No previous machine learning
    is assumed. By the end, you should be able to explain which model made the more
    useful list and whether that result justifies using it in Cyprus.

    **How to work through this notebook**

    1. Run cells in order with **Shift+Enter**. Wait for a cell to finish before continuing.
       Stop at the questions; do not use Run all for your first pass.
    2. Run code as given unless a task asks you to change it. Replace `...` only in the
       two calculation exercises. **Supplied** cells contain working code and may appear folded.
    3. Write in the **Your notes** cells: double-click the text to edit, then Shift+Enter
       to display it again. Keep these observations for your final recommendation.
    4. After section 5, choose one optional lab or go straight to section 6. Save your
       recommendation before looking at the final test result.

    Use Jupyter or Colab. First setup needs internet and downloads about 73 MB of images;
    no GPU is needed. If you restart the kernel/runtime, rerun from setup to restore
    the variables. Save your notebook regularly.
    ''', "introduction")]
    if instructor:
        cells += [md('''
    This copy includes solutions, lab outputs and the final test result. Teaching prompts,
    timing and troubleshooting are in the [facilitator guide](https://github.com/DLR-MF-DAS/hackathEOn/blob/main/archaeology-challenge/FACILITATOR_GUIDE.md).
    ''', "instructor-introduction")]

    cells += [md('''
    ## 1. Look at the images

    A **label** is an answer supplied with an image. In the Enki image collection,
    `1` means **tell** and `0` means **negative** (not labelled as a tell).
    We use these supplied answers to teach and check the models; you do not need to
    identify archaeology yourself to complete the calculations.

    Run setup, then the image-loading cell below the table.
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
    We start with 600 images, half labelled tell and half negative. Removing repeated
    or very similar images leaves 593. We give different images different jobs:

    | Set | Images | Use |
    |---|---:|---|
    | Training | 200 | Examples the models learn from: 100 tells and 100 negatives. |
    | Validation | 119 | New examples used to compare the two models. |
    | Test | 119 | Examples saved for one final check after choosing a model. |
    | Unused | 155 | Left out to keep this exercise small. |

    Checking only the examples used for learning would be like marking an exam that
    the student has already practised. The separate sets let us check new examples.
    The final test stays closed while we choose between the models.
    Our deliberately balanced sample does not mean half of real survey images contain tells.
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
    **Inspect:** the gallery captions give each image's ID and supplied label.
    Choose one image of each label. Describe a visible difference, such as shape,
    vegetation, colour or sharpness. You are describing the picture, not proving
    that its label is correct.

    **Your notes — replace the prompts below:**

    - Tell image ID and one visible feature:
    - Negative image ID and one visible feature:
    ''' if not instructor else '''
    **Worked observation:** tell-labelled `3136089f51a53066` has a rounded boundary
    within straighter field lines. Negative `18b968d5f28bcd21` shows faint field lines
    in a visibly blockier image. These are descriptions of the displayed pictures,
    not independent confirmation of the labels or proof of what the model uses.
    Accept other specific observations tied to an image ID.
    ''', "image-question")]

    cells += [md('''
    ## 2. What makes a useful shortlist?

    A **shortlist** is the first few images in a model's ordered list. Two questions
    tell us how useful it is:

    - **Precision:** of the images we review, what fraction are labelled tells?
    - **Recall:** of all the tell-labelled images available, what fraction do we find?

    Practise on five invented images. Their labels below are already ordered by an
    imagined model, best candidate first. A **positive** means label `1` (tell);
    a negative means label `0`.
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
    **Try it:** before running, write the two fractions for the first four entries.
    Change `k = 2` to `k = 4` in the cell above and rerun that cell to check.

    **Your notes:** precision = ___ / ___; recall = ___ / ___. Which extra labels
    did you include, and why did precision fall while recall rose?
    ''' if not instructor else '''
    **Worked answer for `k = 4`:** the shortlist is `[1, 1, 0, 1]`. Precision is
    `3/4 = 75%`; recall is `3/3 = 100%`. The two added images include one tell and
    one negative: we now find every tell, but spend one review on a negative.
    ''', "shortlist-discussion")]

    cells += [md('''
    ## 3. Compare two models

    Which model puts more tells into the first **50 images**? Both learn from the
    same training examples and are checked on the same validation images.

    Here is the process we will run:

    1. Turn each image into a list of numbers called **features**.
    2. **Train** a model: use the features and known labels to learn a scoring rule.
    3. Give that rule the features of new images, without their labels. It produces
       a score for each image; we review the highest scores first.
    4. Use the validation labels to count how many tells each review list contains.

    ### 3.1. Turn images into features

    An image must be represented by numbers before these models can use it. We compare
    two ways to make those numbers:

    | Model name in the code | Numbers per image | How the numbers are made |
    |---|---:|---|
    | `colour_texture` | 68 | Measurements of colour, texture and edges. |
    | `dinov2` | 384 | An existing image model, DINOv2, describes visual patterns. |

    DINOv2 learned to describe images before this workshop. Its descriptions are supplied;
    we do not retrain it. We will teach a separate scoring rule to use each set of numbers
    to distinguish the Enki labels.

    A **feature matrix** is a table of these numbers: one image per row, one feature
    per column. Labels stay separate. Run the cell to prepare both tables.
    ''', "models-introduction"), supplied('''
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
            "Model": name,
            "Training images": features["train"].shape[0],
            "Validation images": features["validation"].shape[0],
            "Numbers per image": features["train"].shape[1],
        })
    display(pd.DataFrame(feature_summary))
    ''', "features", title='prepare image features'), md('''
    ### 3.2. Fit the classifiers

    A **classifier** learns to distinguish labels. We use **logistic regression**:
    it learns how to weight and combine the features into a score between 0 and 1.
    Training adjusts those weights using the known training labels.

    The code joins two steps in a **pipeline**. `StandardScaler` first puts feature
    values on comparable scales, so a measurement is not larger just because of its
    units. `LogisticRegression` then learns the scoring rule. Both steps learn from
    **training data only**.

    Run the cell as given. In `.fit(features["train"], train.label)`, the first argument
    is the training feature table; the second is its matching answers. The other
    settings are fixed for this comparison.
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

    Run the next cell as given. Each trained model scores the 119 validation images
    without seeing their labels. `predict_proba(...)` returns a score for each label;
    `[:, 1]` takes the tell score for every image.

    We sort from highest to lowest score. **Rank 1** is the image reviewed first.
    A score of `0.9` is a model output, not evidence that it is correct 90% of the time;
    we check the ranking against the known labels in the next task.

    The galleries show only the first four images from each ranking as a preview.
    The calculation afterwards uses the first **50**.
    ''', "validation-explanation"), supplied('''
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

    ''', "score-validation", title='rank the validation images'), md('''
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
    | Tells found | Tell-labelled images in the review list. |
    | Images reviewed | Number of images in that list. |
    | Precision | Percentage of reviewed images that are tells. |
    | Recall | Percentage of all available tells found. |

    We will keep the model that finds more tells in these 50 reviews for the rest
    of the notebook. If the counts tie, neither wins this comparison.
    ''', "model-comparison-instructions"), supplied('''
    REVIEW_BUDGET = 50
    validation_size = len(validation)
    validation_positives = int(validation.label.sum())
    print(f"Validation set: {validation_size} images, including {validation_positives} labelled tells.")
    print(f"Shortlist budget: {REVIEW_BUDGET} images per model.")

    rows = []
    for name, scores in validation_scores.items():
        metrics = workshop.ranking_metrics(validation.label, scores, k=REVIEW_BUDGET)
        row = {"model": name}
        row.update(metrics)
        rows.append(row)

    # At the same budget, more tells found is the comparison we care about.
    # Name order makes an exact tie reproducible; it does not establish a winner.
    results = pd.DataFrame(rows).sort_values(
        ["hits", "model"], ascending=[False, True],
    ).reset_index(drop=True)
    selected = results.iloc[0]["model"]
    comparison_display = results[["model", "hits", "reviewed"]].rename(columns={
        "model": "Model", "hits": "Tells found", "reviewed": "Images reviewed"})
    comparison_display["Precision"] = results.precision_at_k.map(lambda value: f"{value:.1%}")
    comparison_display["Recall"] = results.recall_at_k.map(lambda value: f"{value:.1%}")
    display(comparison_display)

    winner, runner_up = results.iloc[0], results.iloc[1]
    extra_hits = int(winner["hits"] - runner_up["hits"])
    print(f"Selected using validation: {selected}")
    if extra_hits > 0:
        print(f"Its shortlist contains {extra_hits} more tells than {runner_up['model']}, "
              "using the same number of slots.")
    else:
        print("The tell counts tie. We use the first model by name to continue; neither finds more tells.")
    ''', "compare-models", title='compare the two review lists'), md('''
    ### 3.5. Compare with random selection and the budget limit

    Would the same number of reviews find this many tells without a model? Picking
    50 images at random from 119 would find about 25 of the 60 tells on average.
    An individual random selection could find more or fewer.

    At the other extreme, even a perfect ordering can fit only 50 of the 60 tells
    into 50 places. Its recall would be `50/60 = 83.3%`, not 100%.
    Run the cell to put your model between these two reference points.
    ''', "comparison-references"), code('''
    shortlist_size = min(REVIEW_BUDGET, validation_size)
    random_recall = shortlist_size / validation_size
    random_hits = random_recall * validation_positives
    maximum_hits = min(shortlist_size, validation_positives)
    recall_ceiling = maximum_hits / validation_positives

    print(f"Random selection: {random_hits:.1f} tells on average in {shortlist_size} reviews; "
          f"expected recall = {shortlist_size}/{validation_size} = {random_recall:.1%}.")
    print(f"Best possible: {maximum_hits} tells in {shortlist_size} reviews; "
          f"maximum recall = {maximum_hits}/{validation_positives} = {recall_ceiling:.1%}.")
    print(f"Selected model: {int(winner['hits'])} tells in the shortlist, "
          f"{validation_positives - int(winner['hits'])} outside it.")
    ''', "validation-benchmarks"), md('''
    ### 3.6. A limitation of this image collection

    All tell images were supplied at 350 × 350 pixels, but most negatives at 64 × 64.
    A rule that guesses “tell” for large images could therefore get many labels right
    without recognising a mound. This is an example of learning an accidental clue
    in the collection rather than the thing we want to find.

    We resize every image to 64 × 64 before making features. That removes the direct
    size difference, but other clues, such as image quality, may remain. We have not
    established which clues the models use. Treat these results as evidence about
    **this collection**; a new landscape still needs its own evaluation.
    ''', "resolution-audit-explanation"), md('''
    **Your notes — complete these sentences:**

    - We keep ___ because it finds ___ tells in 50 reviews, compared with ___ for the other model.
    - Its precision is ___ and its recall is ___. Random selection finds about ___ tells; the best possible list finds ___.
    - These results alone cannot tell us whether it will work in Cyprus because ___.
    ''' if not instructor else '''
    **Worked conclusion for the default run:** keep `dinov2`: 47 tells in 50 reviews,
    compared with 40 for `colour_texture`. Precision is 94.0%; recall is 78.3%.
    Random selection averages about 25 tells and the best possible list contains 50.
    These results describe this image collection, whose labels are associated with
    how images were supplied; they do not establish performance in Cyprus.
    ''', "model-question")]

    cells += [md('''
    ## 4. Inspect the shortlist

    The chosen model makes an ordered list, not a final archaeological judgement.
    Relative to the supplied labels, we can identify:

    - **False alarm:** a negative image in the first 50. It uses a review slot.
    - **Miss:** a tell image at rank 51 or later. We would not reach it in 50 reviews.

    Run the cell to see the counts and up to four examples of each. With 60 tells
    and only 50 places, at least ten tells must remain outside even a perfect list.
    ''', "mistakes-introduction"), supplied('''
    review = validation_rankings[selected].copy()
    shortlist = review.head(REVIEW_BUDGET)
    false_alarms = shortlist.query("label == 0")
    misses = review.iloc[REVIEW_BUDGET:].query("label == 1")

    print(f"{selected}: {len(shortlist)} images in the shortlist")
    print(f"Inside: {int(shortlist.label.sum())} tells, {len(false_alarms)} negatives.")
    print(f"Outside: {len(misses)} tells missed at this budget.")

    for title, group in (
        ("False alarms: negative images inside the first 50", false_alarms),
        ("Misses: tell images after the first 50", misses),
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
    **Your notes:** choose one displayed image and record its ID, rank, and whether
    it is a false alarm or a miss. Describe something visible in it. Seeing vegetation,
    for example, does not prove that vegetation caused its score; that would require
    comparing other images too.
    ''' if not instructor else '''
    **Worked observation for the default model:** `0dd4324ef7a1c904` is a miss at rank 51.
    It shows dark water beside lighter terrain and field patterns. Its score is about
    0.906, but the review list stops at rank 50: a high score can still fall outside a
    fixed budget. The visible water does not establish why the model ranked it there.
    ''', "mistake-question"), md('''
    ### Try ten more reviews

    **Before running:** will ten more reviews necessarily find ten more tells?
    Change `inspection_budget = 50` to `60` in the next cell and run it.
    The image order stays the same; you now also review ranks 51–60.

    Keep `REVIEW_BUDGET = 50` above unchanged: this is a separate experiment.
    ''', "inspection-budget-exercise"), code('''
    inspection_budget = 50  # Change to 60 to inspect ten more images.

    added = review.iloc[REVIEW_BUDGET:inspection_budget]
    remaining_misses = review.iloc[inspection_budget:].query("label == 1")
    new_false_alarms = added.query("label == 0")

    print(f"Extra images reviewed: {len(added)}")
    print(f"Extra tells found: {int(added.label.sum())}")
    print(f"Extra negatives reviewed: {len(new_false_alarms)}")
    print(f"Tells still outside: {len(remaining_misses)}")
    display(added[["rank", "tile_id", "label", "score"]].round(6))
    workshop.contact_sheet(new_false_alarms, new_false_alarms.score.to_numpy(), n=4,
                           title=f"{selected}: extra negative images at {inspection_budget} reviews")
    ''' if not instructor else '''
    inspection_budget = 60  # Worked example: inspect ten more images.

    added = review.iloc[REVIEW_BUDGET:inspection_budget]
    remaining_misses = review.iloc[inspection_budget:].query("label == 1")
    new_false_alarms = added.query("label == 0")

    print(f"Extra images reviewed: {len(added)}")
    print(f"Extra tells found: {int(added.label.sum())}")
    print(f"Extra negatives reviewed: {len(new_false_alarms)}")
    print(f"Tells still outside: {len(remaining_misses)}")
    display(added[["rank", "tile_id", "label", "score"]].round(6))
    workshop.contact_sheet(new_false_alarms, new_false_alarms.score.to_numpy(), n=4,
                           title=f"{selected}: extra negative images at {inspection_budget} reviews")
    ''', "inspection-budget"), md('''
    **Your notes:** the ten extra reviews found ___ tells and ___ negatives, leaving
    ___ tells outside the list. The first two counts should add to ten.
    Would you spend the extra review time? Give one reason.

    **Optional challenge:** find the rank of the last tell in the ordered list.
    That is the number of reviews needed to find all 60 on this sample. You can do
    this by increasing `inspection_budget` until “Tells still outside” reaches zero,
    then finding the smallest budget that gives zero.
    ''' if not instructor else '''
    **Worked answer for the default model:** ten extra reviews add five tells and
    five negatives, leaving eight tells outside the list. Recall rises from `47/60`
    to `52/60`, while precision falls from `47/50` to `52/60`. Whether the five extra
    tells justify ten reviews depends on the survey team's resources and priorities.
    A reasoned answer either way is valid.
    ''', "inspection-budget-notes")]

    cells += [md('''
    ## 5. Try the same model in Cyprus

    ### 5.1. Score a different landscape

    Now keep the models you already trained and give them **12 images from Cyprus**.
    This asks whether a rule learned from Enki images is useful somewhere else.
    Cyprus settlement remains need not look like tells; these satellite images also
    show different landscapes and levels of detail.

    There are six pairs. Each pair contains an image centred on a settlement recorded
    in an archaeological catalogue and an image centred 2 km away. Each image covers
    1 km × 1 km; one original pixel covers 10 m × 10 m. They come from Sentinel-2,
    acquired on 5 August 2024.

    | Image description | What we know |
    |---|---|
    | Catalogued location | A settlement is recorded here, but we do not know whether its remains are visible in this image. |
    | Unlabelled neighbour | No settlement label is supplied here; this does not prove that archaeology is absent. |

    Enki supplied image labels to check our lists. Here we have location records,
    not answers saying **visible archaeology / no visible archaeology** for each image.
    We therefore cannot calculate detection accuracy.
    We can still ask a useful, narrower question: **does the model put the recorded
    settlement locations near the top of its list?** You will calculate that in 5.3.

    Run the cell to view the chosen model's first four Cyprus images. A high score
    means “review earlier” in this list; it is not a measured chance of archaeology.
    ''', "cyprus-introduction"), supplied('''
    cyprus, cyprus_dino = workshop.load_cyprus(assets)
    cyprus_features = {
        "colour_texture": workshop.handcrafted_features(cyprus.path),
        "dinov2": cyprus_dino,
    }
    cyprus_scores = {}
    for name, model in models.items():
        cyprus_scores[name] = model.predict_proba(cyprus_features[name])[:, 1]
        cyprus[f"{name}_score"] = cyprus_scores[name]
    cyprus["score"] = cyprus_scores[selected]
    cyprus = cyprus.sort_values("score", ascending=False, kind="stable").reset_index(drop=True)
    cyprus.insert(0, "rank", range(1, len(cyprus) + 1))
    workshop.contact_sheet(cyprus, cyprus.score.to_numpy(), n=4, show_labels=False,
                           title=f"Cyprus: four highest scores from {selected}")
    ''', "cyprus", title='score and preview Cyprus images'), md('''
    ### 5.2. Compare two pairs

    1. Run the cell with `pair_id = "CY5"`. The recorded settlement is on the left;
       its unlabelled neighbour is on the right. Which gets the higher score?
    2. Change only `"CY5"` to `"CY479"` and rerun this cell. Which image gets the higher
       score this time? Compare with your first pair.
    3. Write your answers below. Describe what you see without assuming that a bright
       patch, field boundary or road is archaeology.
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
    **Your notes — replace the prompts below:**

    - Higher-scoring image in `CY5`:
    - Higher-scoring image in `CY479`:
    - One named image and a visible feature:
    - What would an expert need to check before deciding whether the model found archaeology?
    ''' if not instructor else '''
    **Worked observation for the default model:** `CY5_background` scores higher than
    `CY5_site` (0.888 versus 0.765), whereas `CY479_site` scores higher than its neighbour
    (0.810 versus 0.431). `CY5_site` shows bright angular patches; its neighbour has
    pale winding lines through darker terrain. This describes the images, not the
    classifier's reasoning. Expert checks of what is visible are needed before calling
    either ordering a detection error.
    ''', "cyprus-pair-notes"), md('''
    ### 5.3. Calculate agreement with the Cyprus catalogue

    For this small set, suppose you have time to review **four of the twelve images**.
    We will count how many recorded settlement locations those four include.
    This is the same counting idea as section 3, but catalogue records replace the
    tell labels, so the result answers a different question.

    We also check each pair: a **win** means the recorded location scores higher than
    its neighbour; a **loss** means lower; a **tie** means equal. A win is agreement
    with the catalogue, not a verified archaeological discovery.

    Run the cell for the six pair results, the four-image shortlist and your counts.
    ''', "cyprus-metrics-introduction"), supplied('''
    CYPRUS_BUDGET = 4
    cyprus_pairs, cyprus_counts = workshop.catalogue_agreement(cyprus, k=CYPRUS_BUDGET)
    display(cyprus_pairs.rename(columns={
        "paired_site": "Pair", "catalogued_score": "Recorded location score",
        "unlabelled_score": "Neighbour score", "outcome": "Pair result",
    })[["Pair", "Recorded location score", "Neighbour score", "Pair result"]].round(6))
    print(f"First {CYPRUS_BUDGET} images in the selected model's ranking:")
    cyprus_shortlist = cyprus.head(CYPRUS_BUDGET)[["rank", "tile_id", "kind", "score"]].copy()
    cyprus_shortlist["kind"] = cyprus_shortlist["kind"].map({
        "catalogued_location": "Recorded settlement", "unlabelled_background": "Unlabelled neighbour"})
    display(cyprus_shortlist.rename(columns={"rank": "Rank", "tile_id": "Image ID",
                                           "kind": "Location", "score": "Score"}).round(6))

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
    print(f"Calculation variables: pair_wins={pair_wins}, pair_ties={pair_ties}, pair_count={pair_count}; "
          f"cyprus_hits={cyprus_hits}, cyprus_reviewed={cyprus_reviewed}, catalogued_total={catalogued_total}.")
    ''', "cyprus-counts", title='count the Cyprus results'), md('''
    **Calculate how well the model ranks the catalogued Cyprus locations.** Replace
    `...` on each of the three lines below, using these counts:

    1. `pair_agreement`: how often does the catalogued image score higher than its neighbour?
       Use `(pair_wins + 0.5 * pair_ties) / pair_count`. Half credit treats an equal
       score as no preference for either image, halfway between a win and a loss.
    2. `catalogue_fraction`: what fraction of the reviewed images are catalogued locations?
       Divide `cyprus_hits` by `cyprus_reviewed`.
    3. `catalogue_recovery`: what fraction of all catalogue locations did the review list find?
       Divide `cyprus_hits` by `catalogued_total`.

    Run your calculation cell, then **Supplied · check your Cyprus calculations**.
    ''', "cyprus-calculation-instructions"), code('''
    pair_agreement = ...  # Include half credit for ties.
    catalogue_fraction = ...  # Fraction of reviewed images that are catalogued.
    catalogue_recovery = ...  # Fraction of all catalogued locations recovered.

    if any(value is Ellipsis for value in (pair_agreement, catalogue_fraction, catalogue_recovery)):
        print("Replace ... on all three lines, then rerun this cell and the check below.")
    else:
        print(f"Pair agreement: {pair_agreement:.1%}")
        print(f"Catalogue fraction in shortlist: {catalogue_fraction:.1%}")
        print(f"Catalogue recovery: {catalogue_recovery:.1%}")
    ''' if not instructor else '''
    pair_agreement = (pair_wins + 0.5 * pair_ties) / pair_count
    catalogue_fraction = cyprus_hits / cyprus_reviewed
    catalogue_recovery = cyprus_hits / catalogued_total

    print(f"Pair agreement: {pair_agreement:.1%}")
    print(f"Catalogue fraction in shortlist: {catalogue_fraction:.1%}")
    print(f"Catalogue recovery: {catalogue_recovery:.1%}")
    ''', "cyprus-calculation"), md('''
    **Check and interpret:** the next cell checks your divisions and compares them
    with random selection. With half the images centred on recorded settlements,
    random choices favour the recorded image in half the pairs and include about
    two recorded locations in four reviews: `2/4 = 50%` of the shortlist, and
    `2/6 = 33.3%` of all six recorded locations.

    These are averages over many random choices, not guaranteed results for one list.
    ''', "cyprus-check-explanation"), supplied('''
    cyprus_metric_rows, cyprus_pair_rows = [], []
    for name in models:
        pairs_for_model, metrics = workshop.catalogue_agreement(
            cyprus, score_column=f"{name}_score", k=CYPRUS_BUDGET)
        cyprus_metric_rows.append({"model": name, **metrics})
        cyprus_pair_rows.append(pairs_for_model.assign(model=name))
    cyprus_metrics = pd.DataFrame(cyprus_metric_rows)
    cyprus_pair_results = pd.concat(cyprus_pair_rows, ignore_index=True)
    cyprus_reference = cyprus_metrics.set_index("model").loc[selected]
    cyprus_table = pd.DataFrame({"Measure": ["Pair agreement", "Catalogue fraction", "Catalogue recovery"],
        selected: [f"{cyprus_reference[name]:.1%}" for name in
                   ("pair_agreement", "catalogue_fraction_at_k", "catalogue_recovery_at_k")],
        "Random selection (average)": [f"{cyprus_reference[name]:.1%}" for name in
                   ("random_pair_agreement", "random_catalogue_fraction", "random_catalogue_recovery")]})
    display(cyprus_table)
    print(f"{selected}: pair agreement = ({pair_wins} + 0.5 × {pair_ties})/{pair_count} "
          f"= {cyprus_reference.pair_agreement:.1%}.")
    print(f"Catalogue fraction = {cyprus_hits}/{cyprus_reviewed} "
          f"= {cyprus_reference.catalogue_fraction_at_k:.1%}; "
          f"catalogue recovery = {cyprus_hits}/{catalogued_total} "
          f"= {cyprus_reference.catalogue_recovery_at_k:.1%}.")
    for row in cyprus_metrics.loc[cyprus_metrics.model.eq(selected)].itertuples(index=False):
        if row.cutoff_tie:
            print(f"{row.model}: equal scores straddle the fourth place, so different orders of those images give different lists; "
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
    print("Cyprus detection accuracy: unknown; we lack expert labels of what is visible in each image.")
    ''', "cyprus-metrics-check", title='check your Cyprus calculations'), md('''
    **Your notes — complete these sentences:**

    - The model puts ___ recorded locations in four reviews, compared with about two for random selection.
    - This recovers ___ of the six recorded locations (___%).
    - This is not detection accuracy because we do not know ___.

    Six selected pairs are too few to represent all Cyprus; some of their image areas
    overlap. Keep your conclusion about these images, not the whole island.
    ''' if not instructor else '''
    **Worked interpretation for the default model:** four wins in six pairs gives
    66.7% agreement. The four-image list includes three recorded locations, compared
    with two on average for random selection. This finds `3/6 = 50%` of the recorded
    locations. It does not measure detection accuracy: we do not know which images
    actually show archaeology. The six selected pairs also cannot represent all Cyprus.
    ''', "cyprus-metrics-discussion")]

    cells += [md('''
    ## Optional lab — choose one before section 6

    Allow about 30 minutes for **one lab**, or continue to section 6.

    | Lab | Task |
    |---|---|
    | A. Picture the features | Can a two-dimensional plot show the tell and negative groups? |
    | B. More training examples | Does learning from more labelled images improve the review list? |
    | C. Find similar images | Do similar-looking examples justify trusting a Cyprus score? |

    Each lab supplies working code. Your task is to predict, change a setting, inspect
    the result and record a conclusion. No new Python function is required.
    Only run the cells in your chosen lab. Afterwards go to section 6.
    ''', "optional-experiments"), md('''
    ### Lab A. See the training features in 2D or 3D

    **Question:** do tell images and negative images form separate groups when we
    plot their features?

    Each image has 68 or 384 feature values, too many to plot on ordinary axes.
    **Principal component analysis (PCA)** compresses those values into two coordinates
    per image. It keeps as much of the variation between images as it can, but loses
    some information. It does not use tell/negative labels to position the points.

    1. Predict whether the two label groups will separate more clearly for one model.
    2. Run the supplied function cell. Set `RUN_PROJECTION = True` in the settings cell
       and run it, then run **Supplied · draw the PCA plots**.
    3. Read each panel separately: every point is one training image, coloured by its
       supplied label. Look for overlap or distinct groups.

    The “variation retained” percentage says how much of the original variation this
    compressed picture preserves. It is not a percentage of correct labels.
    ''', "projection-instructions"), supplied('''
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
    ''', "projection-exercise", title='define the feature plot'), code(f'''
    RUN_PROJECTION = {instructor}  # Change to True to run this lab.
    N_COMPONENTS = 2         # Start with 2; change to 3 for the extension.
    ELEVATION, AZIMUTH = 20, -60  # Camera angles for the 3D view, in degrees.
    ''', "projection-settings"), supplied(f'''
    if not RUN_PROJECTION:
        print("Lab A skipped. To try it, set RUN_PROJECTION = True, run settings, then rerun this cell.")
    else:
        if N_COMPONENTS not in (2, 3):
            raise ValueError("Choose N_COMPONENTS = 2 or 3 for this plotting exercise.")

        fig = plt.figure(figsize=(13, 6))
        fig.subplots_adjust(left=0.07, right=0.94, bottom=0.16, top=0.84, wspace=0.28)
        labels = train.label.to_numpy()  # Same row order as the training feature matrices.
        class_styles = [
            (0, "Negative (source label 0)", "#2874A6", "o"),
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

            ax.set_xlabel(f"PCA axis 1 ({{variance_ratio[0]:.1%}} of variation)")
            ax.set_ylabel(f"PCA axis 2 ({{variance_ratio[1]:.1%}} of variation)")
            if N_COMPONENTS == 3:
                ax.set_zlabel(f"PCA axis 3 ({{variance_ratio[2]:.1%}} of variation)")
                ax.view_init(elev=ELEVATION, azim=AZIMUTH)
            ax.set_title(f"{{name}}: {{variance_ratio.sum():.1%}} of variation retained")
            ax.legend(loc="upper right", fontsize=8)
            print(f"{{name}}: {{len(coordinates)}} images shown using {{N_COMPONENTS}} coordinates each.")

        fig.suptitle(f"PCA of {{len(train)}} training images — source labels added for display")
        plt.show()
    ''', "projection-plot", title='draw the PCA plots'), md('''
    **Your notes:** in the colour/texture panel I see ___; in the DINOv2 panel I see ___.
    Is separation clearer in either panel, or is there no clear winner? Compare with
    section 3: does this picture make the better validation result obvious?

    Overlap in this picture does not mean the full feature tables are identical:
    compression discards information. A plot of training images also cannot replace
    checking new images, which is why we used validation data.

    **Optional extension:** set `N_COMPONENTS = 3`, run settings and the plotting cell
    again. Change `AZIMUTH` from `-60` to `30` and repeat to view the same points from
    another angle. Does the apparent separation change with the view?
    ''' if not instructor else '''
    **Worked interpretation:** the default two-dimensional panels show considerable
    overlap. Colour/texture retains 32.3% of the variation; DINOv2 retains 22.4%.
    There are areas dominated by one label, but neither picture cleanly separates all
    tells from negatives. These views do not establish which classifier works better;
    section 3's validation count favours DINOv2 (47 versus 40 tells in 50 reviews).

    A third coordinate or another viewing angle may reveal structure, but still cannot
    replace checking unseen examples. Accept descriptions grounded in the panels,
    including “no clear winner”.
    ''', "projection-discussion"), md('''
    ### Lab B. Does learning from more examples help?

    Imagine you can afford labels for only 20 images. Would that give you as useful
    a review list as the model trained on 200 images in section 3?

    This experiment trains a fresh `dinov2` classifier on a smaller set and compares
    it with the existing classifier. The DINOv2 features are unchanged. The difference
    is how many labelled examples the classifier learns from.

    | Changes | Stays the same |
    |---|---|
    | Training images: first 20, then 50 | The 119 validation images and their labels |
    | Which training examples are selected | A review budget of 50 images for both classifiers |
    | | The comparison with the original classifier trained on 200 images |

    **Before running:** predict whether more training examples will find more tells.
    Will every tell found by the 20-image classifier also be found by the 200-image one?

    **First run:** run the supplied function below. Set `RUN_LABEL_EXPERIMENT = True`,
    leave `B_TRAINING_IMAGES = 20` and `B_SEED = 0`, then run the settings cell and
    **Supplied · compare training sets and changed images**.

    `B_SEED` chooses a repeatable sample: the same seed **and training size** select
    the same examples. Changing the seed selects a different sample; increasing the
    size at a fixed seed adds examples. Each set is half tells, half negatives.
    ''', "label-experiment-instructions"), supplied('''
    from sklearn.base import clone


    def small_training_scores(name, rows):
        X_small = feature_sets[name]["train"][rows]
        y_small = train.label.to_numpy()[rows]
        X_validation = feature_sets[name]["validation"]

        # A fresh copy learns only from this subset; the original model is unchanged.
        lab_model = clone(models[name])
        lab_model.fit(X_small, y_small)
        return lab_model.predict_proba(X_validation)[:, 1]
    ''', "label-experiment-function", title='define the training experiment'), code(f'''
    RUN_LABEL_EXPERIMENT = {instructor}  # Change to True to run this lab.
    B_MODEL = "dinov2"
    B_TRAINING_IMAGES = 20  # Total: half tells, half negatives. Then try 50.
    B_SEED = 0             # Then try 1 with 20 images to change which examples are used.
    ''', "label-experiment-settings"), supplied('''
    if not RUN_LABEL_EXPERIMENT:
        print("Lab B skipped. To try it, set RUN_LABEL_EXPERIMENT = True, run settings, then rerun this cell.")
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
            print("No scores returned. Rerun the supplied training-function cell, then this cell.")
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
            for title, changed in (("Tells found with 200 training images but missed with fewer", b_gained),
                                   ("Tells missed with 200 training images but found with fewer", b_lost)):
                print(f"{title}: {len(changed)} images; showing up to four.")
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
    **Read the result:** “enter” means a tell was outside the first 50 with the small
    training set, but inside with 200 training images. “Leave” means the reverse.
    For example, rank `58 → 42` means that image would now be reviewed.
    The pictures let you inspect which tells changed places.

    **Two more runs:** change to `B_TRAINING_IMAGES = 50`, keeping `B_SEED = 0`.
    Run settings and the comparison cell again. Then use `B_TRAINING_IMAGES = 20`
    and `B_SEED = 1` and repeat. The first change adds training examples; the second
    checks a different selection of 20. Compare the third run with the first to see
    the effect of choosing different examples at the same training size.
    Do not rerun the earlier sections.

    **Your notes — fill the table from the outputs:**

    | Training images | Sample seed | Tells found in 50 reviews | Tells entering with 200 images | Tells leaving with 200 images |
    |---:|---:|---|---|---|
    | 20 | 0 | | | |
    | 50 | 0 | | | |
    | 20 | 1 | | | |

    Record one changed image's ID and its two ranks. Then answer: did more training
    data improve the total, every individual image, or neither in these runs?
    Use the counts, including cases where they stayed equal. Three runs do not establish
    what will happen for every possible training sample.

    **Optional extension:** repeat with `B_MODEL = "colour_texture"`.
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

    **Question:** if a Cyprus image resembles tell-labelled training images, is that
    enough to trust a high model score?

    The matches give you concrete training examples to inspect alongside the new image.
    Look for shared patterns and for differences in detail before drawing a conclusion.

    We will use DINOv2's image descriptions as an image-search tool. For each Cyprus
    image, the search returns the **four Enki training images with the most similar
    feature values**. It ignores their labels while searching; the gallery adds the
    labels afterwards so you can inspect the matches.

    There are two different numbers to read:

    | Number | Question it answers |
    |---|---|
    | Classifier score, on the Cyprus image | How highly does the rule learned from Enki labels rank this image as a tell candidate? |
    | Feature similarity, on an Enki match | How similar are this image's DINOv2 feature values to those of the Cyprus image? Higher means closer; 1 is the maximum. |

    Neither number is a measurement of whether archaeology is visible in Cyprus.

    **First run:** run the supplied function. Set `RUN_CYPRUS_NEIGHBOURS = True`, keep
    `C_PAIR_IDS = ["CY754"]`, run the settings cell, then run **Supplied · show the
    Cyprus image matches**. Each row shows a Cyprus image on the left and its four
    matches on the right. There are two rows, one for each image in the pair.
    ''', "cyprus-neighbour-instructions"), supplied('''
    def nearest_training_indices(query_vector, training_vectors, k=4):
        # These supplied feature vectors have unit Euclidean norm.
        # Their dot products therefore give cosine similarities.
        similarities = training_vectors @ query_vector
        indices = np.argsort(-similarities, kind="stable")[:k]
        return indices, similarities[indices]
    ''', "cyprus-neighbour-function", title='define the image search'), code(f'''
    RUN_CYPRUS_NEIGHBOURS = {instructor}  # Change to True to run this lab.
    C_PAIR_IDS = {["CY754", "CY479"] if instructor else ["CY754"]}  # Pair IDs to investigate.
    C_NEIGHBOURS = 4
    ''', "cyprus-neighbour-settings"), supplied('''
    if not RUN_CYPRUS_NEIGHBOURS:
        print("Lab C skipped. To try it, set RUN_CYPRUS_NEIGHBOURS = True, run settings, then rerun this cell.")
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
                print("No matches returned. Rerun the supplied search-function cell, then this cell.")
                break
            indices, similarities = matches
            neighbours = train.iloc[indices]
            query_score = models["dinov2"].predict_proba(c_vectors[position:position + 1])[0, 1]
            c_summary_rows.append({"Cyprus tile": query.tile_id, "DINOv2 classifier score": query_score,
                "Tell-labelled neighbours": int(neighbours.label.sum()), "Neighbours": len(neighbours)})
            fig, axes = plt.subplots(1, len(neighbours) + 1, figsize=(3 * (len(neighbours) + 1), 3.5),
                                     layout="constrained")
            axes[0].imshow(plt.imread(query.path))
            axes[0].set_title(f"Cyprus: {query.tile_id}\\nClassifier score {query_score:.3f}", fontsize=9)
            for ax, row, similarity in zip(axes[1:], neighbours.itertuples(index=False), similarities):
                ax.imshow(plt.imread(row.path))
                label = "tell" if row.label == 1 else "negative"
                ax.set_title(f"{row.tile_id} · {label}\\nFeature similarity {similarity:.3f}", fontsize=9)
            for ax in axes:
                ax.axis("off")
            fig.suptitle(f"Cyprus image (left) and its {len(neighbours)} closest Enki training matches")
            plt.show()
        if c_summary_rows:
            c_neighbour_summary = pd.DataFrame(c_summary_rows)
            display(c_neighbour_summary.round(3))
            print("The matches' labels describe Enki images. They do not tell us what is visible in the Cyprus image.")
    ''', "cyprus-neighbour-run", title='show the Cyprus image matches'), md('''
    **Inspect and record:** choose one Cyprus image and one of its Enki matches.
    Record both IDs, one visible resemblance and one difference. Consider colour,
    field shapes, vegetation or roads; “similar features” is not itself a visual observation.

    In the summary table, does the Cyprus image with the higher classifier score also
    have more tell-labelled matches? The classifier learned a weighted rule from all
    200 training images; it does not vote among these four neighbours. Agreement is
    therefore something to check, not something the method guarantees.

    **Second run:** change `C_PAIR_IDS` to `["CY479"]`, run settings and the search
    cell again. Compare with the first pair.

    **Your notes:**

    - Cyprus image ID / matched Enki image ID:
    - Visible resemblance / visible difference:
    - Did the higher-scoring Cyprus image have more tell-labelled matches in each pair?
    - Would these matches be enough to confirm archaeology? What evidence is still needed?
    ''' if not instructor else '''
    **Worked interpretation:** `CY754_site` scores 0.829 with one tell-labelled match;
    `CY754_background` scores 0.060 with four. More tell-labelled matches therefore do
    not imply a higher classifier score. Both `CY479` images have one tell-labelled
    match despite scores of 0.810 and 0.431. The classifier uses a learned weighted
    rule, not a vote among these matches.

    Ask for a named image resemblance and difference from the galleries. Even a convincing
    visual resemblance cannot establish archaeology at a new location: experts would need
    to inspect suitable local imagery or other independent evidence.

    **Example observation:** `CY754_background` and `907992cd68f2dfa0` both show pale
    blocks and lines against brown ground. Individual outlines are harder to distinguish
    in the Cyprus image. This resemblance does not identify the pale blocks as archaeology.
    ''', "cyprus-neighbour-discussion")]


    cells += [md('''
    ## 6. Recommend, save and check

    ### 6.1. Write a five-sentence recommendation

    The survey team wants to know: **should we use this model to choose Cyprus
    images for review?** Recommend a limited trial, more evaluation first, or no use
    yet, based on what you have seen. A strong Enki result alone does not settle this.

    Run the summary to bring the relevant counts together, then write your paragraph.
    ''', "report-introduction"), supplied('''
    report_validation = results.loc[results.model.eq(selected)].iloc[0]
    report_cyprus = cyprus_metrics.loc[cyprus_metrics.model.eq(selected)].iloc[0]
    print(f"Selected model: {selected}")
    print(f"Enki validation: {int(report_validation.hits)} of {int(validation.label.sum())} "
          f"tell-labelled images in {int(report_validation.reviewed)} reviews; "
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
    Replace the blank between `"""` and `"""` in the code cell below with **five
    sentences**. These starters describe what each sentence should do:

    1. “I recommend …” — name the model and say whether, and where, you would use it.
    2. “On Enki it found …” — give tells found out of 60 in 50 reviews, precision and recall.
    3. “In image … I observed …” — use one named image from your notes.
    4. “In Cyprus the list included …” — give recorded locations found out of six;
       explain why this does not measure detection accuracy.
    5. “Before relying on it, we need …” — name the local images and expert checks
       needed, including locations kept separate from training and model choice.

    Reuse your notes. Read the paragraph to a partner, or reread it yourself, and revise
    any claim that goes beyond the evidence.
    ''', "report-prompts"), code('''
    # Write five sentences using the evidence and prompts above.
    recommendation = """

    """.strip()
    print(recommendation or "Draft: write your recommendation above, then rerun this cell.")
    ''' if not instructor else '''
    recommendation = (
        f"I recommend further evaluation of {selected} before using it to prioritise Cyprus imagery. "
        f"It retrieves {int(report_validation.hits)} of {int(validation.label.sum())} validation "
        f"tell-labelled images in {int(report_validation.reviewed)} reviews "
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

    Run the export cell. It saves your paragraph and the model results in
    **`team_submission.zip`**. Download it using the displayed link or Colab's Files
    panel. Also save this notebook: **File → Save** in Jupyter, or **File → Download →
    Download .ipynb** in Colab. The notebook keeps your notes and figures.

    Give both files to the facilitator before the reveal. If working alone, save
    both before continuing. The final test labels have not yet been used to check results.

    A blank paragraph produces a draft. Finish it and rerun export before sharing;
    rerun export after any further edits too.
    ''', "export-introduction"), supplied('''
    # Preserve the technical audits in the ZIP; their explanations are in the README.
    validation_audit = workshop.validation_diagnostics(
        train, validation, validation_scores, k=REVIEW_BUDGET)
    transfer_audit = workshop.transfer_summary(validation.label, validation_scores, cyprus_scores)
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

    You chose a model using validation images. The test checks that saved choice
    on 119 other Enki images, using the same 50-review budget.

    After saving or sharing your files, set `REVEAL_TEST = True`, run the settings
    cell, then run the supplied comparison below. Compare the counts and percentages.
    Keep your original recommendation and add the debrief underneath.

    Do not switch models after seeing this result: you would then be using the test
    to choose a model too. All these images come from Enki; we have not verified that
    they are from separate geographic areas. This is not a test of performance in Cyprus.
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
    **Debrief — complete one sentence here:** the Enki test [supports / weakens /
    does not change] my recommendation because ..., while ... still needs to be tested.

    Save the notebook again to keep this reflection alongside your original report.
    ''' if not instructor else '''
    **Worked debrief for the default run:** the test strengthens the evidence for
    useful Enki rankings because 48 tells are found in 50 reviews, close to validation's
    47, but does not change my recommendation to evaluate Cyprus performance before use.

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
