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
    **Participant notebook · four-hour workshop with a choice of optional lab**

    An archaeological survey team has more overhead images than it can inspect carefully.
    Can a model help decide which to open first? We use **50 image reviews** as a teaching
    budget so that every model is compared at the same effort. This is a scenario, not
    a measured limit on a real team's work.

    Our initial target is a *tell*: a mound built up through repeated human occupation.
    [Enki](https://zenodo.org/records/14950565) supplies image crops labelled as tell (`1`)
    or non-tell/background (`0`). We use these source labels to evaluate rankings; we
    do not trace site boundaries or independently verify discoveries. DINOv2 supplies one
    way to turn images into numbers; the classifiers we fit are logistic regressions.

    The workshop follows one decision:

    1. Inspect the labelled images and their training/validation/test split.
    2. Learn how precision and recall describe a shortlist.
    3. Compare two image representations using validation results.
    4. Inspect the chosen model's included and missed images.
    5. Apply it to Cyprus imagery, where catalogue locations are not equivalent to image labels.
    6. Write a recommendation, export the evidence, then assess the saved choice on test images.

    ### How to work through this notebook

    Allow **about 3½ hours for the core path**, including discussion, a short break and
    some setup support. The four-hour plan leaves about **30 minutes for one optional lab
    or catch-up**. These are planning estimates for learners with basic Python and pandas;
    internet or unfamiliar software can add time.

    Run cells in order and pause at each exercise before running the result below it.
    Complete two arithmetic expressions for Enki in section 3.4 and three for Cyprus
    in section 5.3. Other exercises change one value or interpret output. Keep brief notes for
    the final recommendation rather than writing a separate report for each question.
    If working alone, write a short answer in place of pair discussion.

    The **optional lab is between sections 5 and 6**. Choose one activity if time permits,
    or skip the whole block and continue to section 6. Optional plots start disabled.
    Finish any experiments before exporting and revealing the test result.

    Submit **`team_submission.zip` and your saved notebook**. The ZIP contains the ranked
    Enki test CSV, settings, results and a 4–6 sentence recommendation. The notebook keeps
    your exercise notes and figures. Leave `REVEAL_TEST = False` until the final debrief.

    Use Jupyter or [Google Colab](https://colab.research.google.com/). Setup installs missing
    packages and downloads about 73 MB of images; allow about 400 MB of disk space. No GPU,
    API key or model download is needed. For an offline image load, place the facilitator's
    `enki_workshop_sample.zip` in `archaeology_workshop/` before section 1's loading cell.
    Dependencies must already be installed if internet is unavailable.
    ''' if not instructor else '''
    # From tells to Cyprus
    **Instructor copy · four-hour workshop with a choice of optional lab**

    An archaeological survey team has more overhead images than it can inspect carefully.
    Can a model help decide which to open first? We use **50 image reviews** as a teaching
    budget so that every model is compared at the same effort. This is a scenario, not
    a measured limit on a real team's work.

    Our initial target is a *tell*: a mound built up through repeated human occupation.
    [Enki](https://zenodo.org/records/14950565) supplies image crops labelled as tell (`1`)
    or non-tell/background (`0`). We use these source labels to evaluate rankings; we
    do not trace site boundaries or independently verify discoveries. DINOv2 supplies one
    way to turn images into numbers; the classifiers we fit are logistic regressions.

    The workshop follows one decision:

    1. Inspect the labelled images and their training/validation/test split.
    2. Learn how precision and recall describe a shortlist.
    3. Compare two image representations using validation results.
    4. Inspect the chosen model's included and missed images.
    5. Apply it to Cyprus imagery, where catalogue locations are not equivalent to image labels.
    6. Write a recommendation, export the evidence, then assess the saved choice on test images.

    ### How to work through this notebook

    Allow **about 3½ hours for the core path**, including discussion, a short break and
    some setup support. The four-hour plan leaves about **30 minutes for one optional lab
    or catch-up**. These are planning estimates for learners with basic Python and pandas;
    internet or unfamiliar software can add time.

    Run cells in order and pause at each exercise before running the result below it.
    Complete two arithmetic expressions for Enki in section 3.4 and three for Cyprus
    in section 5.3. Other exercises change one value or interpret output. Keep brief notes for
    the final recommendation rather than writing a separate report for each question.
    If working alone, write a short answer in place of pair discussion.

    The **optional lab is between sections 5 and 6**. Choose one activity if time permits,
    or skip the whole block and continue to section 6. Optional plots start disabled.
    Finish any experiments before exporting and revealing the test result.

    Submit **`team_submission.zip` and your saved notebook**. The ZIP contains the ranked
    Enki test CSV, settings, results and a 4–6 sentence recommendation. The notebook keeps
    your exercise notes and figures. Leave `REVEAL_TEST = False` until the final debrief.

    Use Jupyter or [Google Colab](https://colab.research.google.com/). Setup installs missing
    packages and downloads about 73 MB of images; allow about 400 MB of disk space. No GPU,
    API key or model download is needed. For an offline image load, place the facilitator's
    `enki_workshop_sample.zip` in `archaeology_workshop/` before section 1's loading cell.
    Dependencies must already be installed if internet is unavailable.
    ''', "introduction")]
    if instructor:
        cells += [md('''
    This copy supplies the completed expressions, optional lab plots, worked
    interpretations and final test results. Participants choose one optional lab; this
    copy runs the plots and Lab C practice, with worked answers for all three labs.
    Keep it for the debrief.
    ''', "instructor-introduction")]

    cells += [md('''
    ## 1. Look at the images

    Run the folded setup cell once, then the loading cell below the split explanation.
    Setup unpacks the bundled features, Cyprus images and helper functions. Later code
    uses those helpers through `workshop`; you do not need to edit the setup cell.
    ''', "setup-introduction"), code(f'''
    # @title Setup — run once; no edits needed
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
    Our **600-image sample comes from Enki's training archive**; we create our own
    teaching split within it. The release's separate test archive is not used here.
    We selected 300 source-positive and 300 source-negative crops. This balance does
    not estimate the frequency of tells in a real survey.

    We group near-identical images, keep one per group and exclude groups with conflicting
    labels. **593 images remain**. This reduces overlap from near-copies before splitting:

    | Set | Images with default settings | Role |
    |---|---:|---|
    | Training | 200: 100 tells and 100 backgrounds | Fit both classifiers. |
    | Validation | 119 | Compare the fitted classifiers and choose one. |
    | Test | 119 | Assess the saved choice at the final reveal. |
    | Unused | 155 | Left out to keep the training set small. |

    The 100-per-class limit applies only to training. Labels for validation and test are
    reserved for their respective evaluations.

    **Image-size confound:** all 300 retained tell images are originally 350 × 350;
    249 of the 293 negatives are 64 × 64. File size and sharpness can therefore reveal
    the source label. Both feature pipelines now first resize every image to the same
    **64 × 64 grid**; DINO then enlarges this common input to its required 224 × 224.
    Section 3 measures the original size shortcut and checks performance within each
    original size. This reduces a known shortcut, but does not equalise metres per pixel,
    image rendering or all acquisition differences. Cyprus uses the same preprocessing.

    **Limit of the split:** filtering similar images does not identify every repeated
    site or neighbouring landscape. The sample lacks verified site or region groups, so
    we cannot establish geographic independence. Keep that limitation for the final
    recommendation; section 5 will examine a different setting.
    ''', "images-introduction"), code('''
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
    display(pd.Series({name: len(part) for name, part in splits.items()}, name="tiles").to_frame())
    size_counts = images.groupby(["label", "width", "height"]).size().rename("images").reset_index()
    print("Original dimensions by source label (0 = negative, 1 = tell):")
    display(size_counts)
    audit["native_size_counts"] = size_counts.to_dict("records")
    audit["image_preprocessing"] = workshop.IMAGE_PREPROCESS_ID
    print(f"Both feature pipelines use the same {workshop.COMMON_IMAGE_SIZE} × "
          f"{workshop.COMMON_IMAGE_SIZE} input grid. Preview below shows the original images.")
    workshop.contact_sheet(train.sample(12, random_state=42), title="Training images: 1 = tell, 0 = negative")
    ''', "images"), md('''
    **Warm-up: compare one pair from the 12 training images above.** Their captions use
    the source labels: `1` for tell, `0` for background.

    Choose one of each label. Note the tile IDs and describe one visible difference,
    such as a rounded patch, vegetation or blurred edges. Could it help distinguish
    tell form, or could it reflect how the images were collected?

    The size table establishes an acquisition difference associated with the labels.
    A classifier could use the resulting sharpness or texture as a *shortcut* instead
    of recognising a mound. The table does not establish which cues a fitted model uses.
    Discuss briefly, then continue; no site identification
    or written report is required here.
    ''', "image-question")]

    cells += [md('''
    ## 2. What makes a useful shortlist?

    In a real review, a person opens images to examine possible features. Here we can
    evaluate an ordering using labels already supplied by the source: rank the validation
    images, take the first 50 and count their positive labels.

    - **Precision:** what fraction of the selected images have a positive label?
    - **Recall:** what fraction of all available positives were selected?

    **The next example uses five invented labels, not images.** The list is already in
    an imagined model's score order: `1` means positive, `0` negative. It has no connection
    to the training-image grid; there are no images to open or classify for this calculation.
    ''', "shortlist-introduction"), code('''
    # Supplied labels for a counting exercise; no images are loaded here.
    # Position in the list represents position in an imagined model ranking.
    # 1 = tell-labelled example; 0 = background-labelled example.
    ranked_labels = [1, 1, 0, 1, 0]
    ''', "shortlist-labels"), md('''
    **What does "take the first two" mean?** Select the first two entries of
    `[1, 1, 0, 1, 0]`. That gives the shorter list `[1, 1]`.

    Count three things:

    1. There are **2 entries** in the shorter list.
    2. Both are `1`, so it contains **2 positives**.
    3. The complete five-entry list contains **3 positives**.

    Therefore, precision is `2 / 2 = 100%`: both selected entries are positive.
    Recall is `2 / 3 ≈ 67%`: the selection includes two of the three positives available.

    Run the next cell to carry out these same counts. `k` is the number of list entries
    to include. Python's `[:k]` takes that many entries from the beginning, and `sum()`
    counts the `1`s by adding the labels together.
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
    **Try a larger selection.** Before rerunning, predict the fractions for `k = 4`.
    The selected entries are `[1, 1, 0, 1]`. Count their `1`s, then divide by **4** for
    precision and by **3** (all positives in the full list) for recall.

    Change `k = 2` to `k = 4`, run the cell and explain why the two fractions change
    differently. This uses only the supplied labels, not new image judgements.

    Section 3 applies the same calculation to actual model rankings at `k = 50`.
    We will choose the model that puts more source positives into that fixed shortlist.
    ''', "shortlist-discussion")]

    cells += [md('''
    ## 3. Compare two models

    We now return to the **actual Enki images** split in section 1. Our question is:
    **which of two ways of describing an image helps us put more tell-labelled examples
    into a 50-image shortlist?** Run the cells below in order. They prepare the image
    descriptions, train two classifiers, score the validation images and compare the results.

    ### 3.1. Turn each image into a row of numbers

    The classifier used here takes numbers as input. A **feature** is one number describing
    an image, such as its average red intensity. A **feature vector** is the whole list of
    numbers for one image; a **feature matrix** stacks those lists, with one image per row
    and one feature per column. A source label (`1` or `0`) is kept separately: it is the
    answer we want the classifier to learn to predict.

    We give the same images two different numerical descriptions, also called representations:

    | Name in the code | What describes one image? | Where do the numbers come from? |
    |---|---|---|
    | `colour_texture` | `68` measurements of colour distributions, average colour, variation, and edge strength and direction. | Fixed calculations on the image pixels. This is our simple baseline: a reference against which to judge the more complex representation. |
    | `dinov2` | `384` numbers describing visual patterns. This vector is also called an embedding. | A pretrained neural network converts the image into features. Individual numbers do not have simple names such as “mound height” or “probability of a tell”. |

    **Pretrained** means DINOv2 learned visual representations before this workshop, on a
    large image collection without human-provided training labels. ([DINOv2 documentation](https://github.com/facebookresearch/dinov2))
    We keep that encoder **frozen**: its learned parameters do not change when we use our
    Enki labels. Its vectors have already been computed and bundled in
    `enki_embeddings.npz`, so this notebook loads them without running or training the
    neural network. No GPU is needed for this path.

    Both branches start with the same 64 × 64 RGB image. The DINO cache was rebuilt
    after this correction; the loader rejects old preprocessing versions and changed
    image pixels. The displayed original images can contain detail the model never sees.

    Both representations will feed the **same kind of classifier: logistic regression**.
    We train a separate classifier for each. The comparison is therefore simple image
    measurements plus logistic regression versus DINOv2 features plus logistic regression.
    The question is whether the pretrained representation contains useful distinctions
    that our simple colour and edge summaries miss. The validation results will tell us
    whether that helps with this particular labelled sample.

    Run the next cell to prepare both feature matrices. `partition_features` puts their
    rows into the training, validation and test groups already defined in section 1,
    matching image IDs so each row stays aligned with its label. Preparing these fixed
    descriptions does not fit a classifier or use the held-out labels. In the summary,
    `200` training images and `68` features means a matrix with `200` rows and `68` columns;
    it does not mean we have 68 images.
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
    ### 3.2. Learn from the training labels

    **Logistic regression** is a classification method despite its name. It learns weights
    for the features and an offset, then converts their combination into a score between
    `0` and `1` for label `1`. We fit one classifier per representation on the same
    training images and labels. Validation will choose between them; test remains reserved.

    Each classifier is a **pipeline** with two fitted steps:

    1. `StandardScaler` subtracts each feature's training mean and divides by its training
       standard deviation, putting measurements on comparable scales.
    2. Logistic regression learns weights from those scaled training rows and their labels.

    The pipeline saves the training means and scales and reuses them for later images.
    It does not fit either step on validation or test data.

    Both models use `C=1`, which controls the penalty on large weights; smaller values
    apply a stronger penalty. `class_weight="balanced"` gives classes equal total weight,
    which makes no difference to our already balanced training set. `max_iter=2000`
    limits optimisation iterations. See the [scaler](https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.StandardScaler.html)
    and [classifier](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html)
    references for the remaining options.

    Run the next cell: `.fit(training_features, training_labels)` learns each pipeline,
    and `models` stores it. The classifiers use all 68 or 384 features. Optional lab A
    later projects those features into 2D or 3D without changing these classifiers.
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
    ### 3.3. Score images the classifiers were not fitted on

    Each pipeline receives **validation features only**. `predict_proba` returns two
    columns, for labels `0` and `1`; `[:, 1]` takes the second column for every image.
    The source labels are attached afterwards for evaluation.

    Higher scores mean **inspect earlier**. Although these are probability estimates,
    a score of `0.9` has not been established as a 90% chance of archaeology in a new
    survey. We rank images rather than apply a `0.5` cutoff.

    Run the next cell. It keeps both full rankings and displays the **four highest-scoring
    images per model**, colour/texture followed by DINOv2. Read each row left to right;
    captions give rank, score and tile ID, and the tables include source labels. These
    four-image previews are not the 50-image shortlists used in the next calculation.
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
        print(f"{name}: first {len(preview)} of {len(validation)} ranked validation images")
        display(preview[["rank", "tile_id", "score", "label"]].round(6))

        fig, axes = plt.subplots(1, 4, figsize=(12, 3.5), layout="constrained")
        for ax, row in zip(axes, preview.itertuples(index=False)):
            ax.imshow(plt.imread(row.path))
            ax.set_title(f"Rank {row.rank} · score {row.score:.6f}\\n{row.tile_id}", fontsize=9)
            ax.axis("off")
        fig.suptitle(f"{name}: highest-scoring validation images")
        plt.show()

    # Section 3.4's short exercise still uses the colour/texture ranking.
    ranking_example = validation_rankings["colour_texture"]
    ''', "score-validation"), md('''
    ### 3.4. Count positives in each shortlist and choose a model

    We now apply section 2's fractions to the real validation rankings. Our budget is
    50 images per model. Complete the two expressions below **before** running the comparison.

    The resulting table uses these columns:

    | Column | Meaning |
    |---|---|
    | `model` | The representation used by the fitted classifier. |
    | `hits` | Source-positive tiles among the first 50, not newly discovered sites. |
    | `reviewed` | Number of images selected by the simulated budget. |
    | `precision_at_k` | `hits / reviewed`. |
    | `recall_at_k` | `hits / all_validation_positives`. |
    | `AP` | Average precision: how well positives are placed throughout the full ranking. |

    **Selection rule:** higher recall at 50 wins; higher AP breaks a recall tie. Because
    the images and budget are shared, higher recall here also means more hits and higher
    precision. An exact tie in both metrics is resolved by model name for reproducibility,
    not stronger evidence.

    AP can distinguish rankings with the same number of hits in their first 50 places.
    For the untied example `[1, 1, 0, 1, 0]`, average the precisions at the positive ranks:
    `(1/1 + 2/2 + 3/4) / 3 ≈ 0.917`. The helper uses scikit-learn's
    [AP calculation](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.average_precision_score.html),
    including tied scores. No additional AP coding is required.
    ''', "comparison-explanation"), md('''
    **Exercise (about 3 minutes):** replace the two `...` expressions with the fractions
    for precision and recall. `ranking_example` contains the full colour/texture ranking;
    the supplied lines select its first 50 rows and count the positives.

    Run your completed cell, then the comparison underneath. Check against its colour/texture
    row. If your output says `Ellipsis`, the corresponding `...` is still a placeholder.
    ''', "shortlist-exercise-instructions"), code('''
    shortlist = ranking_example.head(50)
    hits = int(shortlist["label"].sum())
    reviewed = len(shortlist)
    total_positives = int(validation.label.sum())

    precision = ...  # Fraction of the shortlist that is positive
    recall = ...     # Fraction of all validation positives included

    print("Precision:", precision)
    print("Recall:", recall)
    ''' if not instructor else '''
    shortlist = ranking_example.head(50)
    hits = int(shortlist["label"].sum())
    reviewed = len(shortlist)
    total_positives = int(validation.label.sum())

    precision = hits / reviewed  # Fraction of the shortlist that is positive
    recall = hits / total_positives     # Fraction of all validation positives included

    print("Precision:", precision)
    print("Recall:", recall)
    ''', "shortlist-metrics-exercise"), code('''
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
    display(results.round(3))

    for row in results.itertuples(index=False):
        print(f"{row.model}: precision = {row.hits}/{row.reviewed} = {row.precision_at_k:.1%}; "
              f"recall = {row.hits}/{validation_positives} = {row.recall_at_k:.1%}.")

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
    ### 3.5. Put the result in context

    The table needs two reference points:

    - **Random selection:** if we selected 50 validation images uniformly at random, each
      positive would have a `50 / number_of_validation_images` chance of being included.
      That fraction is the expected recall. The expected number of hits is that fraction
      times the positive count. This is an average over random selections, not a guarantee
      for one draw or a third trained model.
    - **Best possible at this budget:** a 50-image shortlist can contain at most 50 positives.
      If there are 60 positives in the pool, even a perfect ordering reaches only
      `50/60 ≈ 83.3%` recall. The remaining 10 cannot fit. That limit is the **recall ceiling**;
      it explains why 100% precision can coexist with less than 100% recall.

    The next cell calculates both references from the current validation set. They concern
    this sampled pool and budget, not the unknown rate of archaeological sites in a landscape.
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
    ### 3.6. Check the acquisition shortcut

    A deliberately limited reference predicts labels using **only original width and
    height**. It learns the positive fraction for each size from the training set, then
    scores validation images. It never examines pixels. Its accuracy measures how much
    the source collection process reveals about the labels; it is not a candidate model.

    The second table evaluates the image models within each original size. The
    350 × 350 subset contains both labels, so original dimensions alone cannot separate
    them there. Compare its AP with its positive fraction, the usual no-skill reference:
    this subset contains many more positives than negatives. AP is left undefined for
    the 64 × 64 subset because it has only negatives. Small subsets remain uncertain.

    Common resizing reduces the resolution shortcut but cannot prove it removed all
    acquisition cues. These diagnostics use validation only; test remains reserved.
    ''', "resolution-audit-explanation"), code('''
    validation_audit = workshop.validation_diagnostics(
        train, validation, validation_scores, k=REVIEW_BUDGET)
    size_probe = validation_audit.loc[validation_audit.model.eq("source_size_only")]
    print("Original-size-only reference (learned on training metadata):")
    display(size_probe[["model", "images", "accuracy", "precision_at_k", "recall_at_k", "AP"]].round(3))
    print("Image models after common preprocessing, grouped by original dimensions:")
    display(validation_audit.loc[validation_audit.model.ne("source_size_only"),
        ["model", "subset", "images", "positives", "positive_fraction", "AP"]].round(3))
    ''', "resolution-audit"), md('''
    **Checkpoint:** explain the selected model's result to a partner using its hit count,
    review budget and total positive count. Compare recall with random selection and the
    budget ceiling. If the hit counts tie, say what decided the choice.

    Keep `selected` for the following sections. Next we inspect actual mistakes, then
    ask whether the same fitted pipeline makes sense for different Cyprus images.
    ''', "model-question")]

    cells += [md('''
    ## 4. Inspect the shortlist

    Section 3 counted the selected model's hits. Now inspect what it included and left out
    at the **50-image budget**:

    - **False alarm:** a source-negative image (`0`) inside the shortlist.
    - **Miss:** a source-positive image (`1`) outside the shortlist.

    Ranks 1–50 are included; rank 51 onwards is excluded. A high-scoring positive at rank
    51 is still a miss at this budget. As the recall ceiling showed, some misses cannot
    be avoided when more positives exist than review slots.

    The next cell counts both groups and displays **up to four images per group**, highest
    score first, with ranks, scores and IDs. An empty group is reported explicitly;
    no false alarms does not mean no misses. These categories refer to the source labels,
    not an independent archaeological assessment.
    ''', "mistakes-introduction"), code('''
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
        display(examples[["rank", "tile_id", "label", "score"]].round(6))
        fig, axes = plt.subplots(1, 4, figsize=(12, 3.5), layout="constrained")
        for ax in axes:
            ax.axis("off")
        for ax, row in zip(axes, examples.itertuples(index=False)):
            ax.imshow(plt.imread(row.path))
            ax.set_title(f"Rank {row.rank} · score {row.score:.6f}\\n{row.tile_id}", fontsize=9)
        fig.suptitle(f"{selected} — {title}")
        plt.show()
    ''', "mistakes"), md('''
    **Keep one observation for your recommendation.** Choose a displayed tile; use a miss
    if there are no false alarms. Record its ID, rank and one visible feature. Distinguish
    that observation from a hypothesis about the score, and suggest one comparison that
    could test the hypothesis. Uncertainty is a valid observation.
    ''', "mistake-question"), md('''
    ### Exercise: what would ten more reviews recover?

    Predict whether ten extra slots will recover ten positives. Then change
    **`inspection_budget = 50` to `60`** and run the supplied code. It examines the next
    ten images in the same ranking, displays them in a table and shows any newly included
    source negatives.

    Use the output to explain the trade-off: extra positives recovered, extra negatives
    reviewed and positives still missed. Find one of the previously displayed missed tiles
    in the added rows if it was recovered.

    **Optional five-minute challenge:** find the smallest budget that includes every
    source positive. The comparison and report keep the original `REVIEW_BUDGET = 50`;
    this is a separate validation experiment.
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

    We chose a model using Enki validation labels. Now we ask a different question:
    **what happens when that fitted model receives images from another setting?**
    Applying a model elsewhere is called *transfer*. The selected classifier and its
    training scaler are reused unchanged; this cell does not fit a new Cyprus model.
    We compare both models' score distributions with Enki validation to expose changes
    in their behaviour. The selected model still determines the review order.

    ### 5.1. Understand the new images and records

    The bundle contains **12 Sentinel-2 images**, acquired on **5 August 2024**. Each covers
    a 1 km square at 10 m per pixel: a 100 × 100 pixel image. Enlarging the display does
    not add detail. Small features may be difficult to distinguish, and a catalogued
    settlement need not have the mound form associated with Enki's tell images.
    The common 64 × 64 model input is coarser still: about 15.6 m per input pixel here.
    Matching pixel dimensions does not match the geographic scale of the Enki crops.

    There are six pairs. Each pair contains an image centred on a catalogued settlement
    and one centred 2 km to its east, away from catalogue points. `paired_site` identifies
    the pair; `tile_id` identifies the individual image.

    | Value in `kind` | What it tells us | What it does not tell us |
    |---|---|---|
    | `catalogued_location` | The image is centred on a settlement recorded in the catalogue. | Whether remains are visible in this image on this date. |
    | `unlabelled_background` | The image is centred at a nearby location without a catalogue label. | Whether archaeology is absent. The word `background` in the filename is not a verified negative label. |

    Enki supplied image-class labels for counting hits and misses. These Cyprus records
    supply **locations and occupation periods**, not equivalent image labels. Several
    occupation rows can refer to one settlement. We can measure how well the ranking
    recovers catalogue locations. Section 5.3 calculates that agreement explicitly;
    it does not establish precision, recall or accuracy for visible archaeology.

    Run the next cell. It scores all 12 images with both models, compares their score
    ranges with Enki validation, then ranks by the selected model and shows the first
    four. The score is the model's response learned from Enki; it has not been validated
    as a probability of archaeology in Cyprus. Rank 1 means first among these 12 images.
    Sorting uses unrounded scores, even when displayed values look equal. Use this view
    to ask what should be investigated, rather than to count discovered sites.
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
    print("Source-trained scores: these are not calibrated Cyprus probabilities.")
    display(transfer_audit.round(6))
    cyprus["score"] = cyprus_scores[selected]
    cyprus = cyprus.sort_values("score", ascending=False, kind="stable").reset_index(drop=True)
    cyprus.insert(0, "rank", range(1, len(cyprus) + 1))
    display(cyprus[["rank", "tile_id", "kind", "paired_site", "score"]].round(6))
    workshop.contact_sheet(cyprus, cyprus.score.to_numpy(), n=4, show_labels=False,
                           title=f"Cyprus: four highest scores from {selected}")
    ''', "cyprus"), md('''
    **Read the distribution comparison first.** A shift towards zero or one for the
    whole Cyprus sample, or disagreement between the two models, calls the transfer
    assumptions into question. Neither direction establishes presence or absence.
    The shared preprocessing addresses pixel dimensions; target shape, geographic
    scale, rendering, land cover and label meaning still differ. Low scores should not
    be repaired by relabelling catalogue locations as visible positives or by fitting
    the scaler to these twelve images. A Cyprus model needs suitable local imagery,
    expert visibility labels and a separate geographic evaluation set.

    ### 5.2. Exercise: compare two nearby landscapes

    The next cell places the **catalogued location on the left** and its **nearby unlabelled
    location on the right**. Allow about five minutes for both pairs and your notes,
    then six to eight minutes for the calculation in section 5.3.

    1. Run `pair_id = "CY5"`. Compare scores and ranks, then describe one visible difference.
    2. Change only `pair_id` to `"CY479"` and rerun. Does the same kind of location rank first?
       If displayed scores are indistinguishable, say so.
    3. In the notes cell, name the tiles, record an observation and explain what evidence
       is missing before either ordering can be called a detection error.

    Keep this to two or three sentences for section 6. You do not need to identify remains
    from these images. Optional lab C later develops the missing-evidence question into
    a study design.
    ''', "cyprus-question"), code('''
    pair_id = "CY5"  # Then change this to "CY479" and rerun.
    pair = cyprus.loc[cyprus["paired_site"].eq(pair_id)].sort_values("kind")
    display(pair[["rank", "tile_id", "kind", "score"]].round(6))

    fig, axes = plt.subplots(1, 2, figsize=(8, 4), layout="constrained")
    for ax, row in zip(axes, pair.itertuples(index=False)):
        place = "Catalogued location" if row.kind == "catalogued_location" else "Unlabelled location"
        ax.imshow(plt.imread(row.path))
        ax.set_title(f"{row.tile_id}\\n{place}\\nRank {row.rank} · score {row.score:.6f}", fontsize=9)
        ax.axis("off")
    fig.suptitle(f"{selected}: pair {pair_id}")
    plt.show()
    ''', "cyprus-pair"), md('''
    **Your comparison:** double-click this Markdown cell and replace this text with
    your two or three sentences, then run the cell to display your notes. Include tile IDs,
    a score comparison, one image observation and the evidence needed to judge correctness.
    ''' if not instructor else '''
    **Worked interpretation:** read the current table for `CY5`, then run `CY479`.
    State which image ranks higher in each pair and quote the current scores.
    In the `CY5` pair, the catalogued image has larger bright angular patches, while the
    unlabelled image has winding pale lines through darker terrain; these are observations,
    not evidence that the classifier used those features.
    Neither ordering establishes a detection error because we lack independent labels
    for archaeological features visible in these particular images.

    Compare the two models' distributions above. Scores close to either `0` or `1`
    are responses of a source-trained classifier, not validated local probabilities.
    ''', "cyprus-pair-notes"), md('''
    ### 5.3. Calculate agreement with the Cyprus catalogue

    **Does the model rank catalogued locations ahead of nearby unlabelled locations?**
    Answer with counts and fractions, using all six pairs. This is a small catalogue
    agreement exercise; the missing visibility labels still prevent detection accuracy.

    We use a **four-image Cyprus review budget**, separate from Enki's 50 reviews.
    Run the next cell to see the six pair comparisons and the four-image shortlist.
    A pair is a **win** when the catalogued image scores higher, a **loss** when the
    unlabelled image scores higher, and a **tie** when their unrounded scores are equal.
    A tie gets half credit: a model assigning every image the same score achieves 50%.
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
    **Your calculation:** replace the three `...` expressions below.

    1. **Pair agreement:** `(wins + half the ties) / number of pairs`.
    2. **Catalogue fraction in the shortlist:** `catalogued locations retrieved / images reviewed`.
       This uses the same denominator as precision, but the target is catalogue membership.
    3. **Catalogue recovery:** `catalogued locations retrieved / all catalogued locations`.
       This uses the same denominator as recall.

    Use the named counts from the previous cell. Convert your fractions to percentages
    and predict whether they beat random selection before running the supplied check.
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
    **Check and interpret:** the next cell checks your arithmetic and compares both models.
    Random ordering averages **50% pair agreement**, **50% catalogue fraction** and
    **4/12 = 33.3% catalogue recovery** at four reviews. These are reference expectations,
    not guarantees for one draw. The code recalculates them if the budget changes.

    Exact ties use image-ID order for the shortlist. If a tie crosses its cutoff, the
    output also reports the range of possible catalogue counts from other tie orderings.
    Do not interpret an arbitrary tie order as evidence of better ranking.
    ''', "cyprus-check-explanation"), code('''
    cyprus_metric_rows, cyprus_pair_rows = [], []
    for name in models:
        pairs_for_model, metrics = workshop.catalogue_agreement(
            cyprus, score_column=f"{name}_score", k=CYPRUS_BUDGET)
        cyprus_metric_rows.append({"model": name, **metrics})
        cyprus_pair_rows.append(pairs_for_model.assign(model=name))
    cyprus_metrics = pd.DataFrame(cyprus_metric_rows)
    cyprus_pair_results = pd.concat(cyprus_pair_rows, ignore_index=True)
    display(cyprus_metrics[["model", "wins", "ties", "losses", "pair_agreement"]].round(3))
    display(cyprus_metrics[["model", "catalogued_at_k", "reviewed", "catalogued_total",
        "catalogue_fraction_at_k", "catalogue_recovery_at_k", "cutoff_tie"]].round(3))

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
    ''', "cyprus-metrics-check"), md('''
    **Record your conclusion:** state the three percentages with their counts and compare
    them with the random references. Does this support prioritising catalogue locations
    in this sample? Explain why it does not establish accuracy for visible archaeology.
    Six selected pairs are few, and nearby chips can overlap; they are not an independent
    geographic benchmark. Keep one Cyprus calculation for your final recommendation.
    ''' if not instructor else '''
    **Worked interpretation:** use the selected model's printed fractions above. Compare
    pair wins and ties as well as the total: identical agreement percentages can conceal
    very different numbers of ties. A better-than-random catalogue result supports a
    limited statement about these sampled locations. It does not validate visible-site
    detection, because unlabelled neighbours are not verified negatives and catalogue
    membership does not establish image visibility. Six pairs, with possible overlap
    between nearby chips, are insufficient for a geographic performance claim.
    ''', "cyprus-metrics-discussion")]

    cells += [md('''
    ## Optional lab — choose one before section 6

    **Main path:** if time is short, skip this whole block and go to section 6.
    The four-hour session reserves about 30 minutes here for **one lab or catch-up**,
    not all the extensions. All three choices use the work already completed.

    | Choice | Time | Deliverable |
    |---|---:|---|
    | A. Project the training features | 25–35 min | Complete three expressions; compare the 2D plots and optionally a 3D view. |
    | B. Is it worth labelling more training images? | 25–35 min | Run one cell, fill a two-row comparison table, and choose between 50 and 200 training images. |
    | C. What would a fair Cyprus test look like? | 25–30 min | Fill a five-answer plan, calculate two fractions for an invented example, and apply a supplied decision rule. |

    Optional code starts disabled so an uncompleted lab does not stop the main path.
    For lab A, complete its function before enabling `RUN_PROJECTION`. For lab B, set
    only `RUN_LEARNING_CURVES = True`. For lab C, complete the answer template and two
    arithmetic blanks, then set `RUN_CYPRUS_PRACTICE = True` in its supplied cell.
    Use training/validation evidence only. The original model comparison and the
    50-image review budget remain the report's reference result.
    ''', "optional-experiments"), md('''
    ### Lab A. See the training features in 2D or 3D

    **Question:** do the two source-label groups occupy different parts of feature space?
    Allow 25–35 minutes, including interpretation. You have already fitted and compared
    the classifiers; this lab visualises their inputs without changing them.

    An image has 68 colour/texture values or 384 DINOv2 values. **Principal component
    analysis (PCA)** constructs new axes from combinations of these features. PC1 follows
    the greatest variation; PC2 captures the most remaining variation at right angles
    to PC1. Keeping two or three components gives each image two or three coordinates.
    PCA uses feature values, not labels, to choose these directions.

    We standardise the **training** features first and fit a separate PCA for each
    representation. Every dot will be one of the same 200 training images. Colours and
    shapes add the source labels afterwards; validation and test images are excluded.

    **Fill the three `...` expressions below:**

    1. Select the training matrix for `name` from `feature_sets`.
    2. Use `scaler.fit_transform(...)` to scale that matrix.
    3. Use `pca.fit_transform(...)` to obtain the projected coordinates.

    The returned `explained_variance_ratio_` describes how much feature variation each
    axis retains; it is not classification accuracy. See the
    [PCA reference](https://scikit-learn.org/stable/modules/generated/sklearn.decomposition.PCA.html).

    Run the completed function cell, set `RUN_PROJECTION = True` and run the supplied
    plotting cell. Begin with `N_COMPONENTS = 2`. Until enabled, plotting prints a reminder.
    ''' if not instructor else '''
    ### Lab A. See the training features in 2D or 3D

    **Question:** do the two source-label groups occupy different parts of feature space?
    Allow 25–35 minutes, including interpretation. You have already fitted and compared
    the classifiers; this lab visualises their inputs without changing them.

    An image has 68 colour/texture values or 384 DINOv2 values. **Principal component
    analysis (PCA)** constructs new axes from combinations of these features. PC1 follows
    the greatest variation; PC2 captures the most remaining variation at right angles
    to PC1. Keeping two or three components gives each image two or three coordinates.
    PCA uses feature values, not labels, to choose these directions.

    We standardise the **training** features first and fit a separate PCA for each
    representation. Every dot will be one of the same 200 training images. Colours and
    shapes add the source labels afterwards; validation and test images are excluded.

    **Fill the three `...` expressions below:**

    1. Select the training matrix for `name` from `feature_sets`.
    2. Use `scaler.fit_transform(...)` to scale that matrix.
    3. Use `pca.fit_transform(...)` to obtain the projected coordinates.

    The returned `explained_variance_ratio_` describes how much feature variation each
    axis retains; it is not classification accuracy. See the
    [PCA reference](https://scikit-learn.org/stable/modules/generated/sklearn.decomposition.PCA.html).

    Run the completed function cell, set `RUN_PROJECTION = True` and run the supplied
    plotting cell. Begin with `N_COMPONENTS = 2`. Until enabled, plotting prints a reminder.

    **Instructor copy:** the function is completed and plotting is enabled below.
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

    if not RUN_PROJECTION:
        print("Complete the three blanks above, run that cell, then set RUN_PROJECTION = True here.")
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
    ''', "projection-plot"), md('''
    **Interpret the plots:** describe one region of overlap or separation and record
    the retained variance in each panel. Why could a classifier distinguish images that
    overlap here? Why is separation among training images not validation performance?

    Each panel has its own axes; positions cannot be compared directly between panels.
    PCA prioritises variation, which need not be the variation that separates the labels.
    The classifiers still use their full feature vectors.

    **If time permits:** set `N_COMPONENTS = 3`. Does PC3 reveal structure hidden in 2D?
    Inline views may be static: change `AZIMUTH` from `-60` to `30` and rerun to see another
    angle. The angle changes apparent overlap, not the underlying data.

    After this lab, continue to section 6. Lab B and lab C are alternatives, not prerequisites.
    ''', "projection-discussion"), md('''
    ### Lab B. Is it worth labelling more training images?

    **The question:** would giving the model more labelled examples help it find more
    tell-labelled images in the shortlist? You will run one supplied cell, read its
    table, and write a short answer. Allow 25–35 minutes. All labels already exist;
    the code pretends we have fewer of them by using part of the training set.

    **Keep these two numbers separate:**

    | Quantity | What happens in this experiment? |
    |---|---|
    | Images used to train a model | Changes: 20, 50, 100 or 200 in total. Half are tells and half are negatives. |
    | Images opened for review after ranking | Stays at 50 from the same validation pool. We count how many have the source label “tell”. |

    **Step 1 — predict.** Write down whether you expect 200 training images to find
    more tells than 20. A prediction can be wrong; you will compare it with the result.

    **Step 2 — run.** In the next cell, change only `RUN_LEARNING_CURVES = False` to
    `True`, then run it. The code trains both model types at each size. It repeats
    each size three times with different random choices of training images.

    **Step 3 — read the table.** “Mean tells found” is the average count over those
    three runs. “Fewest” and “Most” show the range. For example, counts of 40, 42 and
    41 would give a mean of 41 and a range of 40–42. The plot shows these same counts:
    training images along the bottom, tells found up the side. A higher point means
    more tells in the same 50 reviews. Complete the answer template underneath it.
    ''', "learning-curve-instructions"), code('''
    RUN_LEARNING_CURVES = False  # Change to True for lab B; the rest is supplied.

    if not RUN_LEARNING_CURVES:
        print("Lab B skipped. Choose one optional lab, or continue to section 6.")
    else:
        learning_rows = []
        for name in ("colour_texture", "dinov2"):
            part = workshop.learning_curve_rows(
                models[name], feature_sets[name]["train"], train.label.to_numpy(),
                feature_sets[name]["validation"], validation.label.to_numpy(),
                workshop.ranking_metrics, REVIEW_BUDGET,
            )
            part["model"] = name
            learning_rows.append(part)
        learning_results = pd.concat(learning_rows, ignore_index=True)
        learning_summary = learning_results.groupby(["model", "labels_per_class"]).agg(
            mean_recall=("recall_at_k", "mean"),
            min_recall=("recall_at_k", "min"),
            max_recall=("recall_at_k", "max"),
            mean_hits=("hits", "mean"), min_hits=("hits", "min"), max_hits=("hits", "max"),
        ).reset_index()
        learning_summary["training_images"] = 2 * learning_summary["labels_per_class"]
        learning_table = learning_summary[["model", "training_images", "mean_hits", "min_hits", "max_hits"]].rename(
            columns={"model": "Model", "training_images": "Training images (total)",
                     "mean_hits": "Mean tells found", "min_hits": "Fewest", "max_hits": "Most"})
        print(f"Every row uses the same {REVIEW_BUDGET} reviews from {len(validation)} validation images "
              f"containing {int(validation.label.sum())} source-labelled tells.")
        display(learning_table.round(1))

        fig, ax = plt.subplots(figsize=(8, 4), layout="constrained")
        for name, group in learning_summary.groupby("model"):
            spread = np.vstack([group.mean_hits - group.min_hits,
                                group.max_hits - group.mean_hits])
            ax.errorbar(group.training_images, group.mean_hits, yerr=spread,
                        marker="o", capsize=4, label=name)
        ceiling = min(REVIEW_BUDGET, int(validation.label.sum()))
        ax.set(xlabel="Images used for training (total)",
               ylabel=f"Tells found in {REVIEW_BUDGET} validation reviews", ylim=(0, ceiling + 2))
        ax.set_xticks([20, 50, 100, 200])
        ax.set_title("Does more training data put more tells in the shortlist?")
        ax.legend()
        plt.show()
    '''.replace("RUN_LEARNING_CURVES = False", f"RUN_LEARNING_CURVES = {instructor}"),
        "learning-curve"), md('''
    **Step 4 — fill in your answer below.** Double-click the next Markdown cell to edit it.

    1. For each model, copy “Mean tells found” at **20** and **200** training images.
       Subtract the first number from the second to find the gain. For example, 38 minus
       30 would mean eight additional tells found in the same 50 reviews.
    2. Find the **DINOv2 / 50 training images** row. Copy its mean, fewest and most counts.
       Compare its mean with the DINOv2 / 200 row.
    3. Choose **50 or 200 training images** for the next small experiment and justify it
       with those counts. Both choices can be argued; state what the extra 150 labels buy.

    **Finished when:** your two-row table and three short sentences are complete.
    Check whether your original prediction matched the result, then go to section 6.
    The three repeats vary the training examples on one validation set. They do not
    test three regions. At 200 images every repeat uses the same full training set,
    so its range may be zero. This lab gives evidence about this sample; a new area
    needs its own evaluation. The main selected model and test set are unchanged.
    ''', "learning-curve-discussion"), md('''
    **My Lab B answer — replace the blanks.**

    | Model | Mean found with 20 training images | Mean found with 200 | Gain: 200 minus 20 |
    |---|---:|---:|---:|
    | `colour_texture` | ... | ... | ... |
    | `dinov2` | ... | ... | ... |

    With 50 training images, DINOv2 finds ... tells on average, ranging from ... to ...;
    with 200 it finds ... on average.

    I would start the next small experiment with ... training images because ... .

    My prediction was ...; these results still need checking on ... .
    ''' if not instructor else '''
    **Worked Lab B answer:** the next cell calculates the comparison from this run.
    Discuss whether the measured change justifies labelling another 150 images.
    Accept either budget when its reason uses the actual counts and acknowledges that
    one validation sample cannot settle the choice for a new area.
    ''', "learning-curve-response")]
    if instructor:
        cells += [code('''
    if RUN_LEARNING_CURVES:
        comparison = learning_summary.pivot(index="model", columns="training_images", values="mean_hits")
        worked = pd.DataFrame({"Mean found with 20": comparison[20],
                               "Mean found with 200": comparison[200],
                               "Gain": comparison[200] - comparison[20]})
        display(worked.round(1))
        dino_rows = learning_summary.loc[learning_summary.model.eq("dinov2")].set_index("training_images")
        small, full = dino_rows.loc[50], dino_rows.loc[200]
        print(f"With 50 training images, DINOv2 finds {small.mean_hits:.1f} tells on average "
              f"(range {int(small.min_hits)}–{int(small.max_hits)}); with 200 it finds "
              f"{full.mean_hits:.1f}. The extra 150 labels change the mean by "
              f"{full.mean_hits - small.mean_hits:.1f} tells in the same {REVIEW_BUDGET} reviews.")
        if np.isclose(small.mean_hits, full.mean_hits):
            print("One supported choice is 50 images for the next small experiment: the observed means tie. "
                  "This is a starting point to test on another area, not a universal label requirement.")
        else:
            print("Use the measured gain above to argue whether another 150 labels are worthwhile. "
                  "The current evidence cannot give the benefit or labelling cost in a new area.")
    ''', "learning-curve-answer")]

    cells += [md('''
    ### Lab C. What would a fair Cyprus test look like?

    **The question:** what would we need to measure whether the model finds visible
    archaeology in Cyprus? You will complete a five-answer plan, then practise two
    calculations on an invented result. Allow 25–30 minutes. This is a planning and
    arithmetic exercise; all the information you need is below.

    **Step 1 — read the imagined situation.** A team wants to find visible settlement
    remains, such as wall or building-foundation patterns. Assume it can obtain new,
    consistently detailed imagery suitable for judging those features. The existing
    100 × 100 Cyprus chips and catalogue alone cannot supply that evidence.

    The team has three separate areas: **A** for training, **B** for choosing model
    settings, and **C** for the final check. It samples 100 non-overlapping images from
    across area C, including locations outside the catalogue. Two archaeologists can
    inspect them and label each image **visible**, **not visible**, or **uncertain**.
    They do this before seeing model scores. Area C is kept out of model development.

    **Step 2 — complete the next cell.** Write one sentence in each of its five rows.
    The hints refer to the situation above; you are explaining a concrete plan, not
    inventing a whole research project. “Not visible” means no visible target in that
    image, not proof that archaeology is absent underground.
    ''', "cyprus-study-design"), md('''
    **My Lab C plan — replace each blank with one sentence.**

    | Question | My answer |
    |---|---|
    | What visible feature would count as a positive? Name a feature, not just “a catalogue point”. | ... |
    | Who labels the images, and should they see the model scores first? | ... |
    | Should experts label all 100 area-C images or only the top 20? Explain how you would count missed positives. | ... |
    | Which area is reserved for the final check, and when may the team inspect its results? | ... |
    | What happens when the experts cannot tell whether the feature is visible? | ... |
    ''' if not instructor else '''
    **Worked Lab C plan — examples of acceptable answers.**

    | Question | Example answer |
    |---|---|
    | Visible target | A positive is an image in which experts can identify wall or foundation patterns consistent with settlement remains. |
    | Labelling | Two archaeologists judge the images before seeing scores, then discuss disagreements. |
    | Images to label | Label all 100 sampled images, including low-ranked and uncatalogued locations, so positives outside the shortlist can be counted. |
    | Final check | Keep area C separate while fitting on A and choosing settings on B; examine C after the choice is fixed. |
    | Uncertainty | Keep unresolved cases marked uncertain, report their number and arrange further review; never automatically turn them into negatives. |
    ''', "cyprus-study-response"), md('''
    **Step 3 — practise the decision using invented numbers.** These are **not results
    for our twelve Cyprus images**. For this example, assume every label was resolved:

    | Item | Count |
    |---|---:|
    | Images in the final area-C check | 100 |
    | Images experts labelled visible | 25 |
    | Highest-ranked images opened for review | 20 |
    | Visible positives among those 20 | 14 |

    The team agrees on this **practice rule before testing**: continue to a larger
    study only if **precision is at least 70% AND recall is at least 50%**. Otherwise,
    revise the approach. These thresholds are supplied for this exercise.

    Fill the two `...` expressions: precision divides the 14 hits by the **20 reviews**;
    recall divides the same 14 hits by the **25 available positives**. Use the variable
    names `practice_hits`, `practice_reviewed` and `practice_positives` so changing the
    hit count also updates your answers. Then set `RUN_CYPRUS_PRACTICE = True` and run
    the cell. The supplied code checks your arithmetic
    and applies the rule. Decide whether it should say “continue” before running it.
    ''', "cyprus-study-calculation-instructions"), code('''
    RUN_CYPRUS_PRACTICE = False  # Complete both expressions, then change to True.

    if not RUN_CYPRUS_PRACTICE:
        print("Lab C practice skipped. Complete its plan and two expressions if you choose this lab.")
    else:
        print("Invented practice data — not measured Cyprus performance.")
        practice_positives = 25
        practice_reviewed = 20
        practice_hits = 14  # After the first run, predict what happens with 10, then try it.

        practice_precision = ...  # Hits divided by images reviewed.
        practice_recall = ...     # Hits divided by all available positives.

        if practice_precision is Ellipsis or practice_recall is Ellipsis:
            print("Fill both expressions using the named counts, then rerun this cell.")
        else:
            expected_precision = practice_hits / practice_reviewed
            expected_recall = practice_hits / practice_positives
            if not np.allclose([practice_precision, practice_recall],
                               [expected_precision, expected_recall], rtol=0, atol=1e-9):
                print("Recheck the denominators: precision uses reviews; recall uses all positives.")
            else:
                print(f"Precision = {practice_hits}/{practice_reviewed} = {practice_precision:.0%}")
                print(f"Recall = {practice_hits}/{practice_positives} = {practice_recall:.0%}")
                passes = expected_precision >= 0.70 and expected_recall >= 0.50
                print("Decision: continue to a larger study." if passes else "Decision: revise the approach.")
    '''.replace("RUN_CYPRUS_PRACTICE = False", f"RUN_CYPRUS_PRACTICE = {instructor}")
       .replace("practice_precision = ...", "practice_precision = practice_hits / practice_reviewed" if instructor else "practice_precision = ...")
       .replace("practice_recall = ...", "practice_recall = practice_hits / practice_positives" if instructor else "practice_recall = ..."),
        "cyprus-study-practice"), md('''
    **Step 4 — change one number.** Predict what happens if the shortlist contains
    only **10** positives. Change `practice_hits = 14` to `10`, rerun the cell and check.

    **My conclusion — replace the blanks:** with 14 hits, precision is ... and recall
    is ..., so the practice rule says ... . With 10 hits, they become ... and ...,
    so the rule says ... .

    **Finished when:** the five plan answers and both decisions are written. Ask a
    partner whether your plan would count missed positives as well as successful finds.
    Use one next step from the plan in section 6. Keep these invented percentages in
    the practice exercise; section 6's Cyprus measurement comes from section 5.3.
    ''' if not instructor else '''
    **Worked conclusion:** 14 hits give precision `14/20 = 70%` and recall `14/25 = 56%`.
    Both meet the supplied rule, so the decision is to continue to a larger study.
    With 10 hits, precision is `10/20 = 50%` and recall is `10/25 = 40%`; the decision
    becomes “revise”. Meeting the practice rule would support more investigation,
    not deployment. These are invented counts, not measurements on the bundled Cyprus
    images. Use the five-row plan to discuss how the real evidence would be collected.
    ''', "cyprus-study-discussion")]

    cells += [md('''
    ## 6. Make a recommendation and export

    Bring together the measurement from section 3, an image observation from section 4
    or 5, and the limits of transfer. Finish any optional lab first. A recommendation
    against deployment is valid if the evidence supports it.

    ### 6.1. Exercise: write a 4–6 sentence recommendation

    Write between the triple quotes in the next cell. Use these five prompts:

    1. **Decision:** name the selected model and the use you recommend, with its scope.
    2. **Measurement:** cite hits, reviews and available positives, then precision and recall
       from section 3.4. Keep the original 50-image comparison; identify the larger-budget
       experiment separately if you mention it.
    3. **Observation:** cite one tile ID and a visible feature, separating observation
       from a possible explanation of its score.
    4. **Cyprus measurement and limit:** cite one catalogue-agreement calculation from
       section 5.3, then explain why it does not establish detection accuracy.
    5. **Next evidence:** propose suitable local imagery, expert labels and locations
       reserved for independent assessment. Use lab C's plan if you completed it.

    Run the cell to display your text. Exchange it with a partner, or reread it yourself:
    can the reader identify the proposed use, its evidence and a claim that still needs
    testing? Make one revision if needed, then export. This is the only report to submit;
    reuse your earlier notes.
    ''', "report-introduction"), code('''
    # Write your 4–6 sentence recommendation between the triple quotes.
    recommendation = """

    """.strip()
    print(recommendation or "Draft: write your recommendation above, then rerun this cell.")
    ''' if not instructor else '''
    winner = results.loc[results["model"].eq(selected)].iloc[0]
    recommendation = (
        f"Evaluate {selected} further as an aid to expert review of imagery similar to Enki. "
        f"On this validation sample it finds {int(winner.hits)} of {int(validation.label.sum())} "
        f"source positives in {int(winner.reviewed)} reviews "
        f"(recall {winner.recall_at_k:.3f}, precision {winner.precision_at_k:.3f}). "
        "Cyprus tile CY5_site shows bright angular patches in the displayed image; "
        "that observation does not establish which cues the classifier used. "
        f"In Cyprus, {int(cyprus_reference.catalogued_at_k)}/{int(cyprus_reference.catalogued_total)} "
        f"catalogue locations are recovered in {int(cyprus_reference.reviewed)} reviews "
        f"({cyprus_reference.catalogue_recovery_at_k:.1%}), but detection accuracy remains unknown "
        "because catalogue membership is not an image-visibility label. "
        "Common 64-pixel preprocessing reduces the source-size shortcut, but acquisition "
        "differences and geographic independence remain unresolved. "
        "Before field deployment, evaluate representative local imagery with expert visibility "
        "labels and a geographic holdout, using imagery appropriate to the size of the features sought."
    )
    print(recommendation)
    ''', "recommendation"), md('''
    ### 6.2. Save the choice and its evidence

    Run the next cell after finishing your recommendation and validation experiments.
    It uses the selected model to score the **Enki test images without using their labels**,
    then saves the ranking and your current evidence in `team_submission.zip`.

    | File in the ZIP | Purpose |
    |---|---|
    | `submission.csv` | Every test tile's ID, score and rank; no test labels. |
    | `configuration.json` | The selected model, settings, sample information and review budget. |
    | `validation_results.csv` | The two models' validation results used for selection. |
    | `cyprus_review.csv` | Cyprus scores and location metadata, not measured detection outcomes. |
    | `data_audit.json` | Duplicate audit, original size/label counts and common preprocessing. |
    | `validation_diagnostics.csv` | Size-only reference and validation AP within original dimensions. |
    | `transfer_diagnostics.csv` | Both models' score distributions on Enki validation and Cyprus. |
    | `cyprus_catalogue_metrics.csv` | Both models' catalogue agreement, shortlist recovery and random references. |
    | `cyprus_pairs.csv` | Scores, margins and outcomes for each pair and model. |
    | `recommendation.md` | Your written recommendation. |

    This records the current run; it does not lock the notebook or save a fitted model
    object. Rerunning export replaces the ZIP with the current results and text.
    An empty recommendation produces a draft: finish it and export again before sharing.
    Download the ZIP using the link or Colab's Files panel, and save the notebook so your
    exercise notes and image observations are preserved too.
    ''', "export-introduction"), code('''
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
    display(submission.head())
    display(FileLink(str(bundle)))
    print("Saved:", bundle, "— in Colab, download it from the Files panel.")
    ''', "export"), md('''
    ### 6.3. Final reveal — after sharing your recommendation

    Validation helped choose the model. The reserved **Enki test set** now assesses that
    saved choice; using its result to choose another model would turn it into development
    data. The Cyprus images are not part of this test.

    Keep `REVEAL_TEST = False` until the debrief. Then set it to `True` and run the next
    cell for the exported submission. It matches saved scores to the test source labels
    and uses the original review budget.

    Compare the test counts and recall with validation and random ordering. Which limitation
    from your recommendation still applies? A strong result does not establish geographic
    independence or Cyprus performance. Stop model experiments after this reveal.
    ''' if not instructor else '''
    ### 6.3. Final reveal — after sharing your recommendation

    Validation helped choose the model. The reserved **Enki test set** now assesses that
    saved choice; using its result to choose another model would turn it into development
    data. The Cyprus images are not part of this test.

    The participant copy leaves `REVEAL_TEST = False` until the debrief. This instructor
    copy sets it to `True` to display the worked result. The cell matches saved scores to
    the test source labels and uses the original review budget.

    Compare the test counts and recall with validation and random ordering. Which limitation
    from the recommendation still applies? A strong result does not establish geographic
    independence or Cyprus performance. Stop model experiments after this reveal.
    ''', "reveal-introduction"), code('''
    REVEAL_TEST = False
    if REVEAL_TEST:
        final_result = workshop.score_submission(submission, test, workshop.ranking_metrics, REVIEW_BUDGET)
        display(pd.Series(final_result, name="final_test").round(3).to_frame())
        print(f"Found {int(final_result['hits'])} of {int(test.label.sum())} source positives "
              f"in {int(final_result['reviewed'])} test-image reviews.")
        print(f"Random ordering: expected recall {min(REVIEW_BUDGET, len(test)) / len(test):.3f}.")
    else:
        print("Test labels remain unused. Finish and share your recommendation before the reveal.")
    ''' if not instructor else '''
    REVEAL_TEST = True
    if REVEAL_TEST:
        final_result = workshop.score_submission(submission, test, workshop.ranking_metrics, REVIEW_BUDGET)
        display(pd.Series(final_result, name="final_test").round(3).to_frame())
        print(f"Found {int(final_result['hits'])} of {int(test.label.sum())} source positives "
              f"in {int(final_result['reviewed'])} test-image reviews.")
        print(f"Random ordering: expected recall {min(REVIEW_BUDGET, len(test)) / len(test):.3f}.")
    else:
        print("Test labels remain unused. Finish and share your recommendation before the reveal.")
    ''', "reveal")]

    if instructor:
        cells += [md('''
    ### Instructor discussion notes

    - **Shortlist example:** this is arithmetic on supplied labels, not an image-review
      task. No images are displayed or judged here; the list is unrelated to the earlier
      training-image grid. At `k=2`, select `[1, 1]`: precision is 2/2 (100%) and recall
      is 2/3 (about 67%). At `k=4`, select `[1, 1, 0, 1]`: they are 3/4 (75%) and
      3/3 (100%). The extra entries are `0` and `1`, so recall rises while precision
      falls. The denominators refer to the selection size and the total positive count.
    - **Model comparison:** distinguish feature extraction from classifier fitting. The
      colour/texture vector has 68 measurements; the cached DINOv2 vector has 384 values.
      Each gets a separately fitted scaler and logistic regression using the same training
      images and labels. DINOv2 stays frozen. Validation labels enter the evaluation after
      prediction, not the classifier fit. Test results must not choose the representation.
    - **Optional lab A:** complete the participant function with the training matrix,
      `scaler.fit_transform(X)`, then `pca.fit_transform(X_scaled)`. Fit on training
      features only; labels supply plot colours, not PCA directions. Compare both
      representations in 2D and optionally 3D, including the variance retained.
      PCA axes differ between panels, and overlap can hide separation in omitted
      dimensions. Visual separation is not validation performance. The classifiers
      still use their full feature vectors; the projection is only for exploration.
    - **Optional lab B:** distinguish training-image counts from the fixed 50 reviews.
      Learners change one flag, copy counts into a two-row table, subtract the means
      at 20 and 200 images, and use the DINOv2 50/200 comparison to choose a starting
      budget. The instructor answer is calculated from the run. Bars show the range
      from three choices of training examples, not three independent regions.
    - **Optional lab C:** use the supplied A/B/C-area scenario and five-answer template.
      Then calculate precision and recall from the invented 14/20/25 counts and apply
      the supplied 70%/50% rule. Changing hits to 10 should reverse the decision.
      Keep this practice result separate from the real catalogue metrics in section 5.3.
    - **Short exercise:** precision is `hits / reviewed`; recall is
      `hits / total_positives`. Have learners calculate these from the ranking before
      displaying the comparison table, then check against its colour/texture row.
    - **Model choice:** use the actual counts printed in section 3. Higher recall at 50
      wins; AP describes the full ranking and breaks a recall tie. Equal hits mean neither
      shortlist finds more positives at this budget. The name breaks an exact metric tie
      without establishing a superior model. Compare against expected random recall and
      the budget ceiling. A strong baseline can reflect useful information or shortcuts;
      performance alone cannot tell us which.
    - **Mistakes:** use the displayed ranks to explain inclusion at the review budget.
      Zero false alarms can coexist with missed positives; do not invent an empty group's
      examples. Ask for a tile ID and a visible observation before accepting an explanation.
      The sample recommendation cites the displayed Cyprus pair.
    - **Resolution audit:** the original image sizes predict labels surprisingly well.
      Both models now use a common 64 × 64 input; the DINO cache was regenerated.
      Compare within-size AP with positive fraction. This is a diagnostic of remaining
      acquisition bias, not a geographic evaluation or proof that all shortcuts vanished.
    - **Budget experiment:** read the actual added positives, negatives and remaining
      misses after increasing the budget from 50 to 60. Find the last positive's rank
      for the optional all-positives question. This
      experiment does not change `REVIEW_BUDGET` or the reported model comparison.
    - **Cyprus pairs:** participants change one pair ID, from `CY5` to `CY479`, and compare
      the two landscapes. Use the current scores to describe the order. Ask for IDs and an observation,
      then the missing image-visibility labels. Neither ordering establishes an error.
      Compare both models' score distributions; extremes do not establish archaeological presence.
      For the extension, sample across scores and regions, retain uncertain judgements,
      and reserve separate locations for assessment. Catalogue absence is not a negative label.
    - **Cyprus calculation:** section 5.3 has three arithmetic blanks: pair agreement
      with half credit for ties, catalogue fraction in a four-image shortlist, and
      catalogue recovery. The instructor code supplies all three expressions; the next
      cell checks them and displays both models' results and random references. A tie
      at the shortlist cutoff is flagged with the range of possible catalogue counts.
      Interpret these as catalogue agreement, keeping detection accuracy unknown.
    - **Recommendation:** use five prompts: decision, measured evidence, image observation,
      limitation and next study. Accept a recommendation against deployment. Keep the
      original 50-image comparison distinct from the larger-budget experiment. The sample
      report uses six sentences and the currently selected model's validation row.
      Lab C supplies a concrete plan that can contribute one next step to the report.
    - **Export and debrief:** the ZIP records the current run; rerunning export replaces it.
      Share the recommendation before the reveal. The test cell scores the saved ranking
      against Enki source labels, not Cyprus labels. Discuss the comparison with validation
      and random ordering without selecting again on test performance. Neither a strong
      nor a weak test result resolves the missing geographic or Cyprus evidence.

    Choose only one optional lab at the block before section 6, or use that time for
    catch-up. The instructor copy shows the plots and worked Lab C practice. Model
    extraction and live imagery acquisition belong in preparation, outside class.
    ''', "instructor-notes")]

    cells += [md('''
    ### Data and credits

    - **Enki:** [Rosati (2025)](https://zenodo.org/records/14950565), a curated 600-image sample
      from `Normalized_Tell_combined_350.zip` (whole-archive MD5 `1d4ea398ea5669f9bca4539416847206`).
      Member CRCs and decoded-pixel hashes verify the partial download. Five RGB positive
      categories are used; masked and black-and-white folders are excluded.
      The source declares CC BY-SA 4.0 and separate Google/Bing imagery conditions; retain its notices.
    - **Cyprus catalogue:** [Crawford & Vella (2022)](https://doi.org/10.5334/joad.96), CC BY 4.0;
      [source revision](https://github.com/K-A-Crawford/Cyprus_Settlements/tree/70a5c3d399a2451e1d8dd5eb947eb80055276b44).
      There are 8,766 occupation rows and 1,559 settlement IDs.
    - **Image features:** [DINOv2](https://github.com/facebookresearch/dinov2),
      `facebook/dinov2-small`, revision `ed25f3a31f01632728cabb09d1542f84ab7b0056`.
      Enki and Cyprus use full-frame bicubic resizing to 64 × 64 before DINO's 224 × 224
      input, ImageNet normalization and L2 normalization. Both caches bind image IDs to
      decoded-pixel hashes, pinned weights and the preprocessing version.
    - **Cyprus imagery:** modified Copernicus Sentinel data (2024), via
      [Microsoft Planetary Computer](https://planetarycomputer.microsoft.com/dataset/sentinel-2-l2a).
      Scene `S2A_MSIL2A_20240805T082601_R021_T36SVD_20240805T144559`.
      The [Sentinel-2 band specifications](https://documentation.dataspace.copernicus.eu/APIs/SentinelHub/Data/S2L2A.html)
      give 10 m sampling for the visible bands used here.
      Nearby centres are 2 km east and at least 750 m from catalogue points; chips have at least
      90% clear-land pixels. Acquisition details are in `assets/cyprus/provenance.json`.

    **If a download fails:** use the facilitator's sample ZIP at the path given at the start;
    do not rename the full source archive to that filename. After restarting the kernel, rerun
    from the top. In Colab, download your outputs before the runtime ends.
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
