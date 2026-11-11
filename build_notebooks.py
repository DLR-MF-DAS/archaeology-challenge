"""Generate the two self-contained notebooks from shared, reviewable Python source."""
from pathlib import Path
import base64
import gzip
import io
import json
import textwrap
import zipfile
import nbformat as nbf

HERE = Path(__file__).resolve().parent


def code(source, **metadata):
    return nbf.v4.new_code_cell(textwrap.dedent(source).strip() + "\n", metadata=metadata)


def md(source):
    return nbf.v4.new_markdown_cell(textwrap.dedent(source).strip())


METRICS_SOLUTION = '''
def ranking_metrics(y_true, scores, k=50):
    y_true = np.asarray(y_true, dtype=int)
    scores = np.asarray(scores, dtype=float)
    if y_true.ndim != 1 or len(y_true) != len(scores) or len(y_true) == 0:
        raise ValueError("Use nonempty, aligned one-dimensional arrays.")
    if not set(np.unique(y_true)).issubset({0, 1}) or k < 1:
        raise ValueError("Binary labels and k >= 1 are required.")
    order = ranking_indices(scores)
    budget = min(int(k), len(y_true))
    hits = int(y_true[order[:budget]].sum())
    positives = int(y_true.sum())
    return {
        "AP": float(average_precision_score(y_true, scores)) if positives else float("nan"),
        "recall_at_k": hits / positives if positives else float("nan"),
        "precision_at_k": hits / budget,
        "hits": hits, "reviewed": budget,
    }
'''
METRICS_EXERCISE = '''
# Worked example: run this cell, then run the check below it.
# Try later: call ranking_metrics with a different k in a new cell.
''' + METRICS_SOLUTION

PROBE_SOLUTION = '''
def make_probe(C=1.0):
    # Fitting the pipeline on training data fits the scaler there, too.
    return make_pipeline(
        StandardScaler(),
        LogisticRegression(C=C, class_weight="balanced", max_iter=2000, random_state=SEED),
    )
'''
PROBE_EXERCISE = '''
# Worked example: this pipeline scales the training features and fits a classifier.
# Try later: compare a smaller and a larger C using validation data only.
''' + PROBE_SOLUTION

SEARCH_SOLUTION = '''
candidate_models = {}
validation_rows = []
for representation, matrices in feature_sets.items():
    for C in (0.01, 0.1, 1.0, 10.0):
        estimator = make_probe(C).fit(matrices["train"], y_train)
        scores = estimator.predict_proba(matrices["validation"])[:, 1]
        key = f"{representation}_C{C:g}"
        candidate_models[key] = (representation, estimator)
        validation_rows.append({"model": key, "representation": representation, "C": C,
                                **ranking_metrics(y_val, scores, REVIEW_BUDGET)})
validation_results = pd.DataFrame(validation_rows).sort_values(
    ["recall_at_k", "AP", "model"], ascending=[False, False, True]
).reset_index(drop=True)
best_key = validation_results.iloc[0]["model"]
best_representation, best_model = candidate_models[best_key]
display(validation_results)
print("Selected using validation only:", best_key)
'''
SEARCH_EXERCISE = '''
# Worked example: fit the candidates on training data and compare on validation.
# Try later: inspect validation_results and explain why the winning model was chosen.
''' + SEARCH_SOLUTION

CYPRUS_SOLUTION = '''
def summarise_sites(phases):
    frame = phases.copy()
    frame["LocQual"] = frame.LocQual.fillna("").astype(str).str.strip()
    for column in ("Long_Cord", "Lat_Cord", "Longitude", "Latitude"):
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
    # Do not invent a precise point for records with inconsistent coordinates.
    sites.loc[sites.coordinate_conflict, ["lon", "lat"]] = np.nan
    return sites
'''
CYPRUS_EXERCISE = '''
# Worked example: combine occupation rows into one record per settlement ID.
# Try later: inspect sites with quality_conflict or coordinate_conflict set to True.
''' + CYPRUS_SOLUTION


def embedded_assets():
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as bundle:
        bundle.write(HERE / ".work/sample_embeddings.npz", "enki_embeddings.npz")
        bundle.write(HERE / ".work/sample_manifest.json", "sample_manifest.json")
        bundle.write(HERE / ".work/data/Cyprus_Settlements.csv", "Cyprus_Settlements.csv")
        for path in sorted((HERE / ".work/cyprus").iterdir()):
            if path.suffix in {".png", ".csv", ".json", ".npz"}:
                # Only the 12 selected, paired chips belong in the bundle.
                if path.suffix == ".png":
                    import csv
                    with open(HERE / ".work/cyprus/manifest.csv") as f:
                        names = {r["filename"] for r in csv.DictReader(f)}
                    if path.name not in names:
                        continue
                bundle.write(path, "cyprus/" + path.name)
    return base64.b64encode(buf.getvalue()).decode()


def notebook(instructor, payload):
    role = "Instructor — worked solution" if instructor else "Participant challenge"
    cells = [md(f'''
    # Can AI help archaeologists decide where to look?
    **{role} · approximately 4 hours · Python / Google Colab**

    Archaeologists often have more imagery than they have time to inspect. A **tell** is a
    mound formed by repeated human occupation. In this workshop, you will use labelled image
    tiles from the Enki dataset to rank likely tells for an expert to review.

    **Your task:** imagine the team can inspect only **50 tiles**. Train a simple classifier
    with at most 100 labelled examples per class, then rank the remaining images. Compare two
    ways to describe each image: simple colour/texture measurements and features from a
    pretrained image model called DINOv2. Finally, explore what happens when that model is
    applied to a different landscape in Cyprus.

    **What you will deliver:** a ranked CSV and a short evidence-based recommendation, including
    what the model does well and what it cannot tell us. A high score is a review priority,
    not proof that a site exists.

    **By the end**, you should be able to explain precision and recall at a fixed review budget,
    select a model without using the final test, identify possible shortcuts in the images, and
    describe why the Cyprus examples cannot measure detection accuracy.

    ### How to use this notebook
    1. Open it in [Google Colab](https://colab.research.google.com/) (File → Upload notebook),
       or run it with a local Jupyter kernel. Run cells from top to bottom.
    2. Setup downloads a curated sample of 600 real images (about 73 MB); allow about 400 MB
       of free disk. It can take a few minutes. The default workflow runs on CPU.
    3. Read the short explanations and inspect the figures. {'The instructor exercises are solved and can be run in order.' if instructor else 'The exercises contain runnable examples: run them and their checks, then try the optional changes in the comments.'}
    4. In the final section, download the submission ZIP and save your completed notebook.

    **How to learn from this:** pause at each question and write or discuss an answer before
    opening the feedback. Run the supplied examples, then compare their output with your
    prediction. These checks are practice, not a grade. The GPU section is optional; leave its
    switch off. Complete the three response dictionaries before sharing your report.

    You need basic Python, pandas and scikit-learn familiarity; the diagrams explain the
    modelling and archaeological decisions. A GPU and API accounts are not required.
    ''')]
    if instructor:
        cells += [md('''
        ### Facilitation notes
        Share the participant notebook and reserve this copy for facilitation. Its saved outputs
        use the real sample; they illustrate one run, not a target score learners must match.

        **Before the session:** run the participant notebook in the intended environment, check
        that the sample downloads, and allow time for setup. The cached features make a GPU
        unnecessary. If connectivity is unreliable, provide `enki_workshop_sample.zip` for the
        `archaeology_workshop/` folder. Invite learners to work in pairs, especially if their
        Python or archaeology experience differs. The optional extensions can be skipped.

        **Four-hour guide** (adapt to the group):

        | Time | Focus | Facilitation move |
        |---|---|---|
        | 00:00–00:20 | Goal and setup | Ask what an archaeologist gains from a shortlist and what it cannot prove. |
        | 00:20–00:40 | Images and splits | Have pairs identify a possible visual shortcut before discussing duplicates. |
        | 00:40–01:05 | Review budget | Ask learners to predict hits, precision and recall before running the example. |
        | 01:05–01:40 | Features and model choice | Ask why validation, not test, is used to choose between candidates. |
        | 01:40–01:50 | Break | Pause; help anyone still setting up. |
        | 01:50–02:15 | Labels and mistakes | Compare the learning curve and examples; separate observation from hypothesis. |
        | 02:15–02:45 | Cyprus catalogue | Ask why multiple occupation rows do not mean multiple sites. |
        | 02:45–03:20 | Transfer | Ask what changed in the images and why target accuracy is unavailable. |
        | 03:20–03:40 | Recommendation | Ask teams to cite one result and one limitation before freezing their choice. |
        | 03:40–04:00 | Reveal and debrief | Compare reasoning and next-study proposals, not just the ranking score. |

        Give learners time to think independently before discussing in pairs; invite them to
        explain *why*, then provide feedback and revisit misconceptions. Treat the holdout as
        educational and honour-based: source labels are publicly recoverable. Similarity filtering
        does not prove independent sites or geographic generalisation.
        ''')]
    cells += [md('''
    ### Your route through the workshop

    | Stage | What you do | Why it matters |
    |---|---|---|
    | 1. Understand the images | Inspect labels and filter similar images | Spot shortcuts and reduce duplication |
    | 2. Build a shortlist | Use ranking metrics and fit a simple model | Measure what an expert can review |
    | 3. Compare models | Choose a model using validation data | Avoid choosing on the final test |
    | 4. Examine limits | Inspect mistakes and apply the model to Cyprus | Understand what might not transfer |
    | 5. Report | Freeze the model and export results | Make the recommendation reproducible |

    Enki labels come from the source archive; they are not independently verified site identities.
    The Cyprus catalogue describes settlement locations and occupation periods, not whether a
    site is visible in a modern image. Cyprus images are 1 km square at 10 m per pixel, from one
    satellite scene. They are for interpreting transfer, not measuring detection accuracy.
    ''')]
    cells += [md("### 1. Setup\nInstall only the small core dependencies. Existing Colab PyTorch is not replaced."),
              code('''
              import importlib.util, os, shutil, subprocess, sys
              
              def install_packages(*requirements):
                  if importlib.util.find_spec("pip") is not None:
                      command = [sys.executable, "-m", "pip", "install", "-q"]
                  elif shutil.which("uv"):
                      command = [shutil.which("uv"), "pip", "install", "--python", sys.executable, "-q"]
                  else:
                      raise RuntimeError("Install pip or uv for this Python environment, then rerun this cell.")
                  subprocess.check_call(command + list(requirements))
              
              if not os.environ.get("ARCHAEOLOGY_SKIP_INSTALL"):
                  install_packages("imagehash==4.3.2", "ipywidgets>=8,<9", "pillow>=10,<14", "scikit-learn>=1.4,<2",
                                   "pandas>=2.1,<4", "matplotlib>=3.8,<4", "requests>=2.31,<3", "tqdm>=4.66,<5")
              from IPython.display import display
              print("Core dependencies ready.")
              '''),
              code((HERE / "src/archaeology_challenge/runtime.py").read_text() +
                   "\nrandom.seed(SEED)\nnp.random.seed(SEED)\nROOT.mkdir(parents=True, exist_ok=True)\n",
                   tags=["infrastructure"],
                   **{"jupyter": {"source_hidden": True}, "colab": {"cellView": "form"}})]
    cells += [md('''
    The next cell unpacks the provided image features, Cyprus catalogue and 12 satellite images.
    **Run it; you do not need to edit it.** The Enki images themselves are downloaded next.
    Cached features let the workshop run without downloading or running the pretrained model.
    '''), code(f'''
    # Embedded reproducible workshop assets; no edits needed.
    ASSET_B64 = "{payload}"
    assets = ROOT / "assets"
    assets.mkdir(exist_ok=True)
    with zipfile.ZipFile(io.BytesIO(base64.b64decode(ASSET_B64))) as bundle:
        for member in bundle.infolist():
            target = (assets / member.filename).resolve()
            if not target.is_relative_to(assets.resolve()):
                raise ValueError("Unsafe bundle path")
        bundle.extractall(assets)
    del ASSET_B64
    print("Unpacked frozen embeddings, Cyprus catalogue and 12 Sentinel-2 chips.")
    ''', tags=["infrastructure"], **{"jupyter": {"source_hidden": True}, "colab": {"cellView": "form"}})]
    cells += [md('''
    ### 2. Meet the Enki images

    **Question:** what examples are we learning from, and can very similar images make a
    model look better than it is?

    The workshop uses 600 archive images: 300 source-labelled negative examples and 300
    positives from five landscape categories. A positive is labelled as a tell in the source;
    a negative comes from its negative folder. The sample is deliberately balanced for teaching,
    so its 50% positive rate is not an estimate of how common tells are in the world.

    Run the next two cells. They download and verify the selected images, group visually similar
    images, remove contradictory groups, and keep one image per group. Then they split the
    remaining images into training, validation and test sets and show some training images.
    Context categories are for inspection, not model inputs.

    **Why filter?** Near-duplicates across splits can give an overly optimistic result. Visual
    similarity checks miss some repeated sites, however, and do not prove geographic independence.
    These workshop splits are not the original authors’ experimental splits.
    ''')]
    cells += [code('''
    archive = download_enki_sample(assets / "sample_manifest.json", ROOT / "enki_workshop_sample.zip")
    all_images = index_enki(archive, ROOT / "images")
    clean_images, duplicate_audit, audit_report = deduplicate_images(all_images)
    display(pd.Series(audit_report, name="audit"))
    display(all_images.groupby(["label", "category"]).size().rename("images").to_frame())
    print("Image dimensions:", all_images[["width", "height"]].value_counts().to_dict())
    ''')]
    cells += [code('''
    splits = workshop_split(clean_images, per_class=100)
    train, validation, test = (splits[k] for k in ("train", "validation", "test"))
    REVIEW_BUDGET = 50
    display(pd.DataFrame({name: part.label.value_counts().sort_index()
                          for name, part in (("train", train), ("validation", validation))}).rename_axis("label"))
    contact_sheet(train.sample(12, random_state=SEED), title="Training examples: 1=tell; 0=background")
    # The remaining unused training-pool images are not extra labels for this challenge.
    print("Do not inspect test labels or test images until model selection is frozen.")
    ''')]
    cells += [md('''
    #### What is in this teaching sample?
    The left chart counts the **original sampled images** by archive context and label; these
    are source categories, not independent regions. The right chart shows what remains after
    similarity filtering and how the retained images are allocated. Unused images belong to
    the training pool but are withheld to enforce the label budget. The balanced sampling
    was designed for teaching and does not estimate the prevalence of tells in the landscape.
    '''), code('''
    counts = (all_images.groupby(["category", "label"]).size().unstack(fill_value=0)
              .reindex(columns=[0, 1], fill_value=0).sort_index())
    used = [len(train), len(validation), len(test)]
    allocation = used + [len(clean_images) - sum(used)]
    fig, (ax_context, ax_split) = plt.subplots(1, 2, figsize=(13, 5),
                                               gridspec_kw={"width_ratios": [1.35, 1]})
    positions = np.arange(len(counts))
    ax_context.barh(positions, counts[0], color="#4477aa", label="Source negative (0)")
    ax_context.barh(positions, counts[1], left=counts[0], color="#ee7733", label="Source positive (1)")
    ax_context.set_yticks(positions, labels=counts.index.astype(str))
    ax_context.invert_yaxis()
    ax_context.set(xlabel="Images before filtering", title="Archive context in the 600-image sample")
    ax_context.legend(loc="lower right", fontsize=8)
    bars = ax_split.barh(["Train labels", "Validation", "Test", "Unused train pool"], allocation,
                         color=["#4477aa", "#66c2a5", "#aa4499", "#bbbbbb"])
    ax_split.invert_yaxis()
    ax_split.bar_label(bars, padding=3)
    ax_split.set(xlabel="Retained images", title=f"Allocation after filtering (n={len(clean_images)})")
    ax_split.margins(x=0.16)
    fig.tight_layout()
    plt.show()
    ''', tags=["visual-explanation"])]
    cells += [md('''
    **Discuss:** What visual evidence distinguishes the classes? Could rendering, masking or
    land-cover context be a shortcut? Why does duplicate removal not establish geographic transfer?
    Raw accuracy is a poor description of a ranking system with a limited review budget.
    ''')]
    if instructor:
        cells += [md('''
        **Worked interpretation:** several displayed source-negative training tiles are visibly
        pixelated, while several positive tiles show finer detail. This is an observed rendering
        shortcut worth testing, not proof of what the encoder uses. Look for class-correlated
        saturation and land-cover composition too. Source subfolder names describe context categories;
        they are not independent surveys or verified geographic regions. Good holdout results
        may still depend on similar landscapes or unrecognised repeated sites. Neither an image
        hash nor a random split establishes that the model generalises to a new country.
        ''')]
    cells += [md('''
    **Why keep three partitions?** Training fits candidate models; validation compares them;
    the untouched test set provides one final assessment after the choice is frozen. Similarity
    filtering reduces image duplication but does not make these independent geographic regions.
    '''), code('''
    fig, ax = plt.subplots(figsize=(10, 2.8))
    roles = [
        ("TRAIN", "Fit candidate models", "#e1eff0"),
        ("VALIDATION", "Compare and select", "#f8edda"),
        ("TEST", "Assess the frozen choice", "#efe8f1"),
    ]
    for i, (name, role, colour) in enumerate(roles):
        x = .04 + i * .34
        ax.add_patch(plt.Rectangle((x, .25), .26, .5, transform=ax.transAxes,
                                   facecolor=colour, edgecolor="none"))
        ax.text(x + .13, .58, name, transform=ax.transAxes, ha="center", weight="bold")
        ax.text(x + .13, .40, role, transform=ax.transAxes, ha="center", fontsize=10)
        if i < 2:
            ax.annotate("", xy=(x + .33, .5), xytext=(x + .27, .5),
                        xycoords="axes fraction", arrowprops={"arrowstyle": "->", "lw": 1.5})
    ax.set_title("Fit → select → freeze → assess")
    ax.axis("off")
    fig.tight_layout()
    plt.show()
    ''', tags=["visual-explanation"])]
    cells += [md('''
    ### 3. Exercise 1 — measure a useful shortlist

    **Question:** if an expert can inspect only `k` tiles, how many known positives reach the
    shortlist? The picture below uses an invented ranking to show the idea.

    Run the example `ranking_metrics(y_true, scores, k)` below. It returns:
    - `hits`: number of positive tiles among the top `k` scores.
    - `reviewed`: number actually reviewed, `min(k, number_of_tiles)`.
    - `recall_at_k`: hits divided by *all* positive tiles.
    - `precision_at_k`: hits divided by tiles reviewed.
    - `AP`: average precision, a measure over the complete ranking.

    `y_true` contains source labels (1 or 0); `scores` are model priorities. Higher scores rank
    first. If there are no positives, return `np.nan` for AP and recall. Use the provided stable
    ordering helper for ties. Run the check cell, then experiment with the optional change in the comments.
    '''), code('''
    # A schematic, not a result from the Enki test set.
    from matplotlib.patches import Patch
    example_labels = [1, 0, 1, 0, 0, 1, 0, 1, 0, 0]
    example_budget = 4
    fig, ax = plt.subplots(figsize=(11, 2.8))
    colors = ["#ee7733" if label else "#4477aa" for label in example_labels]
    ax.bar(np.arange(1, 11), np.ones(10), color=colors, edgecolor="white", width=0.85)
    ax.axvline(example_budget + 0.5, color="#222222", linestyle="--", linewidth=2)
    ax.text(2.5, 1.08, "Reviewed first", ha="center", fontsize=10)
    ax.text(7.5, 1.08, "Not reviewed", ha="center", fontsize=10)
    ax.set_xticks(np.arange(1, 11))
    ax.set(xlabel="Rank (highest model score on the left)", yticks=[],
           title="A fixed review budget turns scores into a shortlist")
    ax.legend(handles=[Patch(color="#ee7733", label="Source positive"),
                       Patch(color="#4477aa", label="Source negative")],
              loc="center left", bbox_to_anchor=(1.01, 0.5), frameon=False)
    ax.set_ylim(0, 1.28)
    fig.tight_layout()
    plt.show()
    ''', tags=["visual-explanation"]), md('''
    In this illustration, each rectangle is a tile and the dashed line is the review limit.
    Precision asks what fraction of reviewed tiles are positive; recall asks what fraction of
    all positives were reviewed. The colors represent known labels for evaluation, not information
    available to a reviewer before labels are known. Average precision considers the full ranking.
    '''), md('''
    **Pause and predict:** A ranked list has labels `[1, 0, 1, 1, 0]`, in that order.
    If the team reviews the first two tiles, what are the precision and recall?
    Work it out before opening the explanation.

    <details><summary>Check your reasoning</summary>

    There is one hit in two reviewed tiles, so precision is 1/2. There are three positives
    in the full list, so recall is 1/3. The denominators answer different questions: precision
    describes the shortlist, while recall describes coverage of all known positives.

    </details>
    '''), code(METRICS_SOLUTION if instructor else METRICS_EXERCISE, tags=["exercise-1"]),
              code('''
              toy = ranking_metrics(np.array([1, 0, 1, 0]), np.array([.9, .8, .7, .1]), k=2)
              assert np.isclose(toy["recall_at_k"], .5)
              assert np.isclose(toy["precision_at_k"], .5)
              assert np.isclose(toy["AP"], 5 / 6)
              assert toy["hits"] == 1 and toy["reviewed"] == 2
              assert ranking_metrics([1, 0], [.2, .1], k=10)["reviewed"] == 2
              assert np.isnan(ranking_metrics([0, 0], [.9, .1], k=1)["recall_at_k"])
              print("Metric checks passed.")
              ''')]
    cells += [md('''
    ### 4. Describe each image in two ways

    A classifier needs numbers, not image files. We compare two sets of numbers for the same tiles:

    - **Colour and texture:** simple summaries of colours, brightness and edges. This is our baseline.
    - **DINOv2:** a pretrained general-purpose image model turns each tile into a 384-number
      description called an *embedding*. Its encoder stays frozen: it is not trained on our labels.

    Run the next cell to calculate the baseline and load the provided DINOv2 features. Comparing
    both on the same splits helps us ask whether the pretrained representation adds value here.
    DINOv2 is not specialised for Earth observation. The same DINOv2 preprocessing is used
    for Enki and Cyprus.
    '''), code('''
    X_handcrafted = handcrafted_features(clean_images.path.tolist())
    X_dino = load_embedding_cache(clean_images, assets / "enki_embeddings.npz")
    feature_sets = {
        "colour_texture": partition_features(clean_images, X_handcrafted, splits),
        "dinov2": partition_features(clean_images, X_dino, splits),
    }
    y_train = train.label.to_numpy()
    y_val = validation.label.to_numpy()
    print("Feature shapes:", X_handcrafted.shape, X_dino.shape)
    print("Pinned model:", MODEL_ID, MODEL_REVISION)
    '''), md('''
    **Optional:** the next cell normally reuses the cached features and needs no GPU. Leave
    `RECOMPUTE_DINO = False` for the main workshop. To reproduce the features yourself, set it to
    `True`; this downloads model weights (~90 MB) and may take time on a CPU. In Colab you can
    select a T4 GPU if one is available. Do this before fitting models.
    '''), code('''
    RECOMPUTE_DINO = False
    if RECOMPUTE_DINO:
        install_packages("transformers==4.57.6")
        fresh = extract_dino(clean_images.path.tolist())
        print("Mean cosine agreement with cache:", np.mean(np.sum(fresh * X_dino, axis=1)))
        X_dino = fresh
        feature_sets["dinov2"] = partition_features(clean_images, X_dino, splits)
    else:
        print("Using verified cached embeddings; no GPU required.")
    ''')]
    cells += [md('''
    ### 5. Exercise 2 — build a simple classifier

    **Question:** can a small model learn to score tiles using their features?

    The `make_probe(C)` example returns a scikit-learn pipeline with `StandardScaler` followed by
    `LogisticRegression`. The scaler learns feature ranges from the training data inside the
    pipeline, so validation data cannot influence fitting. `C` controls regularisation: smaller
    values constrain the classifier more strongly. Run the check to fit a colour/texture baseline
    and display its validation metrics.
    '''), md('''
    **Pause and explain:** Why does the pipeline learn its scaling from the training tiles,
    instead of calculating the scaling using training and validation tiles together?

    <details><summary>Check your reasoning</summary>

    Validation data is meant to simulate unseen examples. Using it to calculate feature
    scaling would leak information into training and make that comparison less independent.
    The pipeline fits the scaler anew using only the training part of each fit.

    </details>
    '''), code(PROBE_SOLUTION if instructor else PROBE_EXERCISE, tags=["exercise-2"]),
              code('''
              baseline = make_probe(C=1.0).fit(feature_sets["colour_texture"]["train"], y_train)
              baseline_val = baseline.predict_proba(feature_sets["colour_texture"]["validation"])[:, 1]
              display(pd.Series(ranking_metrics(y_val, baseline_val, REVIEW_BUDGET), name="baseline"))
              assert len(baseline.steps) == 2
              assert not hasattr(make_probe().steps[0][1], "mean_")
              ''')]
    cells += [md('''
    ### 6. Exercise 3 — choose a model fairly

    The next cell compares both feature sets and four `C` values: `0.01`, `0.1`, `1`, `10`. It fits each candidate
    on training data and scores it on validation data. It chooses the highest validation recall at
    the 50-tile budget, using AP to break ties. This gives eight candidates.

    **Why validation?** Choosing with the test set would make the final assessment less independent.
    Keep the test set untouched until your choice is frozen. Report the baseline if it wins.
    Run the following check to see the comparison and precision–recall curves.
    '''), md('''
    **Try a selection:** Model A has validation recall 0.40 and AP 0.90. Model B has
    validation recall 0.50 and AP 0.60. Which would the workshop rule select, and why?

    <details><summary>Check your reasoning</summary>

    Choose B: recall at the review budget is the primary criterion, and 0.50 exceeds 0.40.
    AP breaks a tie in recall; it does not override a higher recall under this rule. The
    rule is a workshop choice, not a universal measure of a model's usefulness.

    </details>
    '''), code(SEARCH_SOLUTION if instructor else SEARCH_EXERCISE, tags=["exercise-3"]),
              code('''
              assert len(validation_results) == 8
              assert set(validation_results.representation) == {"colour_texture", "dinov2"}
              assert best_key in candidate_models
              validation_results.to_csv(ROOT / "validation_results.csv", index=False)
              for representation in feature_sets:
                  key = validation_results.loc[validation_results.representation == representation, "model"].iloc[0]
                  fitted = candidate_models[key][1]
                  scores = fitted.predict_proba(feature_sets[representation]["validation"])[:, 1]
                  PrecisionRecallDisplay.from_predictions(y_val, scores, name=key, ax=plt.gca())
              plt.title("Validation precision–recall curves")
              plt.show()
              ''')]
    if instructor:
        cells += [md('''
        **Worked interpretation:** evaluate the measured improvement, not the model's reputation.
        A bigger AP does not guarantee more positives within the first 50 reviews. A large gap
        between conventional and pretrained features supports using the representation on this
        split; it does not establish general archaeological detection or superiority across regions.
        Eight validation comparisons already create selection optimism. Keep the final test for one
        frozen evaluation, and keep any future comparisons on new development splits.
        ''')]
    cells += [md('''
    ### 7. How many labels help, and where does the model struggle?

    The next cell fits the DINOv2 classifier with 10, 25, 50 and 100 labelled training examples
    per class. It repeats each selection three times and plots validation recall at the review
    budget. A rising curve suggests more labels helped; a flat curve suggests diminishing returns.
    The variation reflects these three training samples, not independent geographic regions.

    The following cell shows high-scoring source negatives and low-scoring source positives.
    Look for landscape patterns, rendering or other cues that could explain the scores; images
    alone cannot prove why the model made a decision. These examples are not necessarily
    mistakes at the 50-tile cutoff, and source labels are not independent ground truth.
    '''), code('''
    best_dino_key = validation_results.loc[validation_results.representation == "dinov2", "model"].iloc[0]
    dino_probe = candidate_models[best_dino_key][1]
    curve = learning_curve_rows(dino_probe, feature_sets["dinov2"]["train"], y_train,
                               feature_sets["dinov2"]["validation"], y_val,
                               ranking_metrics, REVIEW_BUDGET)
    display(curve.groupby("labels_per_class")[["AP", "recall_at_k"]].agg(["mean", "std"]))
    summary = curve.groupby("labels_per_class").recall_at_k.agg(["mean", "std"])
    plt.errorbar(summary.index * 2, summary["mean"], yerr=summary["std"], marker="o", capsize=4)
    plt.xlabel("Total labelled training tiles")
    plt.ylabel("Validation recall at review budget")
    plt.ylim(0, 1)
    plt.title("Frozen DINOv2: label efficiency (three training samples)")
    plt.show()
    curve.to_csv(ROOT / "learning_curve.csv", index=False)
    '''), code('''
    dino_val_scores = dino_probe.predict_proba(feature_sets["dinov2"]["validation"])[:, 1]
    failures = validation.copy()
    failures["score"] = dino_val_scores
    high_negatives = failures.loc[failures.label == 0].sort_values("score", ascending=False)
    low_positives = failures.loc[failures.label == 1].sort_values("score")
    contact_sheet(high_negatives, high_negatives.score.to_numpy(), n=8, title="Highest-scoring labelled backgrounds")
    contact_sheet(low_positives, low_positives.score.to_numpy(), n=8, title="Lowest-scoring labelled positives")
    # Inspect model neighbours for one difficult validation image.
    query_index = int(np.argmax(np.where(y_val == 0, dino_val_scores, -np.inf)))
    neighbours, similarity = nearest_examples(feature_sets["dinov2"]["validation"][query_index],
                                              feature_sets["dinov2"]["train"], train)
    contact_sheet(neighbours, similarity, n=4, title="Training neighbours of the highest-scoring source negative")
    ''')]
    cells += [md('''
    **Your interpretation:** use the images to suggest two reasons a source-negative tile might
    rank highly and one reason a source-positive tile might rank poorly. Fill in the dictionary
    below and note what additional evidence would help an archaeologist check your ideas.
    '''), code('''
    error_notes = {
        "high_scoring_negative_1": "",
        "high_scoring_negative_2": "",
        "low_scoring_positive": "",
        "next_evidence_needed": "",
    }
    ''')]
    if instructor:
        cells[-1] = code('''
        error_notes = {
            "high_scoring_negative_1": "Validation tile 2c58b208b shows field boundaries and a road junction; it receives a high score despite its source-negative label. Linear landscape structure is a plausible confounder.",
            "high_scoring_negative_2": "Tile c77ec1cba contains a rectilinear built or disturbed area. Its high source-negative score suggests that geometry and exposed-soil context merit inspection; this is not a causal attribution.",
            "low_scoring_positive": "The lowest-ranked source-positive cec8faef9 shows muted, disturbed-looking terrain. It may fall outside a limited shortlist even with a score above 0.5; shortlist misses are not the same as thresholded false negatives.",
            "next_evidence_needed": "Expert image review, source provenance, additional dates and finer-resolution imagery; field evidence for archaeological confirmation.",
        }
        # These visual interpretations refer to the pinned validation contact sheet.
        # They are not new archaeological ground-truth annotations or causal explanations.
        ''')
    cells += [md('''
    ### 8. Exercise 4 — understand the Cyprus catalogue

    **Question:** how many distinct settlements are represented, and how reliable are their
    locations? Each catalogue row is an occupation phase, so a settlement can appear multiple
    times. The next cell loads the catalogue. Run `summarise_sites(phases)` to return one row
    per settlement ID and run the check to plot the sites.

    Use `Long_Cord` and `Lat_Cord` for longitude and latitude in degrees. Despite their names,
    `Longitude` and `Latitude` are UTM zone 36N coordinates in metres. For each ID, the function counts
    occupation rows and distinct periods, checks location quality and flags conflicting coordinates.
    When coordinates conflict, keep the ID but leave its summary longitude and latitude missing.

    The [field dictionary](https://github.com/K-A-Crawford/Cyprus_Settlements/blob/70a5c3d399a2451e1d8dd5eb947eb80055276b44/settlement_key.txt)
    defines quality A–E by the source used to locate a settlement (from supplied coordinates to
    modern place names). Keep other or blank codes and flag them as undocumented. These codes
    are provenance categories, not numerical error radii.
    '''), code('''
    phases = pd.read_csv(assets / "Cyprus_Settlements.csv")
    print("Occupation rows:", len(phases), "Unique settlement IDs:", phases.ID.nunique())
    display(phases[["ID", "Period", "Longitude", "Latitude", "Long_Cord", "Lat_Cord", "LocQual"]].head())
    '''), md('''
    **Try a small catalogue:** Suppose three occupation rows have IDs `A`, `A`, and `B`.
    How many settlement records should a one-row-per-ID summary contain? What should you
    do if the two rows for `A` report conflicting coordinates?

    <details><summary>Check your reasoning</summary>

    There are two settlement IDs, not three. Keep one row for each ID and flag the
    coordinate conflict for `A`; leave its summary coordinates missing rather than silently
    selecting a point. A phase is a record about a settlement, not a separate settlement.

    </details>
    '''), code(CYPRUS_SOLUTION if instructor else CYPRUS_EXERCISE, tags=["exercise-4"]),
              code('''
              sites = summarise_sites(phases)
              assert len(sites) == phases.ID.nunique() == 1559
              assert sites.ID.is_unique
              assert sites.occupation_rows.sum() == len(phases) == 8766
              assert sites.lon.dropna().between(31, 35).all() and sites.lat.dropna().between(34, 36).all()
              assert sites.loc[sites.coordinate_conflict, ["lon", "lat"]].isna().all().all()
              assert sites.loc[sites.coordinate_conflict, "ID"].tolist() == ["CY453"]
              display(sites.groupby(["LocQual", "quality_documented"]).size().rename("sites").to_frame())
              print("Undocumented/blank values:", sorted(sites.loc[~sites.quality_documented, "LocQual"].unique()))
              display(sites.loc[sites.coordinate_conflict | sites.quality_conflict])
              for quality, group in sites.groupby("LocQual"):
                  plt.scatter(group.lon, group.lat, s=8, alpha=.6, label=quality or "blank")
              plt.xlabel("Longitude (degrees)")
              plt.ylabel("Latitude (degrees)")
              plt.title("Cyprus catalogue: one point per settlement")
              plt.legend(title="Location quality", fontsize=8)
              plt.gca().set_aspect(1 / np.cos(np.deg2rad(35)))
              plt.show()
              ''')]
    if instructor:
        cells += [md('''
        **Worked interpretation:** the pinned source has 8,766 occupation rows and 1,559 settlement
        IDs. Besides A–E, it contains blank and F values, which the supplied dictionary does not
        define. Do not silently interpret F as a quality rank. `CY453` has two distinct coordinate
        pairs, so its summary retains the ID and flags the conflict without choosing a point.
        Multiple IDs may also share a location,
        so deduplicating coordinates alone could erase distinct records. Survey intensity and
        historical coverage affect the observed spatial distribution. No catalogue entry does not
        mean no archaeology, and known occupation does not imply visibility in today's imagery.
        ''')]
    cells += [md('''
    ### 9. What happens in Cyprus?

    **Question:** does a model trained on Enki give meaningful priorities in a different place
    and type of imagery? We apply the Enki-trained DINOv2 classifier unchanged, even if the
    baseline won the source comparison, to inspect how that representation transfers.

    The sample contains 12 real Sentinel-2 chips from one day in southwest Cyprus: six centred
    on catalogue locations and six nearby **unlabelled** locations. Each chip covers 1 km × 1 km
    at 10 m per pixel. A catalogue entry does not guarantee that archaeology is visible in the
    image; an unlabelled location is not a confirmed absence. Therefore these chips cannot give
    us a valid detection-accuracy score.

    Background centres were placed 2 km east and checked to be at least 750 m from catalogue
    points. Images with less than 90% clear-land pixels were excluded. A larger, independently
    labelled geographic study would be needed to measure transfer.
    '''), md('''
    **What changes during transfer?** The model is carried across domains unchanged, but the
    imagery and meaning of available labels differ. A catalogue location is not an image-visibility
    label, and a nearby background location is not a verified absence. The diagram shows why these
    twelve chips support an interpretive review rather than an accuracy estimate.
    '''), code('''
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.5))
    cards = [
        ("Source: Enki", "Rendered RGB tiles\\nSource tell / negative labels\\nCurated teaching split", "#e1eff0"),
        ("Same model", "Frozen DINOv2 features\\nEnki-trained linear probe\\nScores are review priorities", "#f8edda"),
        ("Target: Cyprus", "Sentinel-2: 10 m pixels\\nCatalogue and unlabelled locations\\nNo visibility ground truth", "#efe8f1"),
    ]
    for ax, (title, details, colour) in zip(axes, cards):
        ax.set_facecolor(colour)
        ax.text(.5, .78, title, ha="center", va="center", weight="bold", fontsize=12,
                transform=ax.transAxes)
        ax.text(.5, .43, details, ha="center", va="center", fontsize=10,
                linespacing=1.8, transform=ax.transAxes)
        ax.set_xticks([]); ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_visible(False)
    fig.suptitle("Transfer changes the evidence, not just the location", fontsize=13)
    fig.tight_layout()
    plt.show()
    ''', tags=["visual-explanation"]), code('''
    cyprus_dir = assets / "cyprus"
    cyprus = pd.read_csv(cyprus_dir / "manifest.csv")
    cyprus["path"] = [str(cyprus_dir / f) for f in cyprus.filename]
    with np.load(cyprus_dir / "embeddings.npz", allow_pickle=False) as cache:
        assert np.array_equal(cache["tile_ids"], cyprus.tile_id.to_numpy())
        assert str(cache["revision"]) == MODEL_REVISION
        assert str(cache["preprocessing"]) == PREPROCESS_ID
        cyprus_vectors = cache["embeddings"].copy()
    cyprus["score"] = dino_probe.predict_proba(cyprus_vectors)[:, 1]
    display(cyprus[["tile_id", "kind", "LocQual", "gsd_m", "clear_land_fraction", "score"]].sort_values("score", ascending=False))
    ranked_cyprus = cyprus.sort_values("score", ascending=False)
    contact_sheet(ranked_cyprus, ranked_cyprus.score.to_numpy(), n=12,
                  title="Cyprus: transferred Enki scores; no archaeological visibility labels", show_labels=False)
    display(json.loads((cyprus_dir / "provenance.json").read_text()))
    cyprus.to_csv(ROOT / "cyprus_review.csv", index=False)
    ''')]
    cells += [md('''
    **Pause and predict:** A high-scoring Cyprus background tile has no catalogue entry.
    Is that enough evidence to count it as a false alarm? Explain your reasoning.

    <details><summary>Check your reasoning</summary>

    No. The background location is unlabelled, not a verified absence. Nor does a catalogue
    record prove that archaeology is visible in the satellite image. Independent visibility
    labels and verified negatives would be needed for a detection-accuracy assessment.

    </details>
    '''), md('''
    **Your transfer review:**
    1. Compare a high-scoring background and a low-scoring catalogued location.
    2. Identify at least three plausible causes of a source-to-Cyprus performance change.
    3. Explain why treating all background chips as negatives would invalidate an accuracy claim.
    4. Explain whether the scores support morphology detection or landscape-context ranking.
    5. Specify the imagery, labels and independent geographic split needed for a real transfer study.
    '''), code('''
    transfer_review = {
        "high_scoring_background": "",
        "low_scoring_catalogued_location": "",
        "domain_shift_causes": [],
        "why_no_accuracy_score": "",
        "next_study": "",
    }
    ''')]
    if instructor:
        cells[-1] = code('''
        high_bg = cyprus.loc[cyprus.kind == "unlabelled_background"].sort_values("score", ascending=False).iloc[0]
        low_site = cyprus.loc[cyprus.kind == "catalogued_location"].sort_values("score").iloc[0]
        transfer_review = {
            "high_scoring_background": f"{high_bg.tile_id}: score {high_bg.score:.3f}; this is unlabelled, not a confirmed false positive.",
            "low_scoring_catalogued_location": f"{low_site.tile_id}: score {low_site.score:.3f}; catalogue membership does not establish visible tell morphology.",
            "domain_shift_causes": ["10 m Sentinel-2 versus finer rendered source imagery",
                                   "Different RGB rendering and acquisition conditions",
                                   "Cypriot settlement types need not resemble source tells",
                                   "Landscape, survey coverage and coordinate uncertainty"],
            "why_no_accuracy_score": "We lack image-visibility ground truth and verified negatives. Source probabilities are uncalibrated in this target domain.",
            "next_study": "Acquire licensed finer-resolution multi-date imagery, expert visibility labels and provenance; hold out complete geographic areas and distinct sites, then test ranking at a realistic review budget.",
        }
        display(transfer_review)
        ''')
    cells += [md('''
    ### 10. Freeze your choice and prepare your submission

    You have chosen a model using validation data. **Freeze it before looking at the test labels.**
    Run the next cell to score each test tile and write `submission.csv` and `configuration.json`.
    Then fill in the recommendation and run the final cell to create `team_submission.zip`.

    The archive is publicly labelled, so this is an educational holdout, not a secure blind test.
    Your recommendation should describe the intended use, validation evidence, limitations and
    the next study an archaeologist would need.
    '''), code('''
    frozen_configuration = {
        "selected_model": best_key, "seed": SEED, "review_budget": REVIEW_BUDGET,
        "training_tiles": len(train), "test_tiles": len(test),
        "dataset": "Enki Normalized_Tell_combined_350.zip, curated 600-image workshop sample",
        "sample_manifest_sha256": file_digest(assets / "sample_manifest.json"),
        "feature_representation": best_representation,
        "encoder": MODEL_ID if best_representation == "dinov2" else None,
        "encoder_revision": MODEL_REVISION if best_representation == "dinov2" else None,
        "preprocessing": PREPROCESS_ID if best_representation == "dinov2" else None,
        "geographic_independence_verified": False,
    }
    test_scores = best_model.predict_proba(feature_sets[best_representation]["test"])[:, 1]
    submission = write_submission(test, test_scores, ROOT / "submission.csv")
    (ROOT / "configuration.json").write_text(json.dumps(frozen_configuration, indent=2))
    display(submission.head())
    print("Model frozen. Predictions saved to", ROOT / "submission.csv")
    '''), code('''
    recommendation = {
        "use_case": "",
        "validation_evidence": "",
        "main_limitations": [],
        "what_we_would_do_next": "",
    }
    # Replace the empty responses before sharing your submission.
    ''' if not instructor else '''
    selected = validation_results.iloc[0]
    recommendation = {
        "use_case": "Prioritise expert review of imagery similar to the source dataset.",
        "validation_evidence": f"{best_key}: validation recall@{REVIEW_BUDGET}={selected.recall_at_k:.3f}; AP={selected.AP:.3f}.",
        "main_limitations": ["Image-similarity split is not a geographic holdout",
                             "Curated class prevalence differs from operational survey prevalence",
                             "Cyprus sample lacks archaeological visibility labels",
                             "10 m context is insufficient for many small archaeological features"],
        "what_we_would_do_next": "Commission an independently labelled geographic holdout and audit high-priority candidates with an archaeologist before allocating field survey.",
    }
    display(recommendation)
    '''), code('''
    report = {"error_analysis": error_notes, "transfer_review": transfer_review,
              "recommendation": recommendation}
    (ROOT / "team_report.json").write_text(json.dumps(report, indent=2))
    missing = [f"{section}.{key}" for section, answers in report.items()
               for key, value in answers.items()
               if not value or (isinstance(value, str) and not value.strip())]
    if missing:
        print("Before sharing, complete these responses and rerun this cell:", ", ".join(missing))
    else:
        print("All response fields have entries. Review them for evidence and clarity.")
    output_zip = ROOT / "team_submission.zip"
    with zipfile.ZipFile(output_zip, "w", zipfile.ZIP_DEFLATED) as out:
        for name in ("submission.csv", "configuration.json", "validation_results.csv",
                     "learning_curve.csv", "cyprus_review.csv", "team_report.json"):
            out.write(ROOT / name, name)
    print("Submission saved to", output_zip, "(in Colab, use the Files panel). Save your notebook too.")
    ''')]
    cells += [md('''
    ### Before the final reveal: check what you can explain

    Without looking back, explain these three questions to a partner or jot down your answers.
    Then compare them with the explanations below. These are practice, not a grade.

    1. What is the difference between precision and recall at the review budget?
    2. Why should the final model be selected before inspecting the test results?
    3. What extra evidence would you need before claiming accuracy on the Cyprus images?

    <details><summary>Compare your answers</summary>

    1. Precision is the fraction of reviewed tiles that are positive; recall is the fraction
       of all known positive tiles that are reviewed.
    2. Selecting after seeing test performance would reuse the test for tuning. Freezing
       first leaves it as a more independent final assessment, though the split is not
       guaranteed to be geographically independent.
    3. Independently established image-visibility labels and verified negatives, plus an
       appropriate independent geographic evaluation, are needed. A catalogue location is
       not itself a visibility label and an unlabelled location is not a known absence.

    </details>
    '''), md('''
    ### 11. Reveal the final result

    After you have submitted, ask the facilitator when to change `REVEAL_TEST` to `True` and run
    the cell. Do not use this result to choose a replacement model: the test is the final check.
    Recall counts positive **tiles**, not independently verified sites. The random-ranking
    comparison is a reference for this small, balanced sample, not operational prevalence.
    '''), code(f'''
    REVEAL_TEST = {instructor}
    if REVEAL_TEST:
        final_metrics = score_submission(submission, test, ranking_metrics, REVIEW_BUDGET)
        display(pd.Series(final_metrics, name="final_holdout"))
        print("Random ranking expected recall:", min(REVIEW_BUDGET, len(test)) / len(test))
        print("Random ranking expected precision:", test.label.mean())
        # A single diagnostic view after the model was frozen.
        order = ranking_indices(test_scores)
        contact_sheet(test.iloc[order], test_scores[order], n=12, title="Final top-ranked holdout tiles")
    else:
        print("Test reveal disabled. Submit before asking the facilitator to reveal.")
    ''')]
    if instructor:
        cells += [md('''
        ### Instructor: debrief and marking guide
        **Assess reasoning, not who happened to get the highest test score.** One possible
        rubric is 40% for explaining the shortlist and validation-only model choice, 35% for
        interpreting limitations and the Cyprus transfer, and 25% for a reproducible submission
        and evidence-based recommendation. Accept a recommendation against deployment when
        supported by the results. A correct CSV has one score per requested tile ID; source
        labels and image similarity are not verified site identities.

        Use these prompts after learners have attempted the relevant questions:

        - **Precision or recall?** Precision divides hits by reviewed tiles; recall divides hits
          by all positive tiles. A higher review budget can increase hits without increasing precision.
        - **Why not choose on the test set?** Repeated choices based on test results adapt the
          workflow to that set; validation supports the choice and test is reserved for the frozen model.
        - **Does a high score prove a site?** No. It is a ranking priority; source labels and
          visually plausible features are not independent archaeological confirmation.
        - **Can we score Cyprus accuracy?** No. Catalogue presence does not establish visibility,
          and nearby unlabelled locations are not verified negatives. Ten-metre pixels also limit
          the visible detail. Ask what expert labels and geographically separate data are needed.
        - **What is a catalogue row?** A settlement can have several occupation-phase rows.
          The pinned catalogue has 1,559 IDs; blank/F quality codes are undocumented, and `CY453`
          has conflicting coordinates. A quality code is not a measured error radius.

        Before revealing the test result, ask teams to state their chosen model, supporting
        validation evidence, one limitation, and a next study. Afterward, compare the result with
        the stated expectation and discuss uncertainty and unresolved geographic leakage.
        The model cannot establish chronology, burial, legal status or a new archaeological site.
        ''')]
        cells += [md('''
        ### Optional instructor refresh of the Cyprus bundle
        The normal notebook uses the embedded sample, so a live imagery service is not required
        during class. The helper `make_cyprus_bundle` above contains the complete acquisition
        and feature-extraction recipe. To regenerate it, enable the following cell. This downloads
        public COG windows and the pinned model, and needs additional network access.
        It does not overwrite the bundled results used above.
        '''), code('''
        REFRESH_CYPRUS = False
        if REFRESH_CYPRUS:
            install_packages("rasterio>=1.3,<2", "planetary-computer==1.0.0", "transformers==4.57.6")
            regenerated = make_cyprus_bundle(phases, ROOT / "refreshed_cyprus")
            display(regenerated)
        ''')]
    cells += [md('''
    ### Sources, provenance and attribution
    - Crawford & Vella (2022), [Cyprus Dataset: Settlements from 11000 BCE to 1878 CE](https://doi.org/10.5334/joad.96).
      [Source repository](https://github.com/K-A-Crawford/Cyprus_Settlements), pinned revision
      `70a5c3d399a2451e1d8dd5eb947eb80055276b44`. Dataset: CC BY 4.0.
    - Rosati (2025), [Enki: AI for Archaeology — datasets](https://zenodo.org/records/14950565).
      We sample `Normalized_Tell_combined_350.zip` (published whole-archive MD5
      `1d4ea398ea5669f9bca4539416847206`). Partial downloads verify member CRCs and pixel hashes,
      not the full archive MD5. The embedded manifest records every selected member and byte range.
      The record declares CC BY-SA 4.0 and separately describes Google/Bing imagery provenance
      and provider conditions. Retain those notices when reusing its imagery or derived materials.
    - Oquab et al., [DINOv2](https://github.com/facebookresearch/dinov2) and
      [model card](https://huggingface.co/facebook/dinov2-small). Model revision and preprocessing
      are recorded in the cache. The workshop adds no encoder fine-tuning.
    - Cyprus chips: contains modified Copernicus Sentinel data (2024), accessed through
      [Microsoft Planetary Computer](https://planetarycomputer.microsoft.com/dataset/sentinel-2-l2a).
      Scene `S2A_MSIL2A_20240805T082601_R021_T36SVD_20240805T144559`.
      No commercial imagery account or Earth Engine authentication is needed for this sample.
    - Follow-on EO model: [RemoteCLIP](https://github.com/ChenDelong1999/RemoteCLIP).
      It is not used to generate this notebook's reported results.

    **Troubleshooting:** if Zenodo is temporarily unavailable or refuses byte ranges, use the
    facilitator's `enki_workshop_sample.zip` in `archaeology_workshop/`; the member checks still
    apply. Do not put the original full archive at that path. A restarted runtime loses local outputs
    unless you download them. If a helper
    name is undefined, rerun setup and the infrastructure cells in order. For the default path,
    leave GPU recomputation and live Cyprus refresh disabled.
    ''')]
    nb = nbf.v4.new_notebook(cells=cells, metadata={
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "version": "3.12"},
        "colab": {"name": "archaeology_instructor.ipynb" if instructor else "archaeology_participants.ipynb",
                  "provenance": [], "toc_visible": True},
    })
    # Stable cell IDs make differences between regenerated notebooks reviewable.
    for i, cell in enumerate(nb.cells):
        cell.id = f"{'instructor' if instructor else 'participant'}-{i:03d}"
    return nb


if __name__ == "__main__":
    payload = embedded_assets()
    for instructor, filename in ((False, "archaeology_participants.ipynb"),
                                 (True, "archaeology_instructor.ipynb")):
        nb = notebook(instructor, payload)
        nbf.validate(nb)
        nbf.write(nb, HERE / filename)
        print(filename, len(nb.cells), "cells")
