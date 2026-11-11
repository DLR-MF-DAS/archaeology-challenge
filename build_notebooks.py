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
def ranking_metrics(y_true, scores, k=50):
    # TODO 1: implement the metric dictionary described above.
    # ranking_indices(scores) provides descending, stable ordering.
    # Use average_precision_score for AP. Clip k to the number of tiles.
    # Return NaN for AP/recall if there are no positive labels.
    raise NotImplementedError("Complete Exercise 1: ranking under a review budget")
'''

PROBE_SOLUTION = '''
def make_probe(C=1.0):
    # Fitting the pipeline on training data fits the scaler there, too.
    return make_pipeline(
        StandardScaler(),
        LogisticRegression(C=C, class_weight="balanced", max_iter=2000, random_state=SEED),
    )
'''
PROBE_EXERCISE = '''
def make_probe(C=1.0):
    # TODO 2: return a pipeline containing StandardScaler and LogisticRegression.
    # Pass C, class_weight="balanced", max_iter=2000 and random_state=SEED.
    # Do not fit anything in this function.
    raise NotImplementedError("Complete Exercise 2: the linear probe")
'''

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
# TODO 3: compare both representations and C in (0.01, 0.1, 1.0, 10.0).
# Fit on matrices["train"] / y_train; score matrices["validation"] / y_val.
# Required outputs:
# candidate_models[key] = (representation, fitted_estimator)
# validation_results: DataFrame with model, representation, C and metric columns
# best_key, best_representation, best_model
# Select highest recall_at_k, breaking ties with AP; never inspect test labels.
raise NotImplementedError("Complete Exercise 3: validation-only model selection")
'''

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
def summarise_sites(phases):
    # TODO 4: return one row per ID, with columns:
    # ID, Toponym, lon, lat, LocQual, occupation_rows, periods, quality_documented,
    # quality_conflict, coordinate_conflict.
    # lon/lat come from Long_Cord/Lat_Cord, NOT Longitude/Latitude.
    # Strip whitespace in LocQual; only A-E have documented definitions.
    # Keep unknown values and mark quality_documented=False. Join distinct quality
    # codes with "|" and flag quality_conflict if a site has multiple codes.
    # Count distinct (ID, Long_Cord, Lat_Cord) rows per ID. For sites with multiple
    # locations, set coordinate_conflict=True and lon/lat=NaN rather than choosing one.
    # Use groupby(...).agg(...) and count distinct Period values.
    raise NotImplementedError("Complete Exercise 4: occupation phases versus sites")
'''


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
    # From Tells to Cyprus
    ## Foundation models for archaeological survey prioritisation
    **{role} · four hours · Python / Google Colab · no paid services required**

    You support an archaeological team with time to review **50 image tiles**. Can a
    frozen foundation model improve their shortlist with only **up to 100 training
    examples per class**? What happens when its image features are transferred to Cyprus?

    **You will produce:** a ranked CSV, a validation comparison, a label-budget experiment,
    a Cyprus transfer review, and a short recommendation to an archaeologist.

    This notebook is self-contained. Upload it at [Google Colab](https://colab.research.google.com/)
    using **File → Upload notebook**. A standard CPU runtime is enough for the default path:
    verified DINOv2 features and 12 real Cyprus Sentinel-2 chips are embedded below.
    The first data cell downloads a **600-image, balanced Enki sample (~73 MB transfer)** using
    indexed HTTP byte ranges. It does not download the complete 1.7 GB training or 12 GB test
    archive. Allow roughly 400 MB of free disk for the workshop.
    Optional feature extraction uses a small pretrained model and benefits from a T4 GPU;
    GPU availability in Colab is not guaranteed.

    **Run cells in order.** {'All exercises below are solved; this notebook can run top to bottom.' if instructor else 'Complete the four TODO cells and their checks as you go. Do not use Run all until those exercises are complete.'}
    ''')]
    if instructor:
        cells += [md('''
        ### Facilitation notes
        Share only the participant notebook during the challenge. The saved outputs here were
        produced using real Enki images and the pinned foundation model; performance is measured,
        not an expected target that participants must reproduce exactly.

        Prepare by opening the participant notebook in a fresh Colab session and running setup
        and data acquisition before the session. The embedded feature cache removes GPU queues
        and model downloads from the critical path. For poor venue connectivity, distribute
        `enki_workshop_sample.zip` beforehand and upload it to `archaeology_workshop/`.
        Every member is checked against the original ZIP directory's CRC and decoded pixel hashes.

        This is an **educational, honour-based holdout** drawn from a public labelled archive.
        It is not a secure blind competition: labels remain recoverable from the source data.
        Do not claim geographical independence: this archive supplies no verified site grouping
        or coordinate manifest. Similar-image deduplication reduces one source of leakage only.
        ''')]
    cells += [md('''
    ### Schedule and scientific framing

    | Time | Task |
    |---|---|
    | 00:00–00:20 | Briefing; launch data download |
    | 00:20–00:45 | Inspect imagery, labels and duplicate audit |
    | 00:45–01:10 | Implement ranking metrics; run baseline |
    | 01:10–01:55 | Frozen foundation-model features and model selection |
    | 01:55–02:05 | Break |
    | 02:05–02:40 | Label efficiency and error analysis |
    | 02:40–03:15 | Cyprus catalogue and image-transfer audit |
    | 03:15–03:35 | Freeze model; export shortlist and recommendation |
    | 03:35–04:00 | Presentations and debrief |

    Enki's labels concern tells and mound-like features in rendered RGB imagery. A high score
    is a review priority, not evidence of a newly discovered site. The Cyprus catalogue records
    settlement locations and occupation periods; it does **not** provide image visibility labels.
    Our Cyprus images are **1 km windows at 10 m/pixel** from a single Sentinel-2 scene.
    This introduces resolution, rendering, landscape and target differences simultaneously.
    The Cyprus task is a stress test of interpretation, not a numerical detection benchmark.
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
    The next collapsed cell unpacks the embedded **frozen features**, Cyprus catalogue and
    Cyprus sample. Features are bound to image pixel hashes, model revision and preprocessing.
    They contain no fitted classifier or solution to the exercises. The full Enki imagery is
    downloaded from its original repository in the next section.
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
    ### 2. Load and audit Enki
    We use **600 images from the published training archive**: 300 negatives and 300 positives
    sampled across five RGB landscape-context folders. The explicitly masked and black-and-white
    positive folders are excluded. The fixed manifest samples blocks of ten files to reduce network
    requests; this is a curated teaching sample, not a representative prevalence estimate.
    Our train/validation/test names are workshop splits, not the authors' original experiment.
    Labels come from the archive's `Tell` and `Negative` directories; macOS sidecars are excluded.
    Context subfolders are retained for error analysis, never as features. Byte ranges and ZIP-member
    CRCs are verified, then decoded image hashes are checked against the embedding cache.

    We group images with close 64-bit perceptual hashes (Hamming distance ≤4), exclude groups
    with contradictory labels, and retain one representative per group. This can miss shifted,
    rotated or differently rendered views of the same site. Similarity is not a site identifier.
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
                          for name, part in splits.items()}).rename_axis("label"))
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
    ### 3. Exercise 1 — evaluate the archaeologist's shortlist
    Implement `ranking_metrics(y_true, scores, k)` returning:
    - `AP`: average precision over the full ranked set.
    - `recall_at_k`: positives in the top k / all positives.
    - `precision_at_k`: positives in the top k / number actually reviewed.
    - `hits` and `reviewed`: integer counts.

    Use `min(k, number_of_tiles)` as the effective review budget. With no positives,
    AP and recall are undefined: return `np.nan`. Rank by score descending; score ties follow
    the tile-ID order. A score is not automatically a calibrated archaeological probability.
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
    ### 4. Conventional features and foundation-model features
    The conventional baseline uses RGB histograms, moments and gradient statistics. DINOv2-small
    supplies 384-dimensional frozen CLS-token embeddings. It is a general visual foundation
    model, **not an EO-specialised model**. Comparing its features to the baseline is the main
    experiment; an EO-specific encoder such as RemoteCLIP is a follow-on experiment.

    Both representations use the same tiles and training labels. The bundled DINOv2 features
    were computed without adapting the encoder to any workshop labels. Each full image is
    resized to 224×224, normalised using ImageNet channel statistics, and the CLS vector is L2
    normalised. We use the same preprocessing for Cyprus.
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
    **Optional GPU exercise:** set `RECOMPUTE_DINO=True` below. In Colab, select
    **Runtime → Change runtime type → T4 GPU** before starting if one is available.
    The default cached path needs no model download. Recomputing downloads about 90 MB of
    weights and uses batches of 16 at 224×224; it does not fine-tune the model.
    A CPU also works, but allow additional time. This cell must precede model fitting.
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
    ### 5. Exercise 2 — a lightweight classifier
    Return a scikit-learn pipeline with `StandardScaler` followed by `LogisticRegression`.
    The pipeline ensures that feature scaling is learned from training data only.
    `C` controls inverse regularisation strength: smaller values mean stronger regularisation.
    '''), code(PROBE_SOLUTION if instructor else PROBE_EXERCISE, tags=["exercise-2"]),
              code('''
              baseline = make_probe(C=1.0).fit(feature_sets["colour_texture"]["train"], y_train)
              baseline_val = baseline.predict_proba(feature_sets["colour_texture"]["validation"])[:, 1]
              display(pd.Series(ranking_metrics(y_val, baseline_val, REVIEW_BUDGET), name="baseline"))
              assert len(baseline.steps) == 2
              assert not hasattr(make_probe().steps[0][1], "mean_")
              ''')]
    cells += [md('''
    ### 6. Exercise 3 — select on validation data only
    Compare the two representations with `C ∈ {0.01, 0.1, 1, 10}`. Use validation
    recall at the fixed review budget as the primary criterion, then AP as a tie-breaker.
    Keep the test set untouched. Even if the conventional baseline wins, report that result.
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
    ### 7. Label efficiency and failure analysis
    Compare 10, 25, 50 and 100 training labels **per class**, with three random selections at each
    budget. The validation set and selected regularisation stay fixed. These curves describe
    variation in training-label sampling, not uncertainty across independent geographical regions.
    Choose one result to explain: a small-label gain, a plateau, or a failure mode.
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
    false_alarms = failures.loc[failures.label == 0].sort_values("score", ascending=False)
    missed = failures.loc[failures.label == 1].sort_values("score")
    contact_sheet(false_alarms, false_alarms.score.to_numpy(), n=8, title="Highest-scoring labelled backgrounds")
    contact_sheet(missed, missed.score.to_numpy(), n=8, title="Lowest-scoring labelled positives")
    # Inspect model neighbours for one difficult validation image.
    query_index = int(np.argmax(np.where(y_val == 0, dino_val_scores, -np.inf)))
    neighbours, similarity = nearest_examples(feature_sets["dinov2"]["validation"][query_index],
                                              feature_sets["dinov2"]["train"], train)
    contact_sheet(neighbours, similarity, n=4, title="Nearest training examples to the strongest false alarm")
    ''')]
    cells += [md('''
    **Record your interpretation:** identify two false-alarm mechanisms and one missed-positive
    mechanism from the actual images. Are the nearest neighbours similar because of morphology,
    colour or land use? What extra evidence would resolve the ambiguity?
    '''), code('''
    error_notes = {
        "false_alarm_1": "",
        "false_alarm_2": "",
        "missed_positive": "",
        "next_evidence_needed": "",
    }
    ''')]
    if instructor:
        cells[-1] = code('''
        error_notes = {
            "false_alarm_1": "Validation tile 2c58b208b shows field boundaries and a road junction; it receives a high score despite its source-negative label. Linear landscape structure is a plausible confounder.",
            "false_alarm_2": "Tile c77ec1cba contains a rectilinear built or disturbed area. Its high source-negative score suggests that geometry and exposed-soil context merit inspection; this is not a causal attribution.",
            "missed_positive": "The lowest-ranked source-positive cec8faef9 shows muted, disturbed-looking terrain. It may fall outside a limited shortlist even with a score above 0.5; shortlist misses are not the same as thresholded false negatives.",
            "next_evidence_needed": "Expert image review, source provenance, additional dates and finer-resolution imagery; field evidence for archaeological confirmation.",
        }
        # These visual interpretations refer to the pinned validation contact sheet.
        # They are not new archaeological ground-truth annotations or causal explanations.
        ''')
    cells += [md('''
    ### 8. Exercise 4 — Cyprus catalogue audit
    Load the pinned catalogue and produce one record per settlement `ID`. Occupation-phase
    rows must not become independent training examples. `Long_Cord` / `Lat_Cord` contain
    longitude / latitude in degrees; the misleadingly named `Longitude` / `Latitude` contain
    UTM zone 36N metres. The dictionary defines location quality A–E; retain and flag other values.
    Check whether all occupation rows for an ID agree on coordinates and quality. Retain IDs with
    conflicting locations, flag them and leave their single-point coordinates missing; do not invent
    a centroid or silently choose one location.

    [Field definitions](https://github.com/K-A-Crawford/Cyprus_Settlements/blob/70a5c3d399a2451e1d8dd5eb947eb80055276b44/settlement_key.txt):
    A = supplied projected coordinates; B = cadastral plans; C = maps with UTM references;
    D = maps without coordinate references; E = modern locality names.
    These are provenance categories, not numerical error radii.
    '''), code('''
    phases = pd.read_csv(assets / "Cyprus_Settlements.csv")
    print("Occupation rows:", len(phases), "Unique settlement IDs:", phases.ID.nunique())
    display(phases[["ID", "Period", "Longitude", "Latitude", "Long_Cord", "Lat_Cord", "LocQual"]].head())
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
    ### 9. Transfer stress test using real Cyprus imagery
    The embedded sample contains six catalogue locations with `LocQual=A`, plus six nearby
    **unlabelled** background locations. Background centres were placed 2 km east and checked
    to be at least 750 m from catalogue points. This does not establish archaeological absence.
    Windows with less than 90% clear-land pixels in Sentinel-2's scene classification were rejected.

    All chips use the same pinned scene from 5 August 2024, in southwest Cyprus. They were
    read from the public Planetary Computer Sentinel-2 L2A collection using COG windows.
    Each chip is 100×100 pixels (1 km square) before model resizing. Display interpolation or
    resizing cannot restore spatial detail missing at 10 m. Six pairs cannot establish transfer accuracy.

    We deliberately transfer the Enki-trained **DINOv2 probe**, even if a different model won the
    source comparison, to examine what that representation does in a different domain.
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
    ### 10. Freeze your model and submit
    Freeze the selected representation and classifier before touching the test set. Export one
    score per test tile and your scientific recommendation. The test labels are recoverable from
    the public archive, so this is an educational holdout rather than a secure competition.
    '''), code('''
    frozen_configuration = {
        "selected_model": best_key, "seed": SEED, "review_budget": REVIEW_BUDGET,
        "training_tiles": len(train), "test_tiles": len(test),
        "dataset": "Enki Normalized_Tell_combined_350.zip, curated 600-image workshop sample",
        "sample_manifest_sha256": file_digest(assets / "sample_manifest.json"),
        "encoder": MODEL_ID, "encoder_revision": MODEL_REVISION,
        "preprocessing": PREPROCESS_ID, "geographic_independence_verified": False,
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
    output_zip = ROOT / "team_submission.zip"
    with zipfile.ZipFile(output_zip, "w", zipfile.ZIP_DEFLATED) as out:
        for name in ("submission.csv", "configuration.json", "validation_results.csv",
                     "learning_curve.csv", "cyprus_review.csv", "team_report.json"):
            out.write(ROOT / name, name)
    print("Download", output_zip, "from Colab's Files panel; save your completed notebook too.")
    ''')]
    cells += [md('''
    ### 11. Final reveal — only after submission
    Ask the facilitator when to enable this cell. Do not use the result to tune or replace your model.
    Recall is measured on positive **tiles**, not independently verified archaeological sites.
    For a random ranking, expected recall@k is k/N; expected precision is the positive fraction.
    Neither this prevalence nor this small sample represents the operational archaeological base rate.
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
        - **50% ranking:** compare held-out recall at the same 50-tile budget; use AP to break ties.
        - **30% reasoning:** recognise unresolved spatial leakage, label/visibility differences,
          source-prevalence effects and the resolution mismatch in Cyprus.
        - **20% reproducibility:** correct ID-aligned CSV, pinned provenance, reproducible notebook
          and a concise recommendation supported by measured results.

        A participant may reasonably recommend against operational deployment. Reward that
        conclusion when the evidence supports it. The model is an image-ranking aid; it cannot
        establish chronology, burial, legal status or the presence of a previously unknown site.

        Expected exercise outcomes: correct top-k denominators; scaling fitted on training only;
        model selection on validation only; 1,559 distinct Cyprus IDs; blank/F quality codes and
        the conflicting coordinates for `CY453` flagged;
        no numerical transfer-accuracy claim from the unlabelled background sample.
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
