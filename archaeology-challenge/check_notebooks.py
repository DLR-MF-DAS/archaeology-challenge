"""Check scientific helpers and run the standalone notebooks on the prepared real sample."""
import ast
import base64
import io
import json
import os
from pathlib import Path
import shutil
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock
import zipfile

HERE = Path(__file__).resolve().parent
WORK = HERE / ".work"
os.environ.setdefault("MPLCONFIGDIR", str(WORK / "matplotlib"))
sys.path.insert(0, str(HERE / "src"))

import nbformat
from nbclient import NotebookClient
from jupyter_client import KernelManager
import numpy as np
import pandas as pd
from PIL import Image
from archaeology_challenge import runtime
import build_notebooks as builder


class ScientificChecks(unittest.TestCase):
    @staticmethod
    def catalogue_fixture():
        rows = []
        for i, (site, background) in enumerate(((.8, .2), (.7, .1), (.9, .6),
                                               (.5, .4), (.3, .65), (.1, .2))):
            for kind, score in (("catalogued_location", site), ("unlabelled_background", background)):
                rows.append({"tile_id": f"pair{i}_{kind}", "paired_site": f"pair{i}",
                             "kind": kind, "score": score})
        return pd.DataFrame(rows)

    def test_catalogue_metrics_count_pairs_and_recovery_with_random_references(self):
        frame = self.catalogue_fixture()
        pairs, result = runtime.catalogue_agreement(frame, k=4)
        self.assertEqual((result["wins"], result["ties"], result["losses"]), (4, 0, 2))
        self.assertEqual(result["pair_agreement"], 4 / 6)
        self.assertEqual(result["catalogued_at_k"], 3)
        self.assertEqual(result["catalogue_fraction_at_k"], 3 / 4)
        self.assertEqual(result["catalogue_recovery_at_k"], 3 / 6)
        self.assertEqual(result["random_pair_agreement"], .5)
        self.assertEqual(result["random_catalogue_fraction"], .5)
        self.assertEqual(result["random_catalogue_recovery"], 4 / 12)
        self.assertFalse(result["cutoff_tie"])
        self.assertEqual((result["min_catalogued_at_k"], result["max_catalogued_at_k"]), (3, 3))
        self.assertNotIn("accuracy", result)
        shuffled_pairs, shuffled_result = runtime.catalogue_agreement(frame.sample(frac=1, random_state=8))
        pd.testing.assert_frame_equal(pairs, shuffled_pairs)
        self.assertEqual(result, shuffled_result)

    def test_catalogue_ties_and_invalid_pairs(self):
        frame = self.catalogue_fixture().assign(score=.5)
        _, result = runtime.catalogue_agreement(frame)
        self.assertEqual((result["wins"], result["ties"], result["losses"]), (0, 6, 0))
        self.assertEqual(result["pair_agreement"], .5)
        self.assertTrue(result["cutoff_tie"])
        self.assertEqual((result["min_catalogued_at_k"], result["max_catalogued_at_k"]), (0, 4))
        _, full = runtime.catalogue_agreement(frame, k=20)
        self.assertEqual(full["reviewed"], 12)
        self.assertEqual(full["catalogue_recovery_at_k"], 1)
        self.assertFalse(full["cutoff_tie"])
        # Equal displayed decimals must not turn a small genuine margin into a tie.
        frame.loc[0, "score"] = .50000001
        _, small_margin = runtime.catalogue_agreement(frame)
        self.assertEqual((small_margin["wins"], small_margin["ties"]), (1, 5))
        for budget in (0, -1, True, 1.5):
            with self.subTest(budget=budget), self.assertRaises(ValueError):
                runtime.catalogue_agreement(frame, k=budget)
        for invalid in (frame.iloc[1:], pd.concat([frame, frame.iloc[:1]]),
                        frame.assign(kind="unknown"), frame.assign(score=np.nan),
                        frame.assign(score=np.inf), frame.assign(paired_site=None), frame.iloc[:0]):
            with self.subTest(rows=len(invalid)), self.assertRaises(ValueError):
                runtime.catalogue_agreement(invalid)

    def test_cyprus_calculation_has_runnable_blanks_and_worked_solution(self):
        counts = dict(pair_wins=4, pair_ties=0, pair_count=6,
                      cyprus_hits=3, cyprus_reviewed=4, catalogued_total=6)
        for instructor in (False, True):
            source = next(c.source for c in builder.notebook(instructor, "").cells
                          if c.id == "cyprus-calculation")
            ns = dict(counts, print=lambda *args: None)
            exec(source, ns)
            if instructor:
                self.assertEqual(ns["pair_agreement"], 4 / 6)
                self.assertEqual(ns["catalogue_fraction"], 3 / 4)
                self.assertEqual(ns["catalogue_recovery"], 3 / 6)
            else:
                self.assertIs(ns["pair_agreement"], Ellipsis)
                self.assertIs(ns["catalogue_fraction"], Ellipsis)
                self.assertIs(ns["catalogue_recovery"], Ellipsis)

    def test_feature_branches_use_the_same_common_grid(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            rng = np.random.default_rng(42)
            for size in (64, 100, 350):
                raw, common = root / f"raw{size}.png", root / f"common{size}.png"
                Image.fromarray(rng.integers(0, 256, (size, size, 3), dtype=np.uint8)).save(raw)
                source_hash = runtime.image_pixel_digest(raw)
                runtime.image_input(raw).save(common)
                self.assertEqual(Image.open(common).size, (64, 64))
                np.testing.assert_array_equal(runtime.dino_input(raw), runtime.dino_input(common))
                features = runtime.handcrafted_features([raw, common])
                np.testing.assert_array_equal(features[0], features[1])
                self.assertEqual(runtime.image_pixel_digest(raw), source_hash)

    def test_feature_caches_reject_stale_tampered_and_misaligned_data(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            paths = [root / "a.png", root / "b.png"]
            for path, colour in zip(paths, ("red", "blue")):
                Image.new("RGB", (100, 100), colour).save(path)
            frame = pd.DataFrame({"tile_id": ["a", "b"], "path": list(map(str, paths)),
                                  "pixel_sha256": [runtime.image_pixel_digest(p) for p in paths]})
            vectors = np.eye(2, 384, dtype=np.float32)
            cache_path = root / "embeddings.npz"
            runtime.save_embedding_cache(frame, cache_path, vectors)
            np.testing.assert_array_equal(runtime.load_embedding_cache(frame.iloc[::-1], cache_path),
                                          vectors[::-1])
            with np.load(cache_path) as data:
                valid = {key: data[key].copy() for key in data.files}
            corruptions = [
                {"preprocessing": "old-full-resolution-input"},
                {"pixel_sha256": np.array(["changed", "changed"])},
                {"tile_ids": np.array(["a", "a"])},
                {"embeddings": np.full((2, 384), np.nan)},
                {"embeddings": np.zeros((2, 383))},
                {"tile_ids": np.array(["missing", "b"])},
                {"revision": "different-weights"},
            ]
            for change in corruptions:
                with self.subTest(change=list(change)):
                    np.savez_compressed(cache_path, **{**valid, **change})
                    with self.assertRaises(ValueError):
                        runtime.load_embedding_cache(frame, cache_path)
            cy_dir = root / "cyprus"
            cy_dir.mkdir()
            cy_frame = frame.assign(filename=[p.name for p in paths])
            for path in paths:
                shutil.copy2(path, cy_dir / path.name)
            cy_frame[["tile_id", "filename"]].to_csv(cy_dir / "manifest.csv", index=False)
            np.savez_compressed(cy_dir / "embeddings.npz", **valid)
            runtime.load_cyprus(root)
            Image.new("RGB", (100, 100), "green").save(cy_dir / "a.png")
            with self.assertRaisesRegex(ValueError, "Image/cache mismatch"):
                runtime.load_cyprus(root)

    def test_size_probe_uses_training_labels_and_strata_keep_class_counts(self):
        train = pd.DataFrame({"width": [64, 64, 350, 350], "height": [64, 64, 350, 350],
                              "label": [0, 0, 1, 1]})
        validation = pd.DataFrame({"width": [64, 350, 350], "height": [64, 350, 350],
                                   "label": [0, 0, 1]})
        expected = [0, 1, 1]
        np.testing.assert_array_equal(runtime.source_size_scores(train, validation), expected)
        np.testing.assert_array_equal(runtime.source_size_scores(train, validation.assign(label=1)), expected)
        unseen = pd.DataFrame({"width": [100], "height": [100]})
        np.testing.assert_array_equal(runtime.source_size_scores(train, unseen), [.5])
        diagnostic = runtime.validation_diagnostics(train, validation, {"model": [.1, .2, .9]}, k=1)
        strata = diagnostic.set_index("subset")
        self.assertTrue(np.isnan(strata.loc["native_64x64", "AP"]))
        self.assertEqual(strata.loc["native_350x350", "AP"], 1)
        self.assertEqual(strata.loc["native_350x350", "positive_fraction"], .5)

    def test_transfer_summary_never_uses_catalogue_membership_as_labels(self):
        table = runtime.transfer_summary([0, 1], {"a": [.2, .8]}, {"a": [.01, .02, .03]})
        target = table.loc[table.group.eq("Cyprus (unlabelled)")].iloc[0]
        self.assertEqual(target["images"], 3)
        self.assertEqual(target["median"], .02)
        self.assertNotIn("AP", table.columns)

    def test_ranking_denominators_and_empty_positives(self):
        actual = runtime.ranking_metrics([1, 0, 1, 0], [.9, .8, .7, .1], 2)
        self.assertEqual(actual["hits"], 1)
        self.assertEqual(actual["recall_at_k"], .5)
        self.assertEqual(actual["precision_at_k"], .5)
        self.assertAlmostEqual(actual["AP"], 5 / 6)
        self.assertEqual(runtime.ranking_metrics([1, 0], [.2, .1], 20)["reviewed"], 2)
        self.assertTrue(np.isnan(runtime.ranking_metrics([0, 0], [.2, .1], 1)["recall_at_k"]))
        self.assertEqual(runtime.ranking_metrics([1, 0], [.5, .5], 1)["hits"], 1)

    def test_invalid_rankings_are_rejected(self):
        for labels, scores, budget in (
            ([1, 0], [np.nan, .1], 1), ([1, 0], [.5, .1], 0),
            ([1, 0], [.5, .1], 1.5), ([.5, 0], [.5, .1], 1),
            ([], [], 1), ([1], [.5, .1], 1), ([1, 0], [[.5], [.1]], 1),
        ):
            with self.subTest(labels=labels, scores=scores, budget=budget):
                with self.assertRaises(ValueError):
                    runtime.ranking_metrics(labels, scores, budget)

    def test_submission_alignment_and_duplicate_rejection(self):
        truth = pd.DataFrame({"tile_id": ["a", "b", "c"], "label": [1, 0, 1]})
        submission = pd.DataFrame({"tile_id": ["c", "a", "b"], "score": [.7, .9, .8]})
        self.assertEqual(runtime.score_submission(submission, truth, runtime.ranking_metrics, 1)["recall_at_k"], .5)
        for ids in (["a", "a", "c"], ["a", "b"], ["a", "b", "extra"]):
            with self.subTest(ids=ids), self.assertRaises(ValueError):
                runtime.score_submission(pd.DataFrame({"tile_id": ids, "score": .5}),
                                         truth, runtime.ranking_metrics, 1)

    def test_contradictory_similarity_family_excluded(self):
        frame = pd.DataFrame({"tile_id": ["a", "b", "c"], "label": [0, 1, 1],
                              "pixel_sha256": ["a", "b", "c"],
                              "phash": ["0000000000000000", "0000000000000001", "ffffffffffffffff"]})
        clean, audit, report = runtime.deduplicate_images(frame)
        self.assertEqual(clean.tile_id.tolist(), ["c"])
        self.assertEqual(report["mixed_label_families_excluded"], 1)
        self.assertFalse(report["geographic_independence_verified"])

    def test_catalogue_conflicts_keep_ids_without_inventing_coordinates(self):
        phases = pd.DataFrame({
            "ID": ["a", "a", "b", "c"], "Toponym": ["A", "A", "B", "C"],
            "Period": ["early", "late", "early", "late"], "LocQual": [" A ", "A", "F", None],
            "Long_Cord": [32.5, 32.6, 32.8, 32.9], "Lat_Cord": [34.8, 34.8, 35, 35.1],
        })
        sites = runtime.summarise_sites(phases).set_index("ID")
        self.assertEqual(len(sites), 3)
        self.assertEqual(sites.loc["a", "occupation_rows"], 2)
        self.assertTrue(sites.loc["a", ["lon", "lat"]].isna().all())
        self.assertTrue(sites.loc["a", "quality_documented"])
        self.assertFalse(sites.loc["b", "quality_documented"])
        self.assertFalse(sites.loc["c", "quality_documented"])

    def test_both_models_can_drive_the_cyprus_notebook_cell(self):
        nb = builder.notebook(False, "")
        source = next(cell.source for cell in nb.cells if cell.id == "cyprus")
        frame = pd.DataFrame({"tile_id": ["site", "background"], "path": ["a", "b"],
                              "kind": ["catalogued_location", "unlabelled_background"],
                              "paired_site": ["pair", "pair"]})
        matrices = {"colour_texture": np.array([[1, 2], [3, 4]]),
                    "dinov2": np.array([[5, 6, 7], [8, 9, 10]])}
        for selected in matrices:
            with self.subTest(selected=selected):
                models = {name: Mock() for name in matrices}
                for model in models.values():
                    model.predict_proba.return_value = np.array([[.8, .2], [.1, .9]])
                helper = SimpleNamespace(
                    load_cyprus=lambda assets: (frame.copy(), matrices["dinov2"]),
                    handcrafted_features=lambda paths: matrices["colour_texture"],
                    contact_sheet=Mock(),
                    transfer_summary=runtime.transfer_summary,
                )
                ns = dict(workshop=helper, assets=Path("unused"), models=models,
                          selected=selected, display=lambda value: None,
                          validation=pd.DataFrame({"label": [0, 1]}),
                          validation_scores={name: np.array([.1, .9]) for name in matrices})
                exec(source, ns)
                for name in matrices:
                    np.testing.assert_array_equal(models[name].predict_proba.call_args.args[0], matrices[name])
                    self.assertIn(f"{name}_score", ns["cyprus"].columns)
                self.assertEqual(ns["cyprus"].tile_id.tolist(), ["background", "site"])

    def test_exports_are_complete_and_preserve_the_recommendation(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            truth = pd.DataFrame({"tile_id": ["a", "b"], "label": [1, 0]})
            runtime.write_submission(truth, [.1, .9], root / "submission.csv")
            result = pd.DataFrame({"model": ["colour_texture"], "recall_at_k": [.5]})
            cyprus = pd.DataFrame({"tile_id": ["site"], "path": ["/local/image.png"], "score": [.2]})
            path = runtime.export_results(root, result, cyprus, "Inspect before deployment.",
                                          {"selected_model": "colour_texture", "review_budget": 50},
                                          data_audit={"native_size_counts": []},
                                          validation_audit=result, transfer_audit=result,
                                          cyprus_metrics=result, cyprus_pairs=result)
            with zipfile.ZipFile(path) as bundle:
                self.assertEqual(set(bundle.namelist()), {"submission.csv", "configuration.json",
                                 "validation_results.csv", "cyprus_review.csv", "recommendation.md",
                                 "data_audit.json", "validation_diagnostics.csv", "transfer_diagnostics.csv",
                                 "cyprus_catalogue_metrics.csv", "cyprus_pairs.csv"})
                self.assertEqual(bundle.read("recommendation.md").decode(), "Inspect before deployment.\n")
                config = json.loads(bundle.read("configuration.json"))
                self.assertIsNone(config["encoder"])
                self.assertFalse(config["geographic_independence_verified"])
                self.assertEqual(config["common_image_size"], 64)
                self.assertFalse(config["cyprus_scores_calibrated"])
                self.assertNotIn("/local/", bundle.read("cyprus_review.csv").decode())

    def test_test_reveal_uses_exported_choice_and_budget(self):
        source = next(cell.source for cell in builder.notebook(False, "").cells
                      if cell.id == "reveal")
        # The disabled participant cell must run without any test data in scope.
        exec(source, {"print": Mock()})
        truth = pd.DataFrame({"tile_id": list("abcdef"), "label": [1, 1, 0, 1, 0, 0]})
        submission = pd.DataFrame({"tile_id": list("abcdef"), "score": [.8, .6, .7, .9, .4, .1]})
        saved_validation = {"hits": 1, "reviewed": 2, "positives": 3,
                            "precision_at_k": .5, "recall_at_k": 1 / 3, "AP": .5}
        helper = SimpleNamespace(score_submission=Mock(wraps=runtime.score_submission),
                                 ranking_metrics=runtime.ranking_metrics)
        ns = dict(pd=pd, workshop=helper, test=truth, submission=submission,
                  submitted_model="dinov2", submitted_budget=2,
                  submitted_validation=saved_validation.copy(),
                  selected="colour_texture", REVIEW_BUDGET=5,
                  display=Mock(), print=Mock())
        exec(source.replace("REVEAL_TEST = False", "REVEAL_TEST = True"), ns)
        self.assertEqual(helper.score_submission.call_args.args[-1], 2)
        self.assertEqual(ns["comparison"].loc["Test", "reviewed"], 2)
        self.assertEqual(ns["comparison"].loc["Test", "hits"], 2)
        self.assertEqual(ns["comparison"].loc["Test", "recall_at_k"], 2 / 3)
        self.assertEqual(ns["comparison"].loc["Validation", "hits"], 1)
        self.assertEqual(ns["submitted_validation"], saved_validation)
        ns["print"].assert_any_call("Exported model: dinov2; review budget: 2")

    def test_generated_notebooks_match_source_and_bundle(self):
        payload = builder.embedded_assets()
        with zipfile.ZipFile(io.BytesIO(base64.b64decode(payload))) as bundle:
            self.assertEqual(bundle.read("workshop_helpers.py"),
                             (HERE / "src/archaeology_challenge/runtime.py").read_bytes())
            for name in ("enki_embeddings.npz", "cyprus/embeddings.npz"):
                with np.load(io.BytesIO(bundle.read(name))) as cache:
                    self.assertEqual(str(cache["preprocessing"]), runtime.PREPROCESS_ID)
                    self.assertEqual(str(cache["revision"]), runtime.MODEL_REVISION)
                    self.assertEqual(cache["pixel_sha256"].shape, cache["tile_ids"].shape)
                    self.assertTrue(np.isfinite(cache["embeddings"]).all())
        for instructor, filename in ((False, "archaeology_participants.ipynb"),
                                     (True, "archaeology_instructor.ipynb")):
            nb = nbformat.read(HERE / filename, as_version=4)
            nbformat.validate(nb)
            expected = builder.notebook(instructor, payload)
            self.assertEqual([(c.id, c.source) for c in nb.cells], [(c.id, c.source) for c in expected.cells])
            for cell in nb.cells:
                if cell.cell_type == "code":
                    ast.parse(cell.source)
            if not instructor:
                reveal = next(c for c in nb.cells if c.id == "reveal")
                self.assertIn("REVEAL_TEST = False", reveal.source)
                by_id = {cell.id: cell for cell in nb.cells}
                order = [cell.id for cell in nb.cells]
                self.assertLess(order.index("train-models"), order.index("score-validation"))
                self.assertLess(order.index("cyprus-pair"), order.index("optional-experiments"))
                self.assertLess(order.index("cyprus-counts"), order.index("cyprus-calculation"))
                self.assertLess(order.index("cyprus-calculation"), order.index("cyprus-metrics-check"))
                self.assertLess(order.index("cyprus-metrics-check"), order.index("optional-experiments"))
                for identifier in ("projection-exercise", "learning-curve", "cyprus-study-design"):
                    self.assertLess(order.index("optional-experiments"), order.index(identifier))
                    self.assertLess(order.index(identifier), order.index("report-introduction"))
                self.assertIn("RUN_PROJECTION = False", by_id["projection-plot"].source)
                self.assertIn("RUN_LEARNING_CURVES = False", by_id["learning-curve"].source)
                self.assertIn("RUN_CYPRUS_PRACTICE = False", by_id["cyprus-study-practice"].source)
                self.assertLess(order.index("learning-curve"), order.index("learning-curve-response"))
                self.assertLess(order.index("cyprus-study-response"), order.index("cyprus-study-practice"))
                self.assertLess(order.index("cyprus-study-practice"), order.index("report-introduction"))
                final_steps = ("report-evidence", "recommendation", "export", "reveal", "debrief-notes")
                for earlier, later in zip(final_steps, final_steps[1:]):
                    self.assertLess(order.index(earlier), order.index(later))


def execute_notebook(filename, participant=False):
    nb = nbformat.read(HERE / filename, as_version=4)
    workspace = WORK / ("participant_execution" if participant else "instructor_execution")
    data_dir = workspace / "archaeology_workshop"
    data_dir.mkdir(parents=True, exist_ok=True)
    sample = data_dir / "enki_workshop_sample.zip"
    if not sample.exists():
        candidates = [HERE / "archaeology_workshop/enki_workshop_sample.zip",
                      WORK / "data/enki_workshop_sample.zip"]
        cached = next((path for path in candidates if path.exists()), None)
        if cached:
            shutil.copy2(cached, sample)
    env = os.environ.copy()
    env.pop("ARCHAEOLOGY_WORKSHOP_ROOT", None)
    env["ARCHAEOLOGY_SKIP_INSTALL"] = "1"
    env["MPLCONFIGDIR"] = str(WORK / "matplotlib")
    env["JUPYTER_RUNTIME_DIR"] = str(WORK / "jupyter-runtime")
    env["IPYTHONDIR"] = str(WORK / "ipython")
    env["PATH"] = str(Path(sys.executable).parent) + os.pathsep + env.get("PATH", "")
    print("Executing", filename, "with real data", flush=True)
    manager = KernelManager(kernel_name="python3")
    manager.kernel_spec.argv = [sys.executable, "-m", "ipykernel_launcher", "-f", "{connection_file}"]
    client = NotebookClient(nb, timeout=600, km=manager,
                            resources={"metadata": {"path": str(workspace)}})
    try:
        client.execute(env=env)
    except Exception:
        nbformat.write(nb, WORK / (Path(filename).stem + "_failed.ipynb"))
        raise
    else:
        target = WORK / "participant_validation.ipynb" if participant else HERE / filename
        nbformat.write(nb, target)
    finally:
        # A supplied manager is owned by this script, not nbclient.
        client._cleanup_kernel()
    print("Executed successfully:", filename, flush=True)


if __name__ == "__main__":
    result = unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(ScientificChecks))
    if not result.wasSuccessful():
        sys.exit(1)
    if "--execute" in sys.argv:
        execute_notebook("archaeology_instructor.ipynb")
        execute_notebook("archaeology_participants.ipynb", participant=True)
