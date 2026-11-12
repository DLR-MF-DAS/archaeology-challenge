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
from archaeology_challenge import runtime
import build_notebooks as builder


class ScientificChecks(unittest.TestCase):
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
                models[selected].predict_proba.return_value = np.array([[.8, .2], [.1, .9]])
                helper = SimpleNamespace(
                    load_cyprus=lambda assets: (frame.copy(), matrices["dinov2"]),
                    handcrafted_features=lambda paths: matrices["colour_texture"],
                    contact_sheet=Mock(),
                )
                ns = dict(workshop=helper, assets=Path("unused"), models=models,
                          selected=selected, display=lambda value: None)
                exec(source, ns)
                np.testing.assert_array_equal(models[selected].predict_proba.call_args.args[0], matrices[selected])
                models[next(name for name in matrices if name != selected)].predict_proba.assert_not_called()
                self.assertEqual(ns["cyprus"].tile_id.tolist(), ["background", "site"])

    def test_exports_are_complete_and_preserve_the_recommendation(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            truth = pd.DataFrame({"tile_id": ["a", "b"], "label": [1, 0]})
            runtime.write_submission(truth, [.1, .9], root / "submission.csv")
            result = pd.DataFrame({"model": ["colour_texture"], "recall_at_k": [.5]})
            cyprus = pd.DataFrame({"tile_id": ["site"], "path": ["/local/image.png"], "score": [.2]})
            path = runtime.export_results(root, result, cyprus, "Inspect before deployment.",
                                          {"selected_model": "colour_texture", "review_budget": 50})
            with zipfile.ZipFile(path) as bundle:
                self.assertEqual(set(bundle.namelist()), {"submission.csv", "configuration.json",
                                 "validation_results.csv", "cyprus_review.csv", "recommendation.md"})
                self.assertEqual(bundle.read("recommendation.md").decode(), "Inspect before deployment.\n")
                config = json.loads(bundle.read("configuration.json"))
                self.assertIsNone(config["encoder"])
                self.assertFalse(config["geographic_independence_verified"])
                self.assertNotIn("/local/", bundle.read("cyprus_review.csv").decode())

    def test_generated_notebooks_match_source_and_bundle(self):
        payload = builder.embedded_assets()
        with zipfile.ZipFile(io.BytesIO(base64.b64decode(payload))) as bundle:
            self.assertEqual(bundle.read("workshop_helpers.py"),
                             (HERE / "src/archaeology_challenge/runtime.py").read_bytes())
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
                for identifier in ("projection-exercise", "learning-curve", "cyprus-study-design"):
                    self.assertLess(order.index("optional-experiments"), order.index(identifier))
                    self.assertLess(order.index(identifier), order.index("report-introduction"))
                self.assertIn("RUN_PROJECTION = False", by_id["projection-plot"].source)
                self.assertIn("RUN_LEARNING_CURVES = False", by_id["learning-curve"].source)


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
