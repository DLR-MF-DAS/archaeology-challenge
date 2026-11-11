"""Validate scientific helpers and execute both notebook paths on the real prepared sample."""
import ast
import copy
import os
from pathlib import Path
import shutil
import sys
import unittest

os.environ.setdefault("MPLCONFIGDIR", str(Path('.work/matplotlib').resolve()))
import nbformat
from nbclient import NotebookClient
import numpy as np
import pandas as pd
from archaeology_challenge import runtime
import build_notebooks as builder

namespace = vars(runtime).copy()
exec(builder.METRICS_SOLUTION, namespace)
metrics = namespace['ranking_metrics']


class ScientificChecks(unittest.TestCase):
    def test_ranking_denominators_and_empty_positives(self):
        actual = metrics([1, 0, 1, 0], [.9, .8, .7, .1], 2)
        self.assertEqual(actual['hits'], 1)
        self.assertEqual(actual['recall_at_k'], .5)
        self.assertEqual(actual['precision_at_k'], .5)
        self.assertAlmostEqual(actual['AP'], 5 / 6)
        self.assertEqual(metrics([1, 0], [.2, .1], 20)['reviewed'], 2)
        self.assertTrue(np.isnan(metrics([0, 0], [.2, .1], 1)['recall_at_k']))
        with self.assertRaises(ValueError):
            metrics([1, 0], [np.nan, .1], 1)
        with self.assertRaises(ValueError):
            metrics([1, 0], [.5, .1], 0)

    def test_submission_alignment_and_duplicate_rejection(self):
        truth = pd.DataFrame({'tile_id': ['a', 'b', 'c'], 'label': [1, 0, 1]})
        submission = pd.DataFrame({'tile_id': ['c', 'a', 'b'], 'score': [.7, .9, .8]})
        self.assertEqual(runtime.score_submission(submission, truth, metrics, 1)['recall_at_k'], .5)
        duplicate = pd.DataFrame({'tile_id': ['a', 'a', 'c'], 'score': [.9, .8, .7]})
        with self.assertRaises(ValueError):
            runtime.score_submission(duplicate, truth, metrics, 1)

    def test_contradictory_similarity_family_excluded(self):
        frame = pd.DataFrame({'tile_id': ['a', 'b', 'c'], 'label': [0, 1, 1],
                              'pixel_sha256': ['a', 'b', 'c'],
                              'phash': ['0000000000000000', '0000000000000001', 'ffffffffffffffff']})
        clean, audit, report = runtime.deduplicate_images(frame)
        self.assertEqual(clean.tile_id.tolist(), ['c'])
        self.assertEqual(report['mixed_label_families_excluded'], 1)
        self.assertFalse(report['geographic_independence_verified'])

    def test_generated_notebook_structure(self):
        for filename in ('archaeology_participants.ipynb', 'archaeology_instructor.ipynb'):
            nb = nbformat.read(filename, as_version=4)
            nbformat.validate(nb)
            exercises = []
            for cell in nb.cells:
                if cell.cell_type == 'code':
                    ast.parse(cell.source)
                    if any(t.startswith('exercise-') for t in cell.metadata.get('tags', [])):
                        exercises.append(cell)
            self.assertEqual(len(exercises), 4)
            if 'instructor' in filename:
                self.assertTrue(all('NotImplementedError' not in c.source for c in exercises))
            else:
                self.assertTrue(all('NotImplementedError' in c.source for c in exercises))


def execute_notebook(filename, participant=False):
    nb = nbformat.read(filename, as_version=4)
    if participant:
        solutions = {'exercise-1': builder.METRICS_SOLUTION, 'exercise-2': builder.PROBE_SOLUTION,
                     'exercise-3': builder.SEARCH_SOLUTION, 'exercise-4': builder.CYPRUS_SOLUTION}
        for cell in nb.cells:
            for tag in cell.metadata.get('tags', []):
                if tag in solutions:
                    cell.source = solutions[tag]
    workspace = Path('.work/participant_execution' if participant else '.work/instructor_execution').resolve()
    data_dir = workspace / 'archaeology_workshop'
    data_dir.mkdir(parents=True, exist_ok=True)
    sample = data_dir / 'enki_workshop_sample.zip'
    if not sample.exists():
        os.link(Path('.work/data/enki_workshop_sample.zip').resolve(), sample)
    env = os.environ.copy()
    env.pop('ARCHAEOLOGY_WORKSHOP_ROOT', None)
    env['ARCHAEOLOGY_SKIP_INSTALL'] = '1'
    env['MPLCONFIGDIR'] = str(Path('.work/matplotlib').resolve())
    env['JUPYTER_RUNTIME_DIR'] = str(Path('.work/jupyter-runtime').resolve())
    env['IPYTHONDIR'] = str(Path('.work/ipython').resolve())
    env['PATH'] = str(Path('.venv/bin').resolve()) + os.pathsep + env.get('PATH', '')
    print('Executing', filename, 'with real data', flush=True)
    client = NotebookClient(nb, timeout=600, kernel_name='python3',
                            resources={'metadata': {'path': str(workspace)}})
    try:
        client.execute(env=env)
    finally:
        target = Path('.work/participant_solved_validation.ipynb') if participant else Path(filename)
        nbformat.write(nb, target)
    errors = [o for c in nb.cells if c.cell_type == 'code' for o in c.outputs if o.output_type == 'error']
    assert not errors
    print('Executed successfully:', filename, flush=True)


if __name__ == '__main__':
    result = unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(ScientificChecks))
    if not result.wasSuccessful():
        sys.exit(1)
    if '--execute' in sys.argv:
        execute_notebook('archaeology_instructor.ipynb')
        execute_notebook('archaeology_participants.ipynb', participant=True)
