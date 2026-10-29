"""Quick scientific-accounting checks; do not need any model weights."""

import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np

from model_download import CHECKPOINT_SOURCES, download_models
from model_runtime import discover_checkpoints, margin_contributions, scenario_probabilities, softmax


class InferenceAccountingTests(unittest.TestCase):
    def test_softmax(self):
        values = softmax(np.array([-2.0, 1.0, 0.0, 3.0, 4.0]))
        self.assertEqual(values.shape, (5,))
        self.assertAlmostEqual(float(values.sum()), 1.0)
        self.assertEqual(int(np.argmax(values)), 4)

    def test_margin_is_exactly_reconstructed(self):
        rng = np.random.default_rng(42)
        weights = rng.normal(size=(5, 7))
        bias = rng.normal(size=(5,))
        concepts = rng.uniform(0, 1, size=7)
        contributions, bias_difference = margin_contributions(weights, bias, concepts, 4, 1)
        expected = (weights @ concepts + bias)[4] - (weights @ concepts + bias)[1]
        self.assertAlmostEqual(float(contributions.sum() + bias_difference), float(expected))

    def test_what_if_probabilities(self):
        weights = np.zeros((5, 7))
        bias = np.zeros(5)
        weights[3, 0] = 2.0
        a = scenario_probabilities(weights, bias, np.zeros(7))
        b = scenario_probabilities(weights, bias, np.array([1., 0, 0, 0, 0, 0, 0]))
        self.assertAlmostEqual(float(a[3]), 0.2)
        self.assertGreater(float(b[3]), float(a[3]))

    def test_invalid_concepts_are_rejected(self):
        with self.assertRaises(ValueError):
            scenario_probabilities(np.zeros((5, 7)), np.zeros(5), np.full(7, 1.2))


class CheckpointDiscoveryTests(unittest.TestCase):
    def test_loads_both_standard_names(self):
        with TemporaryDirectory() as temporary:
            folder = Path(temporary)
            (folder / "best_model.pt").touch()
            (folder / "best_cbm_model.pt").touch()
            selected, problems = discover_checkpoints(folder)
            self.assertEqual({kind: item.name for kind, item in selected.items()},
                             {"blackbox": "best_model.pt", "cbm": "best_cbm_model.pt"})
            self.assertEqual(problems, {})

    def test_reports_missing_model_without_guessing(self):
        with TemporaryDirectory() as temporary:
            folder = Path(temporary)
            (folder / "cbm_fold_1.pth").touch()
            (folder / "mystery.pt").touch()
            selected, problems = discover_checkpoints(folder)
            self.assertEqual(set(selected), {"cbm"})
            self.assertIn("blackbox", problems)

    def test_refuses_ambiguous_fold_selection(self):
        with TemporaryDirectory() as temporary:
            folder = Path(temporary)
            (folder / "blackbox_fold_1.pt").touch()
            (folder / "blackbox_fold_2.pt").touch()
            (folder / "cbm_fold_1.pt").touch()
            selected, problems = discover_checkpoints(folder)
            self.assertEqual(set(selected), {"cbm"})
            self.assertIn("Multiple checkpoints", problems["blackbox"])

    def test_missing_folder_is_reported(self):
        with TemporaryDirectory() as temporary:
            selected, problems = discover_checkpoints(Path(temporary) / "missing")
            self.assertEqual(selected, {})
            self.assertEqual(set(problems), {"blackbox", "cbm"})


class ModelDownloadTests(unittest.TestCase):
    def test_downloads_both_models_to_expected_names(self):
        requested_ids = []

        def fake_download(*, id, output, quiet):
            requested_ids.append(id)
            Path(output).write_bytes(b"checkpoint" * 200)
            return output

        with TemporaryDirectory() as temporary:
            folder = Path(temporary) / "models"
            outcome = download_models(folder, downloader=fake_download)
            self.assertEqual(set(outcome["downloaded"]), {"blackbox", "cbm"})
            self.assertEqual(outcome["errors"], {})
            self.assertTrue((folder / "best_model.pt").is_file())
            self.assertTrue((folder / "best_cbm_model.pt").is_file())
            self.assertEqual(
                requested_ids,
                [CHECKPOINT_SOURCES[kind][1] for kind in ("blackbox", "cbm")],
            )

    def test_preserves_existing_weights(self):
        def fake_download(*, id, output, quiet):
            Path(output).write_bytes(b"new-checkpoint" * 200)
            return output

        with TemporaryDirectory() as temporary:
            folder = Path(temporary)
            current = folder / "best_model.pt"
            current.write_bytes(b"keep-existing-weights")
            outcome = download_models(folder, downloader=fake_download)
            self.assertEqual(current.read_bytes(), b"keep-existing-weights")
            self.assertIn("blackbox", outcome["skipped"])
            self.assertIn("cbm", outcome["downloaded"])

    def test_failed_download_leaves_no_partial_checkpoint(self):
        def fake_download(*, id, output, quiet):
            Path(output).write_bytes(b"incomplete")
            raise OSError("Network interrupted")

        with TemporaryDirectory() as temporary:
            folder = Path(temporary)
            outcome = download_models(folder, downloader=fake_download)
            self.assertEqual(set(outcome["errors"]), {"blackbox", "cbm"})
            self.assertFalse((folder / "best_model.pt").exists())
            self.assertFalse((folder / "best_cbm_model.pt").exists())
            self.assertEqual(list(folder.glob(".model-download-*")), [])


if __name__ == "__main__":
    unittest.main()
