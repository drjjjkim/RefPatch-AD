"""Run all three scoring paths through the command-line interface."""

import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np


class CachedScoringTest(unittest.TestCase):
    def test_three_maps_and_mask(self):
        rng = np.random.default_rng(20261112)

        def unit_features(patches):
            features = rng.normal(size=(3, patches, 384))
            return (features / np.linalg.norm(features, axis=2, keepdims=True)).astype(np.float32)

        bank = unit_features(32)
        query = unit_features(4)
        mask = np.array([True, False, True, False])
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for path, features, angles, flags in (
                (root / "reference.npz", bank, list(range(0, 360, 45)), np.ones(32, dtype=bool)),
                (root / "query.npz", query, [0], mask),
            ):
                np.savez(path, layer_features=features, grid=[2, 2], angles=angles,
                         layers_one_based=[4, 8, 12], query_mask=flags)
            out = root / "prediction.npz"
            subprocess.run([sys.executable, str(Path(__file__).with_name("score_cached.py")),
                            "--reference", str(root / "reference.npz"), "--query",
                            str(root / "query.npz"), "--out", str(out)], check=True)
            direct = ((query[:, mask, None].astype(np.float64) - bank[:, None]) ** 2).sum(axis=-1) / 2
            minima = direct.min(axis=-1).astype(np.float32)
            with np.load(out, allow_pickle=False) as result:
                np.testing.assert_array_equal(result["query_mask"], mask)
                np.testing.assert_array_equal(result["distance_last_identity"].ravel()[mask], minima[2])
                np.testing.assert_array_equal(result["distance_score_mean"].ravel()[mask],
                                              minima.mean(axis=0, dtype=np.float64).astype(np.float32))
                for name in ("last_identity", "score_mean", "reference_weighted"):
                    values = result[f"distance_{name}"].ravel()
                    self.assertEqual(values.dtype, np.float32)
                    self.assertTrue(np.all(values[~mask] == 0))
                weighted = result["distance_reference_weighted"].ravel()[mask]
                self.assertTrue(np.all(weighted >= minima.min(axis=0)))
                self.assertTrue(np.all(weighted <= minima.max(axis=0)))


if __name__ == "__main__":
    unittest.main()
