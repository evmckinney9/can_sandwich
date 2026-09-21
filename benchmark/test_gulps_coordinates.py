"""Cross-check benchmark spectra against the optional public GULPS matrix API."""

from __future__ import annotations

import unittest

import numpy as np

import run as benchmark

try:
    from gulps import LocalEquivalenceClass
except ImportError:
    LocalEquivalenceClass = None


def monodromy(weyl: np.ndarray) -> np.ndarray:
    """Invert the documented pairwise-sum Weyl chart without quantization."""
    a, b, c = weyl
    return np.array([(a + b - c) / 2, (a - b + c) / 2, (-a + b + c) / 2])


@unittest.skipIf(
    LocalEquivalenceClass is None, "optional GULPS package is not installed"
)
class GulpsCoordinateTests(unittest.TestCase):
    """Dogfood public canonical matrices without changing benchmark inputs."""

    def test_spectrum_matches_public_matrix_magic_gram(self) -> None:
        """Check signs, phase factor, and ordering through actual canonical matrices."""
        magic = np.array(
            [[1, 0, 0, 1j], [0, 1j, 1, 0], [0, 1j, -1, 0], [1, 0, 0, -1j]],
            dtype=complex,
        ) / np.sqrt(2)
        points = [
            [0, 0, 0],
            [1, 0, 0],
            [0.5, 0.5, 0.5],
            [0.5, 0.25, 0],
            [0.412345678901, 0.231234567891, 0.112345678901],
            [0.7, 0.2, -0.1],
        ]
        for point in points:
            with self.subTest(weyl=point):
                gate = LocalEquivalenceClass(point)
                # The class constructor quantizes. Compare with its represented
                # coordinates here, never substitute them for raw corpus rows.
                represented = monodromy(gate.weyl)
                frame = magic.conj().T @ gate.matrix @ magic
                gram = frame.T @ frame
                np.testing.assert_allclose(
                    gram, np.diag(benchmark.spectrum(represented)), atol=2e-15, rtol=0
                )

    def test_central_lift_matches_public_reflected_matrix(self) -> None:
        """Check the second target orientation using the public matrix constructor."""
        gate = LocalEquivalenceClass([0.5, 0.25, 0.125])
        w = gate.weyl
        reflected = LocalEquivalenceClass([1 - w[0], w[1], -w[2]])
        original = benchmark.spectrum(monodromy(w))
        reflected_roots = np.linalg.eigvals(reflected.matrix @ reflected.matrix)
        residual = np.min(
            np.max(
                np.abs(reflected_roots[None, :] + original[benchmark.PERMUTATIONS]),
                axis=1,
            )
        )
        self.assertLess(residual, 2e-15)

    def test_class_quantization_must_not_replace_raw_corpus_coordinates(self) -> None:
        """Retain a near-boundary displacement that the public class rounds away."""
        raw = monodromy(np.array([0.5, 0.25, 3e-15]))
        before = raw.tobytes()
        w = np.array([raw[0] + raw[1], raw[0] + raw[2], raw[1] + raw[2]])
        gate = LocalEquivalenceClass(w)
        represented = monodromy(gate.weyl)
        self.assertGreater(np.max(np.abs(raw - represented)), 0)
        self.assertGreater(
            np.max(np.abs(benchmark.spectrum(raw) - benchmark.spectrum(represented))),
            1e-15,
        )
        self.assertEqual(raw.tobytes(), before)


if __name__ == "__main__":
    unittest.main()
