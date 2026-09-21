"""Check independent targeted-case generation and its spectral chart."""

from pathlib import Path
import tempfile
import unittest

import numpy as np

from generate_targeted_cases import (
    PAIRS,
    SCALES,
    fold_spectrum,
    generate,
    monodromy,
    write_corpus,
)
from run import spectrum


class TargetedCaseTests(unittest.TestCase):
    """Verify planted witnesses and intended family coverage without a solver."""

    @classmethod
    def setUpClass(cls) -> None:
        """Generate and strictly validate all deterministic cases once."""
        cls.rows, cls.witnesses, cls.records, cls.metrics = generate()

    def test_all_witnesses_pass_strict_independent_check(self) -> None:
        """Require stringent matrix certificates for every generated row."""
        self.assertEqual(self.rows.shape, (1455, 3, 3))
        self.assertEqual(self.witnesses.shape, (1455, 4, 4))
        self.assertLess(max(self.metrics.values()), 1e-12)
        for record in self.records:
            self.assertLess(max(record["witness_metrics"].values()), 1e-12)

    def test_supports_modes_and_scales_are_independent(self) -> None:
        """Cover all support placements and the full near-local scale ladder."""
        families = {}
        for record in self.records:
            families.setdefault(record["family"], []).append(record)
        self.assertEqual(
            {tuple(row["plane"]) for row in families["one_plane"]}, set(PAIRS)
        )
        self.assertEqual(
            {row["block"] for row in families["two_plus_two_support"]}, {0, 1, 2}
        )
        self.assertEqual(
            {row["fixed"] for row in families["one_plus_three_support"]}, set(range(4))
        )
        for plane in PAIRS:
            self.assertEqual(
                {
                    row["scale"]
                    for row in families["near_local_scale"]
                    if tuple(row["plane"]) == plane
                },
                set(SCALES),
            )
        self.assertEqual({row["target_central_sign"] for row in self.records}, {-1, 1})
        for index, record in enumerate(self.records):
            if record["family"] == "one_plus_three_support":
                fixed = record["fixed"]
                frame = self.witnesses[index]
                self.assertEqual(abs(frame[fixed, fixed]), 1.0)
                self.assertEqual(np.count_nonzero(frame[fixed, :]), 1)
                self.assertEqual(np.count_nonzero(frame[:, fixed]), 1)

    def test_direct_input_coordinates_are_not_quantized(self) -> None:
        """Keep planted input monodromy unchanged through target extraction."""
        expected = monodromy(np.array([0.41, 0.23, 0.07]))
        np.testing.assert_array_equal(self.rows[0, 0], expected)
        offsets = [
            row[0, 0] - row[0, 1]
            for row, record in zip(self.rows, self.records)
            if record["family"] == "anisotropic_pair22_inputs"
            and record["left_scale"] == 1e-14
        ]
        self.assertTrue(offsets)
        self.assertTrue(all(value != 0 for value in offsets))

    def test_input_gap_and_local_angle_form_a_cartesian_product(self) -> None:
        """Keep every multiplicity, role, gap, and angle combination present."""
        rows = [
            record
            for record in self.records
            if record["family"] == "input_gap_local_angle_product"
        ]
        self.assertEqual(len(rows), 320)
        for partition in ("pair211", "pair22", "triple31", "scalar4"):
            for role in ("left", "right"):
                for mode in ("one_plane", "dense"):
                    self.assertEqual(
                        {
                            (row["input_gap"], row["local_angle"])
                            for row in rows
                            if row["partition"] == partition
                            and row["role"] == role
                            and row["mode"] == mode
                        },
                        {
                            (gap, angle)
                            for gap in (0.0, 1e-14, 1e-10, 1e-7, 1e-4)
                            for angle in (0.0, 1e-12, 1e-8, 0.31)
                        },
                    )

    def test_output_refuses_bad_suffix_and_existing_sidecars(self) -> None:
        """Protect locked artifacts before starting corpus generation."""
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory)
            with self.assertRaises(ValueError):
                write_corpus(base / "cases.txt")
            for suffix in (".npy", ".json", ".witnesses.npz"):
                artifact = base / ("cases" + suffix)
                artifact.write_bytes(b"original")
                with self.assertRaises(FileExistsError):
                    write_corpus(base / "cases.npy")
                self.assertEqual(artifact.read_bytes(), b"original")
                artifact.unlink()

    def test_phase_folding_preserves_spectra_and_orders(self) -> None:
        """Check phase lifts and central signs on generic and repeated spectra."""
        rng = np.random.default_rng(731)
        turns = [np.zeros(4), np.array([0.19, 0.19, -0.19, -0.19])]
        for _ in range(50):
            first = rng.uniform(-0.5, 0.5, 3)
            turns.append(np.append(first, -sum(first)))
        for phase in turns:
            roots = np.exp(2j * np.pi * phase)
            m, ordering, sign = fold_spectrum(roots)
            np.testing.assert_allclose(
                spectrum(m), sign * roots[ordering], atol=4e-15, rtol=0
            )
            w0, w1, w2 = m[0] + m[1], m[0] + m[2], m[1] + m[2]
            self.assertGreaterEqual(min(w0 - w1, w1 - w2, w2, 1 - w0 - w1), -2e-13)


if __name__ == "__main__":
    unittest.main(verbosity=2)
