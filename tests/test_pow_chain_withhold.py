import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from pow_chain_withhold import (
    SelfishMiningCanonicalDAASimulation,
    _ocw_difficulty_theory,
    _sm_difficulty_theory,
    calculate_canonical_daa,
    get_plt,
    plot_canonical_daa_comparison,
    run_canonical_daa_comparison,
    summarize_canonical_daa,
)


class CanonicalDAAFormulaTests(unittest.TestCase):
    def test_formula_matches_pow_simulation_daa(self) -> None:
        difficulty_new = calculate_canonical_daa(
            difficulty_old=1.0,
            epoch_len=2016,
            T=10.0,
            elapsed_time=10080.0,
        )

        self.assertAlmostEqual(difficulty_new, 2.0)

    def test_placeholder_theories_are_isolated_for_easy_replacement(self) -> None:
        self.assertAlmostEqual(_ocw_difficulty_theory(0.8), 0.4)
        self.assertAlmostEqual(_sm_difficulty_theory(0.8), 1.6)


class SelfishCanonicalDAATests(unittest.TestCase):
    def test_one_epoch_counts_exactly_epoch_len_blocks(self) -> None:
        sim = SelfishMiningCanonicalDAASimulation(
            T=10.0,
            p=0.75,
            gamma=0.0,
            seed=1,
            epoch_len=20,
            run_id=0,
        )

        result = sim.run()

        self.assertEqual(result.a_blocks_counted + result.h_blocks_counted, 20)
        self.assertGreaterEqual(result.canonical_len_at_stop, 20)
        self.assertGreater(result.t_total, 0.0)
        self.assertAlmostEqual(
            result.difficulty_new,
            20 * 10.0 / result.t_total,
        )


class CanonicalDAAComparisonTests(unittest.TestCase):
    def test_runner_and_summary_include_ocw_and_sm_for_each_p(self) -> None:
        raw = run_canonical_daa_comparison(
            p_list=[0.55, 0.95],
            T=2.0,
            n_repeats=2,
            epoch_len=12,
            gamma=0.0,
            base_seed=2026,
            jobs=1,
            show_progress=False,
        )
        summary = summarize_canonical_daa(raw)

        self.assertEqual(len(raw), 8)
        self.assertEqual(len(summary), 4)
        self.assertEqual(
            {(row["strategy"], row["p"]) for row in summary},
            {("OCW", 0.55), ("OCW", 0.95), ("SM", 0.55), ("SM", 0.95)},
        )
        for row in summary:
            self.assertEqual(row["runs"], 2)
            self.assertGreater(row["difficulty_new_mean"], 0.0)

    def test_plot_writes_combined_pdf_and_png(self) -> None:
        if get_plt() is None:
            self.skipTest("matplotlib/scienceplots unavailable")
        summary = [
            {
                "strategy": strategy,
                "p": p,
                "difficulty_new_mean": difficulty,
                "difficulty_new_std": 0.01,
            }
            for strategy, p, difficulty in (
                ("OCW", 0.55, 0.8),
                ("OCW", 0.95, 0.95),
                ("SM", 0.55, 0.45),
                ("SM", 0.95, 0.05),
            )
        ]

        with TemporaryDirectory() as tmp_dir:
            figures_dir = Path(tmp_dir)
            plot_canonical_daa_comparison(summary, figures_dir)

            self.assertTrue(
                (figures_dir / "canonical_daa_difficulty_vs_p.pdf").is_file()
            )
            self.assertTrue(
                (figures_dir / "canonical_daa_difficulty_vs_p.png").is_file()
            )


if __name__ == "__main__":
    unittest.main()
