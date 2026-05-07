import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import plot_orphan_rate_comparison as comparison


class LoadTBWSummaryTests(unittest.TestCase):
    def test_load_tbw_summary_reads_chain_withhold_summary_columns(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            summary_path = Path(tmpdir) / "chain_withhold_summary.csv"
            summary_path.write_text(
                "\n".join(
                    [
                        "p,runs,A_share_mean,A_share_std,orphan_rate_mean,orphan_rate_std,chain_extensions_mean",
                        "0.70,100,0.78,0.01,0.128,0.005,980",
                        "0.55,100,0.57,0.02,0.119,0.004,610",
                    ]
                )
                + "\n",
                encoding="utf-8",
            )

            rows = comparison.load_tbw_summary(summary_path)

            self.assertEqual([row["p"] for row in rows], [0.55, 0.70])
            self.assertAlmostEqual(rows[0]["sim_mean"], 0.119)
            self.assertAlmostEqual(rows[0]["sim_std"], 0.004)
            self.assertAlmostEqual(rows[1]["sim_mean"], 0.128)
            self.assertAlmostEqual(rows[1]["sim_std"], 0.005)


class ResolveTBWSummaryPathTests(unittest.TestCase):
    def test_default_tbw_summary_path_falls_back_to_legacy_chain_withhold_summary(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            default_path = tmp_path / "results" / "chain_withhold_summary.csv"
            legacy_path = tmp_path / "results" / "chain_withhold" / "summary.csv"
            legacy_path.parent.mkdir(parents=True, exist_ok=True)
            legacy_path.write_text("p\n0.55\n", encoding="utf-8")

            with patch.object(comparison, "DEFAULT_TBW_SUMMARY", default_path), patch.object(
                comparison, "LEGACY_TBW_SUMMARY", legacy_path
            ):
                resolved = comparison.resolve_tbw_summary_path(str(default_path))

            self.assertEqual(resolved, legacy_path)

    def test_explicit_non_default_tbw_summary_path_is_not_rewritten(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            custom_path = tmp_path / "custom" / "tbw.csv"
            default_path = tmp_path / "results" / "chain_withhold_summary.csv"
            legacy_path = tmp_path / "results" / "chain_withhold" / "summary.csv"

            with patch.object(comparison, "DEFAULT_TBW_SUMMARY", default_path), patch.object(
                comparison, "LEGACY_TBW_SUMMARY", legacy_path
            ):
                resolved = comparison.resolve_tbw_summary_path(str(custom_path))

            self.assertEqual(resolved, custom_path)


class MainSmokeTests(unittest.TestCase):
    def test_main_reads_chain_withhold_summary_and_writes_plot_data(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            tbw_summary = tmp_path / "chain_withhold_summary.csv"
            selfish_summary = tmp_path / "selfish_summary.csv"
            output_base = tmp_path / "figures" / "comparison"
            plot_data = tmp_path / "figures" / "plot_data.csv"

            tbw_summary.write_text(
                "\n".join(
                    [
                        "p,runs,A_share_mean,A_share_std,orphan_rate_mean,orphan_rate_std,chain_extensions_mean",
                        "0.55,100,0.57,0.02,0.119,0.004,610",
                        "0.70,100,0.78,0.01,0.128,0.005,980",
                    ]
                )
                + "\n",
                encoding="utf-8",
            )
            selfish_summary.write_text(
                "\n".join(
                    [
                        "p,mean_orphan_rate,std_orphan_rate",
                        "0.55,0.21,0.01",
                        "0.70,0.29,0.02",
                    ]
                )
                + "\n",
                encoding="utf-8",
            )

            argv = [
                "plot_orphan_rate_comparison.py",
                "--simulation-summary",
                str(tbw_summary),
                "--selfish-summary",
                str(selfish_summary),
                "--output",
                str(output_base),
                "--plot-data",
                str(plot_data),
            ]
            with patch("sys.argv", argv):
                comparison.main()

            self.assertTrue(output_base.with_suffix(".png").exists())
            self.assertTrue(output_base.with_suffix(".pdf").exists())
            self.assertTrue(plot_data.exists())


if __name__ == "__main__":
    unittest.main()
