import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

from pow_simulation import (
    Block,
    SCENARIO3_N_VALUES,
    SCENARIO4_N_VALUES,
    SCENARIO4_P_VALUES,
    SelfishMiningDAASimulation,
    TimeCheckpointResult,
    TBWSimulation,
    build_fixed_time_ratio_summary_rows,
    plot_scenario3,
    scenario4,
    scenario4_selfish,
)


class FixedTimeSummaryTests(unittest.TestCase):
    def test_build_fixed_time_ratio_summary_rows_uses_fixed_time_baseline_ratio(self) -> None:
        checkpoint_results = [
            TimeCheckpointResult(
                scenario="scenario3_daa_by_time",
                p=0.55,
                n=1,
                run_id=0,
                seed=1,
                t_checkpoint=20160.0,
                A_blocks_canonical=1300,
                H_blocks_canonical=700,
                canonical_len=1200,
                final_difficulty_so_far=0.9,
                num_epochs_completed_so_far=1,
            ),
            TimeCheckpointResult(
                scenario="scenario3_daa_by_time",
                p=0.55,
                n=1,
                run_id=1,
                seed=2,
                t_checkpoint=20160.0,
                A_blocks_canonical=1500,
                H_blocks_canonical=500,
                canonical_len=1200,
                final_difficulty_so_far=0.8,
                num_epochs_completed_so_far=1,
            ),
        ]

        rows = build_fixed_time_ratio_summary_rows(
            checkpoint_results=checkpoint_results,
            p_values=[0.55],
            n_values=[1],
            metric_name="A_blocks_canonical_fixedtime_n_round_over_pn2016",
            runs=2,
        )

        self.assertEqual(len(rows), 1)
        expected_baseline = 0.55 * 1 * 2016
        expected_blocks_mean = (1300 + 1500) / 2.0
        expected_mean = expected_blocks_mean / expected_baseline
        self.assertEqual(rows[0]["metric"], "A_blocks_canonical_fixedtime_n_round_over_pn2016")
        self.assertEqual(rows[0]["physical_time"], 2016 * 10.0)
        self.assertEqual(rows[0]["attacker_blocks_mean"], expected_blocks_mean)
        self.assertEqual(rows[0]["normalization_blocks"], expected_baseline)
        self.assertAlmostEqual(rows[0]["metric_mean"], expected_mean)

    def test_build_fixed_time_ratio_summary_rows_uses_selfish_checkpoint_ratio_too(self) -> None:
        checkpoint_results = [
            TimeCheckpointResult(
                scenario="scenario4_selfish_public_daa_by_time",
                p=0.65,
                n=2,
                run_id=0,
                seed=1,
                t_checkpoint=40320.0,
                A_blocks_canonical=2100,
                H_blocks_canonical=900,
                canonical_len=3000,
                final_difficulty_so_far=0.9,
                num_epochs_completed_so_far=2,
            ),
            TimeCheckpointResult(
                scenario="scenario4_selfish_public_daa_by_time",
                p=0.65,
                n=2,
                run_id=1,
                seed=2,
                t_checkpoint=40320.0,
                A_blocks_canonical=2300,
                H_blocks_canonical=800,
                canonical_len=3100,
                final_difficulty_so_far=0.8,
                num_epochs_completed_so_far=2,
            ),
        ]

        rows = build_fixed_time_ratio_summary_rows(
            checkpoint_results=checkpoint_results,
            p_values=[0.65],
            n_values=[2],
            metric_name="A_blocks_canonical_fixedtime_n_round_over_pn2016",
            runs=2,
        )

        self.assertEqual(len(rows), 1)
        baseline = 0.65 * 2 * 2016
        expected_blocks_mean = (2100.0 + 2300.0) / 2.0
        expected_mean = expected_blocks_mean / baseline
        self.assertEqual(rows[0]["metric"], "A_blocks_canonical_fixedtime_n_round_over_pn2016")
        self.assertEqual(rows[0]["physical_time"], 4032 * 10.0)
        self.assertEqual(rows[0]["attacker_blocks_mean"], expected_blocks_mean)
        self.assertEqual(rows[0]["normalization_blocks"], baseline)
        self.assertAlmostEqual(rows[0]["metric_mean"], expected_mean)

    def test_build_fixed_time_ratio_uses_configured_epoch_length(self) -> None:
        checkpoint = TimeCheckpointResult(
            scenario="test",
            p=0.5,
            n=2,
            run_id=0,
            seed=1,
            t_checkpoint=400.0,
            A_blocks_canonical=100,
            H_blocks_canonical=100,
            canonical_len=200,
            final_difficulty_so_far=1.0,
            num_epochs_completed_so_far=2,
        )

        rows = build_fixed_time_ratio_summary_rows(
            checkpoint_results=[checkpoint],
            p_values=[0.5],
            n_values=[2],
            metric_name="custom_epoch_ratio",
            runs=1,
            epoch_len=100,
            T=2.0,
        )

        self.assertEqual(rows[0]["physical_time"], 400.0)
        self.assertEqual(rows[0]["attacker_blocks_mean"], 100.0)
        self.assertEqual(rows[0]["normalization_blocks"], 100.0)
        self.assertAlmostEqual(rows[0]["metric_mean"], 1.0)


class TimeCheckpointCaptureTests(unittest.TestCase):
    def test_capture_pending_checkpoints_uses_state_before_future_event(self) -> None:
        sim = TBWSimulation(
            T=10.0,
            p=0.55,
            seed=1,
            mode="by_time",
            t_end=30.0,
            target_blocks=None,
            enable_daa=False,
            epoch_len=1,
            scenario="test",
            run_id=0,
            checkpoint_times=[10.0],
        )

        sim._publish_block(Block(id=1, parent_id=0, height=1, miner="A", t_publish=3.0))
        sim._publish_block(Block(id=2, parent_id=1, height=2, miner="H", t_publish=7.0))

        sim._capture_pending_checkpoints_before(12.0)

        self.assertEqual(len(sim.checkpoint_results), 1)
        checkpoint = sim.checkpoint_results[0]
        self.assertEqual(checkpoint.n, 1)
        self.assertEqual(checkpoint.A_blocks_canonical, 1)
        self.assertEqual(checkpoint.H_blocks_canonical, 1)
        self.assertEqual(checkpoint.canonical_len, 2)
        self.assertAlmostEqual(checkpoint.t_checkpoint, 10.0)


class DifficultyAdjustmentBasisTests(unittest.TestCase):
    def _make_sim(self, basis: str) -> TBWSimulation:
        sim = TBWSimulation(
            T=10.0,
            p=0.55,
            seed=1,
            mode="by_time",
            t_end=100.0,
            target_blocks=None,
            enable_daa=True,
            epoch_len=3,
            scenario="test",
            run_id=0,
            n_value=1,
            daa_count_basis=basis,
        )
        return sim

    def _publish(
        self,
        sim: TBWSimulation,
        block_id: int,
        parent_id: int,
        height: int,
        t_publish: float,
    ) -> None:
        sim._publish_block(
            Block(
                id=block_id,
                parent_id=parent_id,
                height=height,
                miner="A" if block_id % 2 else "H",
                t_publish=t_publish,
            )
        )

    def test_public_basis_adjusts_when_published_blocks_reach_epoch_len(self) -> None:
        sim = self._make_sim("public")

        self._publish(sim, 1, 0, 1, 5.0)
        self._publish(sim, 2, 1, 2, 10.0)
        self._publish(sim, 3, 0, 1, 12.0)

        sim._maybe_adjust_difficulty()

        self.assertEqual(sim.epochs_completed, 1)
        self.assertAlmostEqual(sim.difficulty, (3 * 10.0) / 12.0)

    def test_canonical_basis_waits_for_canonical_chain_epoch_len(self) -> None:
        sim = self._make_sim("canonical")

        self._publish(sim, 1, 0, 1, 5.0)
        self._publish(sim, 2, 1, 2, 10.0)
        self._publish(sim, 3, 0, 1, 12.0)

        sim._maybe_adjust_difficulty()

        self.assertEqual(sim.epochs_completed, 0)
        self.assertAlmostEqual(sim.difficulty, 1.0)


class SelfishDifficultyAdjustmentTests(unittest.TestCase):
    def test_majority_public_daa_counts_work_at_each_mining_time(self) -> None:
        sim = SelfishMiningDAASimulation(
            T=10.0,
            p=0.75,
            gamma=0.0,
            seed=1,
            mode="by_time",
            t_end=100.0,
            enable_daa=True,
            epoch_len=3,
            scenario="selfish-public-test",
            run_id=0,
            n_value=1,
            daa_count_basis="public",
        )

        for t_mined, handler in (
            (10.0, sim._handle_mine_A),
            (15.0, sim._handle_mine_H),
            (20.0, sim._handle_mine_A),
        ):
            sim.t = t_mined
            handler()
            sim._maybe_adjust_difficulty()

        self.assertEqual(sim.epochs_completed, 1)
        self.assertAlmostEqual(sim.epoch_stats[0].t_end, 20.0)
        self.assertAlmostEqual(sim.difficulty, (3 * 10.0) / 20.0)
        self.assertEqual(sim.epoch_stats[0].a_blocks_counted, 2)
        self.assertEqual(sim.epoch_stats[0].h_blocks_counted, 1)

    def test_majority_attacker_reveals_a_full_private_epoch_and_triggers_daa(self) -> None:
        sim = SelfishMiningDAASimulation(
            T=10.0,
            p=0.75,
            gamma=0.0,
            seed=1,
            mode="by_time",
            t_end=100.0,
            enable_daa=True,
            epoch_len=3,
            scenario="selfish-majority-test",
            run_id=0,
            n_value=1,
            daa_count_basis="canonical",
        )

        sim.t = 10.0
        sim._handle_mine_A()
        sim.t = 15.0
        sim._handle_mine_H()
        sim.t = 20.0
        sim._handle_mine_A()

        checkpoint = sim._build_checkpoint_result(20.0)
        self.assertEqual(checkpoint.A_blocks_canonical, 2)
        self.assertEqual(checkpoint.H_blocks_canonical, 0)

        sim.t = 40.0
        sim._handle_mine_A()
        sim._maybe_adjust_difficulty()

        self.assertEqual(sim.epochs_completed, 1)
        self.assertAlmostEqual(sim.epoch_stats[0].t_end, 40.0)
        self.assertAlmostEqual(sim.difficulty, 0.75)
        self.assertEqual(
            [sim.blocks_by_id[bid].miner for bid in sim.canonical_chain_ids],
            ["A", "A", "A"],
        )

    def test_canonical_basis_waits_for_canonical_chain_epoch_len(self) -> None:
        sim = SelfishMiningDAASimulation(
            T=10.0,
            p=0.55,
            gamma=0.0,
            seed=1,
            mode="by_time",
            t_end=100.0,
            enable_daa=True,
            epoch_len=3,
            scenario="selfish-test",
            run_id=0,
            n_value=1,
            daa_count_basis="canonical",
        )

        sim._publish_block(Block(id=1, parent_id=0, height=1, miner="A", t_publish=5.0))
        sim._publish_block(Block(id=2, parent_id=1, height=2, miner="H", t_publish=10.0))
        sim._publish_block(Block(id=3, parent_id=0, height=1, miner="A", t_publish=12.0))

        sim._maybe_adjust_difficulty()

        self.assertEqual(sim.epochs_completed, 0)
        self.assertAlmostEqual(sim.difficulty, 1.0)

    def test_public_basis_adjusts_when_published_blocks_reach_epoch_len(self) -> None:
        sim = SelfishMiningDAASimulation(
            T=10.0,
            p=0.55,
            gamma=0.0,
            seed=1,
            mode="by_time",
            t_end=100.0,
            enable_daa=True,
            epoch_len=3,
            scenario="selfish-test",
            run_id=0,
            n_value=1,
            daa_count_basis="public",
        )

        sim._publish_block(Block(id=1, parent_id=0, height=1, miner="A", t_publish=5.0))
        sim._publish_block(Block(id=2, parent_id=1, height=2, miner="H", t_publish=10.0))
        sim._publish_block(Block(id=3, parent_id=0, height=1, miner="A", t_publish=12.0))

        sim._maybe_adjust_difficulty()

        self.assertEqual(sim.epochs_completed, 1)
        self.assertAlmostEqual(sim.difficulty, (3 * 10.0) / 12.0)


class Scenario4NormalizationTests(unittest.TestCase):
    def test_ocw_keeps_attacker_hashrate_normalization(self) -> None:
        with TemporaryDirectory() as tmp_dir:
            _, _, summary_rows = scenario4(
                T=2.0,
                runs=1,
                epoch_len=4,
                base_seed=2026,
                results_root=Path(tmp_dir),
                jobs=1,
                show_progress=False,
            )

            for row in summary_rows:
                expected_baseline = float(row["p"]) * int(row["n"]) * 4
                self.assertEqual(row["normalization_blocks"], expected_baseline)
                self.assertEqual(
                    row["metric"], "A_blocks_canonical_fixedtime_over_pn4"
                )
                self.assertAlmostEqual(
                    row["metric_mean"],
                    row["attacker_blocks_mean"] / expected_baseline,
                )

    def test_uses_scenario3_physical_time_and_honest_chain_normalization(self) -> None:
        with TemporaryDirectory() as tmp_dir:
            _, _, summary_rows = scenario4_selfish(
                T=2.0,
                runs=1,
                epoch_len=4,
                base_seed=2026,
                gamma=0.0,
                results_root=Path(tmp_dir),
                jobs=1,
                show_progress=False,
            )

            self.assertEqual(
                len(summary_rows),
                len(SCENARIO4_P_VALUES) * len(SCENARIO4_N_VALUES),
            )
            self.assertEqual(
                sorted({int(row["n"]) for row in summary_rows}),
                SCENARIO4_N_VALUES,
            )
            for row in summary_rows:
                expected_baseline = int(row["n"]) * 4
                self.assertEqual(row["physical_time"], int(row["n"]) * 4 * 2.0)
                self.assertEqual(row["normalization_blocks"], expected_baseline)
                self.assertEqual(
                    row["metric"], "A_blocks_canonical_fixedtime_over_n4"
                )
                self.assertAlmostEqual(
                    row["metric_mean"],
                    row["attacker_blocks_mean"] / expected_baseline,
                )


class Scenario3PlotProtocolTests(unittest.TestCase):
    def test_uses_requested_n_values_and_two_horizontal_panels(self) -> None:
        self.assertEqual(SCENARIO3_N_VALUES, [1, 2, 3, 5])
        with patch("pow_simulation.plot_strategy_comparison") as plot_comparison:
            plot_scenario3([], [], Path("unused"))

        self.assertEqual(plot_comparison.call_args.kwargs["max_columns"], 2)


if __name__ == "__main__":
    unittest.main()
