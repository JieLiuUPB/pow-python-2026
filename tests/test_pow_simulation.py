import unittest

from pow_simulation import (
    Block,
    TimeCheckpointResult,
    TBWSimulation,
    build_fixed_time_ratio_summary_rows,
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
        expected_mean = ((1300 / expected_baseline) + (1500 / expected_baseline)) / 2.0
        self.assertEqual(rows[0]["metric"], "A_blocks_canonical_fixedtime_n_round_over_pn2016")
        self.assertAlmostEqual(rows[0]["metric_mean"], expected_mean)


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

    def _publish(self, sim: TBWSimulation, block_id: int, parent_id: int, height: int, t_publish: float) -> None:
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


if __name__ == "__main__":
    unittest.main()
