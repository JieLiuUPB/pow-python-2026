import unittest
from dataclasses import replace

from pow_collusion import (
    BetrayConfig,
    CollusionSimulation,
    ScenarioConfig,
    SimConfig,
    build_scenarios,
    parse_pool_spec,
)


class CollusionSimulationTests(unittest.TestCase):
    def setUp(self) -> None:
        self.config = SimConfig(
            T=10.0,
            gamma=0.0,
            runs=1,
            target_blocks_long=10,
            betray_on_nth_opportunity=1,
            betray_start_height=0,
            q=1.0,
            betray_threshold=10,
        )
        self.pools = parse_pool_spec("b=0.33,s=0.33,h=0.34")

    def make_simulation(self, scenario: ScenarioConfig) -> CollusionSimulation:
        return CollusionSimulation(
            experiment="three",
            sim_config=self.config,
            pools=self.pools,
            scenario=scenario,
            run_id=0,
            seed=1,
            max_events=100,
        )

    def test_tolerated_betrayal_publishes_both_blocks(self) -> None:
        simulation = self.make_simulation(
            ScenarioConfig(
                name="BetrayTolerated",
                mode="long",
                initial_members=("b", "s"),
                break_rule="none",
                betray=BetrayConfig(
                    mode="prob",
                    traitor_id="s",
                    betray_start_height=0,
                    q=1.0,
                ),
            )
        )

        simulation._on_member_first_block(miner_id="b", parent_id=0)
        simulation._on_private_mine(miner_id="s")

        chain = simulation.reconstruct_chain()
        miners = [simulation.blocks_by_id[block_id].miner_id for block_id in chain]
        self.assertEqual(miners, ["b", "s"])
        self.assertEqual(simulation.controller.state, "IDLE")
        self.assertEqual(simulation.betray_count, 1)

    def test_break_rule_depends_on_remaining_cartel_power(self) -> None:
        pools = parse_pool_spec("b=0.49,s=0.11,h=0.40")
        scenarios = build_scenarios(
            sim_config=self.config,
            experiment="three",
            pools=pools,
            initial_members=("b", "s"),
            traitor_id="s",
        )

        self.assertEqual(scenarios[1].break_rule, "dissolve_immediate")
        self.assertEqual(scenarios[2].break_rule, "dissolve_threshold")

    def test_gamma_splits_non_cartel_hashrate_during_race(self) -> None:
        config = replace(self.config, gamma=0.25)
        scenario = ScenarioConfig(
            name="AlwaysCartel",
            mode="long",
            initial_members=("b", "s"),
            break_rule="none",
            betray=BetrayConfig(mode="none", traitor_id="s"),
        )
        simulation = CollusionSimulation(
            experiment="three",
            sim_config=config,
            pools=self.pools,
            scenario=scenario,
            run_id=0,
            seed=1,
            max_events=100,
        )
        simulation._on_member_first_block(miner_id="b", parent_id=0)
        simulation._on_public_mine(miner_id="h", target_tip_id=0)

        honest_processes = [
            process
            for process in simulation.build_mining_processes()
            if process.pool_id == "h"
        ]
        rates_by_target = {
            process.target_kind: process.lambda_rate for process in honest_processes
        }
        self.assertAlmostEqual(rates_by_target["private"], 0.34 / 10.0 * 0.25)
        self.assertAlmostEqual(rates_by_target["public"], 0.34 / 10.0 * 0.75)


if __name__ == "__main__":
    unittest.main()
