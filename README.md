# PoW withholding simulations

Two research projects on block-withholding attacks in proof-of-work blockchains.

- [`cartel/`](cartel/): a mining cartel with a loyal member and a traitor
  (*A Starter–Follower Model for Collusive Withholding Attacks*).
- [`ocw/`](ocw/): one-block-capped withholding (OCW) compared with selfish
  mining, with and without difficulty adjustment (*One-Block-Capped Withholding*).

`harness.py` holds the parallel Monte Carlo runner and figure style that both
projects share. Run any experiment from the repository root as a module:

```bash
pip install -r requirements.txt
python -m cartel.release        # simulate, save <project>/results/release.csv, plot
python -m cartel.release plot   # replot from the saved CSV only
```
