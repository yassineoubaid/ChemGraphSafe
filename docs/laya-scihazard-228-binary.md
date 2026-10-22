# Laya versus Jev: controlled 228-case SciHazard binary evaluation

This is a **runnable experiment definition**, not a new model-performance claim.
The current Laya adapter branch `codex/laya-adapter-local` is the starting point.
This experiment does not modify the older three-class / 100-case Laya pilot.

## Exact sample definition

Pinned SciHazard commit: `f8809e5af23495e469dfc53e99973f74031fcb11`.

The importer has 50 benign and 200 harmful-split Chemistry records; it
quarantines 22 harmful-split examples with conflicting native labels. The remaining
**50 benign + 178 harmful = 228** independent cases are selected when `count = 228`.
The class-stratified 20% split is expected to give **46 development** (10 benign /
36 harmful) and **182 evaluation** (40 benign / 142 harmful).

Matching Thang's counts does **not** prove we have his exact case IDs, policy or
run settings. For a paired comparison, request his immutable `freeze.json` and
verify the dataset hash and individual case identities. When compatible, set
`data.inherited_freeze` in the new config *before preparing* data. Never
overwrite an existing frozen experiment.

## Setup and data preparation

Use your Linux/WSL2 ChemGraphSafe checkout with the Laya adapter installed.
Start the separately managed Laya 0.4.13 service in another WSL terminal
as documented under **Run Laya locally** in the project README. The service
must expose the `english` checkpoint at port 18000 with the pinned revision
`55cf4c4ebb4ebe31b2550e8bdf3bd21b99753851`.

```bash
uv sync --locked --extra dev
curl -fsS http://127.0.0.1:18000/health
uv run chemgraphsafe data prepare --config configs/laya-scihazard-228-binary.toml
uv run chemgraphsafe data freeze --config configs/laya-scihazard-228-binary.toml
```

These preparation steps are offline except for the pinned public source-data
download; they do not call a decision model. Check the preparation counts
and the frozen case IDs/splits before inference.

## Development-only option-order diagnostic

```bash
uv run python scripts/laya_order_diagnostic.py \
  --config configs/laya-scihazard-228-binary.toml \
  --output .local/results/laya-order-dev-v1.json
```

The diagnostic sends each development-only request twice to **local Laya**,
first with ALLOW/BLOCK, then BLOCK/ALLOW. It reports decision stability,
both orders' class rates and failed/abstained cases. Probabilities are preserved
by label name. Local output contains example IDs, labels and predictions, not
hazardous request text. This is an **exploratory protocol diagnostic**, not the
confirmatory holdout score. Use a separate output name if repeating the test.

For a short check, append `--limit 6` and use a unique output name.
The default 46 x 2 diagnostic needs **92 requests**.

## Full Laya binary run

```bash
uv run chemgraphsafe smoke --config configs/laya-scihazard-228-binary.toml
uv run chemgraphsafe evaluate --config configs/laya-scihazard-228-binary.toml
uv run chemgraphsafe report --run runs/laya-scihazard-228-binary-v1
```

Read `report.md`, `comparison.csv`, and `metrics.json` under the new run
directory. **Evaluation-only (182 cases) should be reported separately** from
all 228 cases; don't silently mix metrics across splits.

## Published Jev numbers from Slack (not a paired verified rerun)

Thang reported binary Jev classification on the 228 eligible chemistry cases:
- Development: 32/46 correct.
- Evaluation: 120/182 correct.
- All cases: 152/228 correct (66.7%).
- Benign: 50/50 allowed.
- Harmful: 102/178 blocked; 76/178 allowed.

Interpret Laya against these as an **aggregate-only comparison** until the
exact Jev freeze, per-case predictions, prompt/policy and model identity have
been verified. Do not interpret differences as model ranking or perform a
paired statistical test based only on matching totals.

## Budget and evidence boundaries

The shared ledger `.local/state/pilot-budget.sqlite` allows only
**500 cumulative attempts** across all past runs, retries, diagnostic requests
and smoke checks. A full dev rotation plus full 228-case Laya evaluation
plus 5-case smoke test requires at least **325 NEW attempts** (92 + 228 + 5),
possibly more with retries. Check remaining capacity before starting.
Do not erase or replace the ledger just to bypass the cap; obtain an approved
extension if needed.

No model weights, raw requests, credentials or claimed new results are included.
A canonical option order makes inputs reproducible but **does not remove**
model position bias. If the diagnostic finds order dependence, evaluate a
separately versioned balanced-order inference method with an untouched holdout
before claiming improved model performance.
