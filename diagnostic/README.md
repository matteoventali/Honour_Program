# Diagnostic framework

Standalone copy of the multilevel framework used for the step-by-step PBRS
experiments. It includes:

- bilinear interpolation of the level-1 potential;
- reward and potential normalization;
- zero potential on bootstrap-terminal transitions;
- separate ground and abstract discounts;
- JSONL shaping diagnostics.

The configured abstract discount is `0.99`. With `6.75` ground steps per cell,
the ground learner and ground PBRS both use `0.9985121692742382`.

Run the long shaped experiment from the repository root with:

```bash
python3 diagnostic/src/trainer.py \
  --experiment-name timescale-gamma-5000 \
  --episodes 5000 \
  --shaping-frequency every_step \
  --training-potential bilinear \
  --normalize-training-rewards
```

Run the matching no-shaping baseline with the same seed using:

```bash
python3 diagnostic/src/trainer.py \
  --experiment-name baseline-no-shaping-5000 \
  --episodes 5000 \
  --shaping-frequency every_step \
  --training-potential bilinear \
  --normalize-training-rewards \
  --no-shaping
```
