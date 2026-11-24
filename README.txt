# Reproduction code: Cache-line-aware low-rank decomposition with
# density-peak clustered quantization for efficient DNN inference on edge devices
#
# Anonymized repository accompanying the JSA submission.
#
## Layout
- Core CIFAR-10 pipeline: train_baseline.py -> compose_exp.py (SVD decomposition)
  -> kd_exp.py (KD polish) -> fair_cifar.py (baselines incl. DepGraph)
  -> multiseed_cifar.py (4-seed repetition) -> kd_only.py (KD-only control)
- ImageNet-100: download_imagenet100.py (clane9/imagenet-100 mirror, CMC class
  list) -> train_imagenet100.py / run_imagenet100.py / diag_in100.py
- Latency: latency_interleaved.py (interleaved CPU benchmark)
- Staging simulation: dma_staging_sim.py (three transaction models A/B/C),
  round4_fixes.py (per-layer aggregation; validated against the N=256 table)
- Quantization: round2_fixes.py (uint8-index FCQ variants, compression costs)
- Figures: make_figs_round2.py / make_figs_round4.py
- JSON files are the raw results referenced in the manuscript.

## Environment
Python 3.12, PyTorch 2.x (CPU sufficient except training; CIFAR-10/100 and
ImageNet-100 expected under ./data). Result JSONs use metric names as
documented in each producing script.
