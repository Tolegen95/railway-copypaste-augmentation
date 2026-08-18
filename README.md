# Mask-Guided Copy-Paste Augmentation for Railway Defect Detection — Reproducibility Package

Code, configuration, and raw per-seed results for the reviewer-requested GPU validation of
"Mask-Guided Copy-Paste Augmentation for Data-Efficient Railway Defect Detection on a Small
Imbalanced Dataset" (CMC-88467-Manuscript-127816-1).

This package accompanies the revised manuscript and supersedes the original 5-epoch CPU pilot
protocol described in Sections 5-6 of the paper. See `docs/GPU_VALIDATION_FINDINGS_AND_REWRITE.md`
for the full write-up of what changed and why.

## What this is

- `src/prepare_mask_guided_copypaste_yolo.py` — builds the mask-guided copy-paste training set
  from the source segmentation dataset (connected-component instance extraction, donor sampling,
  paste-and-feather compositing, mask-to-YOLO conversion). Supports three donor-sampling
  policies (`inverse_sqrt`, `uniform_instance` = Simple Copy-Paste, `uniform_class`) and a
  configurable augmentation volume (`--aug_per_train`).
- `src/full_reviewer_experiments.py` — unified, resumable training/evaluation loop across
  models, augmentation variants, and seeds (baseline, Mosaic, MixUp, CutMix, Random Erasing,
  Simple Copy-Paste, mask-guided copy-paste, plus the two sensitivity variants
  `uniform_class` and `aug2x`). Trains with Ultralytics YOLO to convergence (early stopping)
  and records precision/recall/mAP50/mAP50:95/inference time per run.
- `src/analyze_reviewer_experiments.py` — paired t-tests, Wilcoxon signed-rank tests, and 95%
  confidence intervals for each variant vs. baseline, per model.
- `src/per_class_ap.py` — extracts per-class AP50 from trained weights.
- `src/threshold_calibration.py` — sweeps the confidence threshold and reports each model's
  precision/recall/F1 at its own best-F1 operating point, plus a PR-curve comparison plot.
- `results/` — raw output of the above: `priority_reviewer_gpu_results.csv` (one row per
  model/variant/seed run), `statistics/` (paired comparisons and summary statistics),
  `per_class_ap50_n5.csv`, `threshold_calibration.csv`, and `pr_curve_comparison.png`.

## Environment

Python 3.13, CUDA 12.6-compatible GPU (validated on an NVIDIA RTX 4060 Laptop GPU, 8GB VRAM).

```bash
python -m venv .venv
.venv/Scripts/activate  # or source .venv/bin/activate on Linux
pip install -r requirements.txt
```

## Reproducing the results

1. Build the mask-guided copy-paste dataset (and the two sensitivity variants) from the source
   segmentation dataset (`dataset_seg2`, see the paper's data-availability statement for the
   source-dataset link):

```bash
python src/prepare_mask_guided_copypaste_yolo.py --src dataset_seg2 --out dataset_yolo_mask_copypaste \
    --seed 42 --aug_per_train 1.0 --min_area 20 --sampling inverse_sqrt
python src/prepare_mask_guided_copypaste_yolo.py --src dataset_seg2 --out dataset_yolo_simple_copypaste \
    --seed 42 --aug_per_train 1.0 --min_area 20 --sampling uniform_instance
python src/prepare_mask_guided_copypaste_yolo.py --src dataset_seg2 --out dataset_yolo_uniformclass_copypaste \
    --seed 42 --aug_per_train 1.0 --min_area 20 --sampling uniform_class
python src/prepare_mask_guided_copypaste_yolo.py --src dataset_seg2 --out dataset_yolo_aug2x_copypaste \
    --seed 42 --aug_per_train 2.0 --min_area 20 --sampling inverse_sqrt
```

2. Run the main comparison (YOLOv8n, all augmentation variants, 5 seeds, 100 epochs, GPU):

```bash
python src/full_reviewer_experiments.py --models yolov8n.pt \
    --variants baseline,mosaic,mixup,cutmix,random_erasing,mask_guided \
    --seeds 0,1,2,3,4 --device 0 --epochs 100 --patience 20 --imgsz 640 --batch -1 \
    --project priority_reviewer_gpu --results priority_reviewer_gpu/results.csv
```

3. Run the multi-detector generality check:

```bash
python src/full_reviewer_experiments.py --models yolov8s.pt,yolov9t.pt,yolov10n.pt,yolo11n.pt \
    --variants baseline,mask_guided --seeds 0,1,2,3,4 --device 0 --epochs 100 --patience 20 \
    --imgsz 640 --batch -1 --project priority_reviewer_gpu --results priority_reviewer_gpu/results.csv
```

4. Run the sensitivity analysis (donor-sampling policy and augmentation volume):

```bash
python src/full_reviewer_experiments.py --models yolov8n.pt \
    --variants simple_copy_paste,uniform_class,aug2x --seeds 0,1,2 --device 0 --epochs 100 \
    --patience 20 --imgsz 640 --batch -1 --project priority_reviewer_gpu \
    --results priority_reviewer_gpu/results.csv
```

5. Compute statistics, per-class breakdown, and threshold calibration:

```bash
python src/analyze_reviewer_experiments.py --results priority_reviewer_gpu/results.csv \
    --out priority_reviewer_gpu/statistics
python src/per_class_ap.py
python src/threshold_calibration.py
```

The script is resumable: re-running it skips any `(model, variant, seed)` combination already
marked `complete` in the results CSV.

## Headline result

Under the paper's original 5-epoch, batch-2, CPU-only protocol, mask-guided copy-paste appeared
to substantially improve detection (mAP50 +25.7%, recall +45.5%, rare-class AP50 +69%). Repeating
the same comparison with YOLOv8n trained to near-convergence on GPU (100 epochs, early stopping,
5 seeds) does not replicate this: none of precision, recall, mAP50, mAP50:95, or rare-class AP50
differ significantly from baseline (paired t-test, all p>0.12), and a built-in Mosaic augmentation
outperforms the proposed method (p=0.01). The method does show a statistically significant,
seed-robust mAP gain on two newer, anchor-free detector architectures (YOLOv10n: p=0.013; YOLO11n:
p=0.037). Full tables and statistics are in `results/` and `docs/GPU_VALIDATION_FINDINGS_AND_REWRITE.md`.

## Trained weights

Per-run `best.pt` checkpoints (63 runs, ~470 MB total) are not tracked in this repository. They
are provided as a separate archive attached to the corresponding GitHub Release / Zenodo record.

## Citation

If you use this code or these results, please cite the manuscript (citation details to be added
upon publication) and this repository's Zenodo DOI (to be minted on first GitHub Release).

## License

MIT License, see `LICENSE`.
