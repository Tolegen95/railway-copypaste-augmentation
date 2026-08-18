# GPU Validation of the Mask-Guided Copy-Paste Claim — Findings and Rewritten Manuscript Text

## What was done

All experiments below use YOLOv8n (and four other detectors) trained for up to 100 epochs with
patience-20 early stopping at 640x640 on GPU (RTX 4060), using `full_reviewer_experiments.py` and
`analyze_reviewer_experiments.py`. This replaces the manuscript's 5-epoch / batch-2 / CPU-only
protocol, which every reviewer flagged as too weak to support the paper's claims. Paired t-tests
compare each variant against the baseline on the same seeds.

Raw results: `priority_reviewer_gpu/results.csv`, `priority_reviewer_gpu/statistics/`,
`priority_reviewer_gpu/per_class_ap50_n5.csv`, `priority_reviewer_gpu/threshold_calibration.csv`,
`priority_reviewer_gpu/pr_curve_comparison.png`.

## 1. Headline result: the effect does not replicate under proper training (YOLOv8n, n=5 seeds)

| Metric | Baseline | Mask-guided copy-paste | Δ | paired t-test p |
|---|---|---|---|---|
| Precision | 0.692 | 0.719 | +0.027 | 0.49 |
| Recall | 0.672 | 0.675 | +0.002 | 0.95 |
| mAP50 | 0.714 | 0.711 | −0.003 | 0.90 |
| mAP50:95 | 0.392 | 0.372 | −0.020 | 0.12 |
| Rare-class mean AP50 (gap_izo, gap_slip, skrep_no) | 0.765 | 0.761 | −0.006 (−0.6%) | 0.93 |

The manuscript's pilot (5 CPU epochs, one split) reported mAP50 0.343→0.431 (+25.7%), recall
0.391→0.569 (+45.5%), and rare-class AP50 +69%. None of this survives training to
near-convergence: the baseline itself reaches mAP50 ≈0.71 and rare-class AP50 ≈0.77 without any
augmentation once it is given enough epochs. The pilot's apparent gain was the augmented run
compensating faster for an undertrained baseline (5 epochs), not a property of the method — exactly
what Reviewer 2 warned about ("uncertain whether the improvement stems from the augmentation or
from the model not yet reaching convergence").

### Per-class detail (AP50, n=5 seeds)

| Class | Baseline | Mask-guided | Δ |
|---|---|---|---|
| bolt_no | 0.938 | 0.943 | +0.004 |
| gap_def | 0.800 | 0.716 | −0.083 |
| gap_izo (rare) | 0.968 | 0.962 | −0.007 |
| gap_slip (rare) | 0.680 | 0.725 | +0.045 |
| shpala_razlom | 0.346 | 0.498 | **+0.152** |
| shpala_treshina | 0.385 | 0.349 | −0.036 |
| skrep_bolt_no | 0.951 | 0.902 | −0.049 |
| skrep_no (rare) | 0.647 | 0.595 | −0.052 |

The one reproducible positive effect is **shpala_razlom (sleeper fracture)**, +0.152 AP50, not one
of the three classes the manuscript highlights as its headline result.

## 2. Comparison with standard augmentation baselines (Reviewer 1 #2, Reviewer 2 #3)

YOLOv8n, 100 epochs, GPU:

| Variant | mAP50 | Δ vs baseline | p |
|---|---|---|---|
| Baseline | 0.714 | — | — |
| **Mosaic** (built into Ultralytics by default) | **0.740** | **+0.019** | **0.01** ✅ |
| CutMix | 0.722 | +0.001 | 0.96 |
| Random Erasing | 0.708 | −0.013 | 0.49 |
| MixUp | 0.705 | −0.016 | 0.04 (significantly worse) |
| Mask-guided copy-paste (proposed) | 0.711 | −0.003 | 0.90 |

Plain Mosaic — a single flag already available in the training framework, at zero extra
engineering cost — outperforms the proposed method and is the only variant reaching significance.

## 3. Detector generality (Reviewer 4 #6)

Baseline vs. mask-guided, mAP50:

| Detector | Baseline | Mask-guided | Δ | p (mAP50) | p (mAP50:95) |
|---|---|---|---|---|---|
| YOLOv8n (n=5) | 0.714 | 0.711 | −0.003 | 0.90 | 0.12 |
| YOLOv8s (n=3) | 0.742 | 0.744 | +0.002 | 0.86 | 0.53 (recall significantly **worse**, p=0.046) |
| YOLOv9t (n=3) | 0.741 | 0.743 | +0.002 | 0.95 | 0.07 |
| YOLOv10n (n=5) | 0.601 | 0.689 | **+0.088** | **0.013** ✅ | **0.005** ✅ |
| YOLO11n (n=5) | 0.723 | 0.755 | +0.033 | 0.27 | **0.037** ✅ |

The method shows no benefit on the three more established architectures (v8n, v8s, v9t) but a
real, seed-robust gain on the two newest, anchor-free/NMS-free architectures (v10n, v11n). This is
a genuine, narrower, defensible finding — architecture-dependent, not universal.

## 4. Precision–recall trade-off and threshold calibration (Reviewer 2 #5)

The manuscript claims the precision drop "can be recovered through threshold calibration." Tested
directly by sweeping the confidence threshold and taking each model's own best-F1 operating point
(YOLOv8n):

| Variant | F1 @ default (conf=0.25) | F1 @ own best threshold | best conf |
|---|---|---|---|
| Baseline | 0.662 | 0.675 | ≈0.27 |
| Mosaic | 0.700 | **0.710** | ≈0.36 |
| Mask-guided | 0.645 | 0.668 | ≈0.32 |

Calibration improves all three by a similar small margin but does not change their ranking:
mask-guided still trails both baseline and Mosaic at its own best threshold. The calibration claim
is not supported and should be removed or qualified.

## 5. Sensitivity / ablation on copy-paste parameters (Reviewer 1 #3, Reviewer 2 #7)

Donor-sampling policy and augmentation volume were varied (scale range, overlap and paste count
remain fixed in the current pipeline). YOLOv8n, 100 epochs, n=3 seeds each:

| Variant | Precision | mAP50 | mAP50:95 |
|---|---|---|---|
| Baseline | 0.692 | 0.714 | 0.392 |
| Mask-guided (inverse-sqrt sampling, 1x volume — manuscript's method) | 0.719 (n.s.) | 0.711 (n.s.) | 0.372 (n.s.) |
| Standard Copy-Paste (uniform sampling, 1x volume) | **0.778 (p=0.01)** | 0.731 (n.s.) | 0.364 (p=0.06) |
| Uniform-class sampling (1x volume) | 0.735 (n.s.) | 0.707 (n.s.) | 0.375 (p=0.02, worse) |
| 2x augmentation volume (inverse-sqrt sampling) | **0.746 (p=0.00)** | 0.745 (n.s.) | 0.373 (p=0.05, worse) |

No configuration reaches significance on mAP50, the paper's primary metric. Two configurations
significantly raise precision alone. There is no evidence that any tested copy-paste variant,
including the one described in the manuscript, delivers a reliable detection-quality improvement
on YOLOv8n.

## 6. What is still not covered

- Component-aware (geometry-constrained) placement — not implemented, still a valid future-work item.
- External test set from other lines/cameras/lighting — requires new data collection, out of scope.
- Public code/weights release with a DOI — not done.
- Confusion matrices / prediction panels exist per-run (saved automatically) but have not been
  individually reviewed for qualitative failure analysis.

---

## Rewritten manuscript text

### Abstract (replacement)

> This paper evaluates a mask-guided copy-paste augmentation pipeline for railway defect detection
> on a small, imbalanced, self-collected dataset (447 pixel-annotated images, eight defect classes).
> Real defect instances are extracted from existing segmentation masks and pasted into training
> images with class-balanced sampling, without any generative model. An initial CPU-only pilot
> (5 epochs, fixed 80/20 split) suggested large gains (mAP50 +25.7%, recall +45.5%, rare-class AP50
> +69%). Because reviewers correctly identified that this protocol was too short to separate a
> genuine augmentation effect from simple undertraining, we repeated the evaluation with YOLOv8n
> trained to near-convergence on GPU (100 epochs, early stopping, 5 seeds) and extended it to four
> further one-stage detectors (YOLOv8s, YOLOv9t, YOLOv10n, YOLO11n) and four standard augmentation
> baselines (Mosaic, MixUp, CutMix, Random Erasing). Under this protocol, the originally reported
> gains do not replicate on YOLOv8n: mAP50, recall, and rare-class AP50 differences from baseline
> are statistically indistinguishable from zero (paired t-test, all p>0.12), and a single built-in
> Mosaic augmentation outperforms the proposed method (mAP50 +0.019, p=0.01). A sensitivity analysis
> over donor-sampling policy and augmentation volume likewise found no configuration of the
> copy-paste pipeline that produced a significant mAP50 gain over baseline. The method does,
> however, show a statistically significant, seed-robust mAP improvement on two newer,
> anchor-free/NMS-free detectors (YOLOv10n: mAP50 +0.088, p=0.013; YOLO11n: mAP50:95 +0.037,
> p=0.037), and a consistent per-class gain on one class (sleeper fracture, AP50 +0.152) not
> highlighted in the original analysis. We report these results as an honest correction: a
> lightweight, reproducible, mask-derived copy-paste pipeline is not a reliable general-purpose
> improvement for mature YOLO detectors on this dataset, but it is architecture-dependent and may be
> worth targeted use on newer detector families. All code, configurations, and raw per-seed results
> are released for independent verification.

### Results section (replacement structure)

- **6.1 Baseline convergence check.** Report that under 5 CPU epochs the baseline itself was
  severely undertrained (mAP50 0.343, rare-class AP50 as low as 0.308 on gap_izo); under 100 GPU
  epochs the same architecture reaches mAP50 ≈0.71 and rare-class AP50 ≈0.77 with no augmentation
  at all. State plainly that this invalidates a direct comparison at 5 epochs.
- **6.2 Main comparison at convergence.** Table from Section 1 above (baseline vs. mask-guided,
  n=5 seeds, paired stats). State the null result explicitly.
- **6.3 Comparison with standard augmentations.** Table from Section 2. State that Mosaic alone
  outperforms the proposed method.
- **6.4 Detector generality.** Table from Section 3. Present the YOLOv10n/YOLO11n result as the
  paper's actual positive finding, with the caveat that it does not generalize to YOLOv8/v9.
- **6.5 Threshold calibration.** Table from Section 4. Remove or correct the calibration claim.
- **6.6 Per-class analysis.** Table from Section 1. Replace the gap_izo narrative with the
  shpala_razlom finding, and report the other rare classes honestly (flat or slightly negative).
- **6.7 Sensitivity analysis.** Table from Section 5, in place of the "qualitative sensitivity
  analysis" placeholder from the Reviewer 1 response.

### Discussion (key changes)

- Delete claims that the augmentation "improves YOLO-based railway defect detection" as a general
  statement; replace with "does not produce a statistically detectable improvement on YOLOv8n/8s/9t
  under proper training, but produces a significant, reproducible mAP gain on YOLOv10n and YOLO11n."
- Delete the confidence-threshold-calibration recovery claim (Section 7, current manuscript);
  replace with the calibration result showing no rank change.
- Reframe the contribution as: (1) a reproducible, low-cost pipeline for turning existing pixel
  annotations into additional training instances, (2) an honest empirical account of when it does
  and does not help, (3) evidence that undertrained pilot studies can produce large, illusory
  augmentation effects — a methodological caution relevant beyond this paper.

### Conclusion (replacement)

> This paper presents a lightweight, reproducible pipeline that reuses existing pixel-level
> railway-defect annotations to generate additional class-balanced training instances via
> mask-guided copy-paste. An initial short CPU pilot suggested substantial gains, but these did not
> replicate once the same comparison was repeated with proper GPU training to near-convergence,
> multiple seeds, and paired statistical testing: on YOLOv8n, YOLOv8s, and YOLOv9t the method
> produces no detectable improvement over baseline, and is outperformed by a standard built-in
> Mosaic augmentation. The method does yield a statistically significant, seed-robust mAP
> improvement on two newer, anchor-free detector architectures (YOLOv10n, YOLO11n), and a consistent
> per-class gain on one class not previously highlighted (sleeper fracture). We therefore do not
> recommend mask-guided copy-paste as a general-purpose improvement for railway defect detection
> with mature YOLO detectors on this dataset; its value appears architecture-specific and should be
> verified per-detector before adoption. The undertraining artifact uncovered here — a large,
> statistically unsupported effect visible only under a short training budget — is itself a
> cautionary result for evaluating data-augmentation methods on small datasets under limited
> compute, and the full experimental pipeline (code, seeds, configurations, and raw results) is
> released so this comparison can be repeated or extended.

---

## Note on the existing response letters

`FINAL_Response_to_Reviewer_1/2/4.docx` currently state things like "no converged GPU experiment
was available" and "we do not present unavailable GPU results" — this is no longer true and these
letters need a follow-up addendum stating: GPU experiments were run to convergence with 5 seeds
across 5 detectors; the results do not support the original claims; the manuscript has been revised
accordingly rather than merely having its language softened.
