import csv
from pathlib import Path

import numpy as np
from ultralytics import YOLO

BASE = Path(r"D:\gulsiArticleProject\Архив")


def main():
    rows = list(csv.DictReader(open(BASE / "priority_reviewer_gpu" / "results.csv", encoding="utf-8")))
    rows = [r for r in rows if r["status"] == "complete" and r["model"] == "yolov8n.pt"
            and r["variant"] in ("baseline", "mosaic", "mask_guided")]

    out_rows = []
    curves = {}  # (variant) -> list of (px, p_mean, r_mean, f1_mean)
    for r in rows:
        weights = r["best_weights"]
        if not Path(weights).is_absolute():
            weights = str(BASE / weights)
        val = YOLO(weights).val(
            data=r["data"], imgsz=int(r["imgsz"]), batch=16,
            device="0", workers=0, project="priority_reviewer_gpu",
            name=f"thresh_{r['variant']}_seed{r['seed']}", exist_ok=True,
            plots=False, save_json=False, verbose=False,
        )
        box = val.box
        px = np.array(box.px)
        p_mean = np.array(box.p_curve).mean(axis=0)
        r_mean = np.array(box.r_curve).mean(axis=0)
        f1_mean = np.array(box.f1_curve).mean(axis=0)

        default_idx = np.abs(px - 0.25).argmin()
        best_idx = f1_mean.argmax()

        out_rows.append({
            "variant": r["variant"], "seed": r["seed"],
            "P_default": p_mean[default_idx], "R_default": r_mean[default_idx], "F1_default": f1_mean[default_idx],
            "conf_best": px[best_idx],
            "P_best": p_mean[best_idx], "R_best": r_mean[best_idx], "F1_best": f1_mean[best_idx],
        })
        curves.setdefault(r["variant"], []).append((px, p_mean, r_mean, f1_mean))
        print(r["variant"], r["seed"], "F1_default=%.3f F1_best=%.3f @conf=%.2f" % (
            f1_mean[default_idx], f1_mean[best_idx], px[best_idx]))

    with open(BASE / "priority_reviewer_gpu" / "threshold_calibration.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(out_rows[0].keys()))
        w.writeheader()
        w.writerows(out_rows)
    print("wrote threshold_calibration.csv")

    # Averaged PR curve per variant (mean over seeds) for plotting
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.figure(figsize=(6, 5))
    for variant, runs in curves.items():
        px = runs[0][0]
        r_stack = np.stack([c[2] for c in runs])
        p_stack = np.stack([c[1] for c in runs])
        r_avg = r_stack.mean(axis=0)
        p_avg = p_stack.mean(axis=0)
        order = np.argsort(r_avg)
        plt.plot(r_avg[order], p_avg[order], label=variant)
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Precision-Recall curve (mean over seeds, YOLOv8n)")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(BASE / "priority_reviewer_gpu" / "pr_curve_comparison.png", dpi=150)
    print("wrote pr_curve_comparison.png")


if __name__ == "__main__":
    main()
