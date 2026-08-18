import csv
from pathlib import Path

from ultralytics import YOLO

BASE = Path(r"D:\gulsiArticleProject\Архив")


def main():
    rows = list(csv.DictReader(open(BASE / "priority_reviewer_gpu" / "results.csv", encoding="utf-8")))

    names = None
    out_rows = []
    for r in rows:
        if r["variant"] not in ("baseline", "mask_guided"):
            continue
        weights = r["best_weights"]
        if not Path(weights).is_absolute():
            weights = str(BASE / weights)
        val = YOLO(weights).val(
            data=r["data"], imgsz=int(r["imgsz"]), batch=16,
            device="0", workers=0, project="priority_reviewer_gpu",
            name=f"perclass_{r['variant']}_seed{r['seed']}", exist_ok=True,
            plots=False, save_json=False, verbose=False,
        )
        if names is None:
            names = val.names
        ap50_per_class = val.box.ap50
        for cls_id, ap in enumerate(ap50_per_class):
            out_rows.append({
                "variant": r["variant"], "seed": r["seed"],
                "class": names[cls_id], "AP50": float(ap),
            })
        print(r["variant"], r["seed"], "done")

    with open(BASE / "priority_reviewer_gpu" / "per_class_ap50_n5.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["variant", "seed", "class", "AP50"])
        w.writeheader()
        w.writerows(out_rows)
    print("wrote per_class_ap50.csv")


if __name__ == "__main__":
    main()
