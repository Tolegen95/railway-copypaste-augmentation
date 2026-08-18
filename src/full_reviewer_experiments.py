"""Unified, resumable experiments requested by the reviewers.

Runs matched YOLO model/augmentation/seed experiments, validates best weights,
and stores one machine-readable row per run. Designed for GPU use but supports CPU.
"""
import argparse
import csv
import json
import time
from pathlib import Path

from ultralytics import YOLO


AUGMENTATIONS = {
    # Shared conservative photometric/geometric transforms; only the named factor changes.
    "baseline": dict(mosaic=0.0, mixup=0.0, cutmix=0.0, erasing=0.0),
    "mosaic": dict(mosaic=1.0, mixup=0.0, cutmix=0.0, erasing=0.0),
    "mixup": dict(mosaic=0.0, mixup=0.2, cutmix=0.0, erasing=0.0),
    "cutmix": dict(mosaic=0.0, mixup=0.0, cutmix=0.2, erasing=0.0),
    "random_erasing": dict(mosaic=0.0, mixup=0.0, cutmix=0.0, erasing=0.4),
    "simple_copy_paste": dict(mosaic=0.0, mixup=0.0, cutmix=0.0, erasing=0.0),
    # Dataset contains offline mask-guided samples; built-in mixing is disabled.
    "mask_guided": dict(mosaic=0.0, mixup=0.0, cutmix=0.0, erasing=0.0),
    # Sensitivity-analysis variants: same offline copy-paste mechanism, different
    # donor-sampling policy or augmented-volume ratio (see prepare_mask_guided_copypaste_yolo.py).
    "uniform_class": dict(mosaic=0.0, mixup=0.0, cutmix=0.0, erasing=0.0),
    "aug2x": dict(mosaic=0.0, mixup=0.0, cutmix=0.0, erasing=0.0),
}


def parse_list(value, cast=str):
    return [cast(x.strip()) for x in value.split(",") if x.strip()]


def load_rows(path):
    if not path.exists():
        return []
    with path.open(encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def save_rows(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    fields = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)
    path.with_suffix(".json").write_text(json.dumps(rows, indent=2), encoding="utf-8")


def metric(box, name):
    return float(getattr(box, name))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", default="yolov8n.pt,yolov8s.pt,yolov9t.pt,yolov10n.pt,yolo11n.pt")
    ap.add_argument("--variants", default="baseline,mosaic,mixup,cutmix,random_erasing,mask_guided")
    ap.add_argument("--seeds", default="0,1,2")
    ap.add_argument("--original-data", default="dataset_yolo_mask_original/railway_defect_copypaste.yaml")
    ap.add_argument("--mask-data", default="dataset_yolo_mask_copypaste/railway_defect_copypaste.yaml")
    ap.add_argument("--simple-data", default="dataset_yolo_simple_copypaste/railway_defect_copypaste.yaml")
    ap.add_argument("--uniformclass-data", default="dataset_yolo_uniformclass_copypaste/railway_defect_copypaste.yaml")
    ap.add_argument("--aug2x-data", default="dataset_yolo_aug2x_copypaste/railway_defect_copypaste.yaml")
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--patience", type=int, default=20)
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--batch", type=int, default=-1, help="-1 lets Ultralytics choose GPU batch size")
    ap.add_argument("--device", default="0")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--project", default="full_reviewer_experiments")
    ap.add_argument("--results", default="full_reviewer_experiments/results.csv")
    args = ap.parse_args()

    models = parse_list(args.models)
    variants = parse_list(args.variants)
    seeds = parse_list(args.seeds, int)
    unknown = sorted(set(variants) - set(AUGMENTATIONS))
    if unknown:
        raise SystemExit(f"Unknown variants: {unknown}")

    results_path = Path(args.results)
    rows = load_rows(results_path)
    completed = {(r["model"], r["variant"], int(r["seed"])) for r in rows if r.get("status") == "complete"}

    for model_path in models:
        for variant in variants:
            for seed in seeds:
                key = (model_path, variant, seed)
                if key in completed:
                    print("SKIP", key)
                    continue
                if variant == "mask_guided":
                    data = args.mask_data
                elif variant == "simple_copy_paste":
                    data = args.simple_data
                elif variant == "uniform_class":
                    data = args.uniformclass_data
                elif variant == "aug2x":
                    data = args.aug2x_data
                else:
                    data = args.original_data
                name = f"{Path(model_path).stem}_{variant}_seed{seed}_e{args.epochs}_i{args.imgsz}"
                settings = AUGMENTATIONS[variant]
                started = time.time()
                record = {
                    "model": model_path, "variant": variant, "seed": seed,
                    "status": "failed", "data": data, "epochs_max": args.epochs,
                    "patience": args.patience, "imgsz": args.imgsz, "batch": args.batch,
                    "device": args.device, **settings,
                }
                try:
                    # Ultralytics versions differ on whether a relative `project`
                    # is used as-is or nested under a global runs_dir/<task>/,
                    # so check both known layouts before deciding to (re)train.
                    legacy_best = Path(args.project) / name / "weights" / "best.pt"
                    nested_best = Path("runs") / "detect" / args.project / name / "weights" / "best.pt"
                    best = legacy_best if legacy_best.exists() else nested_best
                    if not best.exists():
                        model = YOLO(model_path)
                        model.train(
                            data=data, epochs=args.epochs, patience=args.patience,
                            imgsz=args.imgsz, batch=args.batch, device=args.device,
                            workers=args.workers, seed=seed, deterministic=True,
                            project=args.project, name=name, exist_ok=True,
                            pretrained=True, plots=True, save=True, verbose=False,
                            hsv_h=0.015, hsv_s=0.7, hsv_v=0.4,
                            degrees=0.0, translate=0.1, scale=0.5,
                            shear=0.0, perspective=0.0, flipud=0.0, fliplr=0.5,
                            **settings,
                        )
                        best = Path(model.trainer.save_dir) / "weights" / "best.pt"
                    # val() rejects batch=-1 (only train()'s auto-batch accepts it);
                    # any fixed positive batch is fine since it only affects val speed.
                    val_batch = args.batch if args.batch and args.batch > 0 else 16
                    val = YOLO(str(best)).val(
                        data=data, imgsz=args.imgsz, batch=val_batch,
                        device=args.device, workers=args.workers,
                        project=args.project, name=name + "_val", exist_ok=True,
                        plots=True, save_json=False, verbose=False,
                    )
                    box, speed = val.box, val.speed or {}
                    record.update({
                        "status": "complete", "precision": metric(box, "mp"),
                        "recall": metric(box, "mr"), "mAP50": metric(box, "map50"),
                        "mAP50_95": metric(box, "map"),
                        "inference_ms": float(speed.get("inference", float("nan"))),
                        "best_weights": str(best),
                    })
                except Exception as exc:
                    record["error"] = repr(exc)
                record["elapsed_seconds"] = time.time() - started
                rows.append(record)
                save_rows(results_path, rows)
                print(json.dumps(record, indent=2))


if __name__ == "__main__":
    main()
