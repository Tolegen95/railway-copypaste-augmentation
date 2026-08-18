import argparse
import json
import math
import random
import shutil
from pathlib import Path

import cv2
import numpy as np


IMG_EXTS = (".jpg", ".jpeg", ".png", ".JPG", ".JPEG", ".PNG")


def find_image(images_dir: Path, stem: str) -> Path:
    for ext in IMG_EXTS:
        p = images_dir / f"{stem}{ext}"
        if p.exists():
            return p
    raise FileNotFoundError(f"Image not found for id={stem}")


def load_classes(path: Path):
    with path.open("r", encoding="utf-8") as f:
        name_to_id = json.load(f)
    id_to_name = {int(v): k for k, v in name_to_id.items()}
    return name_to_id, id_to_name


def mask_to_yolo_boxes(mask: np.ndarray, num_classes: int, min_area: int):
    h, w = mask.shape
    boxes = []
    for cid in range(1, num_classes):
        binary = (mask == cid).astype(np.uint8)
        n, labels, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)
        for idx in range(1, n):
            x, y, bw, bh, area = stats[idx]
            if area < min_area or bw < 2 or bh < 2:
                continue
            xc = (x + bw / 2) / w
            yc = (y + bh / 2) / h
            boxes.append((cid - 1, xc, yc, bw / w, bh / h))
    return boxes


def write_yolo_label(path: Path, boxes):
    lines = [f"{c} {xc:.6f} {yc:.6f} {bw:.6f} {bh:.6f}" for c, xc, yc, bw, bh in boxes]
    path.write_text("\n".join(lines), encoding="utf-8")


def copy_image(src: Path, dst: Path):
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)


def build_instance_library(ids, images_dir: Path, masks_dir: Path, num_classes: int, min_area: int):
    instances = []
    class_counts = {cid: 0 for cid in range(1, num_classes)}
    for stem in ids:
        img = cv2.imread(str(find_image(images_dir, stem)), cv2.IMREAD_COLOR)
        mask = cv2.imread(str(masks_dir / f"{stem}.png"), cv2.IMREAD_GRAYSCALE)
        if img is None or mask is None:
            continue
        for cid in range(1, num_classes):
            binary = (mask == cid).astype(np.uint8)
            n, labels, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)
            for idx in range(1, n):
                x, y, w, h, area = stats[idx]
                if area < min_area or w < 2 or h < 2:
                    continue
                inst_mask = (labels[y : y + h, x : x + w] == idx).astype(np.uint8)
                crop = img[y : y + h, x : x + w].copy()
                instances.append(
                    {
                        "class_id": cid,
                        "source": stem,
                        "crop": crop,
                        "mask": inst_mask,
                        "area": int(area),
                        "box": (int(x), int(y), int(w), int(h)),
                    }
                )
                class_counts[cid] += 1
    return instances, class_counts


def choose_instance(instances, class_weights, rng: random.Random):
    weights = [class_weights[i["class_id"]] for i in instances]
    return rng.choices(instances, weights=weights, k=1)[0]


def paste_instance(base_img, base_mask, inst, rng: random.Random, max_overlap: float):
    h, w = base_mask.shape
    crop = inst["crop"]
    imask = inst["mask"]

    scale = rng.uniform(0.75, 1.25)
    new_w = max(2, int(crop.shape[1] * scale))
    new_h = max(2, int(crop.shape[0] * scale))
    if new_w >= w or new_h >= h:
        return False

    crop_r = cv2.resize(crop, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
    mask_r = cv2.resize(imask, (new_w, new_h), interpolation=cv2.INTER_NEAREST).astype(bool)
    if mask_r.sum() < 10:
        return False

    for _ in range(40):
        x = rng.randint(0, w - new_w)
        y = rng.randint(0, h - new_h)
        roi_mask = base_mask[y : y + new_h, x : x + new_w]
        overlap = np.logical_and(roi_mask > 0, mask_r).sum() / max(mask_r.sum(), 1)
        if overlap <= max_overlap:
            feather = cv2.GaussianBlur(mask_r.astype(np.float32), (5, 5), 0)
            feather = np.clip(feather[..., None], 0.0, 1.0)
            roi_img = base_img[y : y + new_h, x : x + new_w].astype(np.float32)
            blended = crop_r.astype(np.float32) * feather + roi_img * (1.0 - feather)
            base_img[y : y + new_h, x : x + new_w] = blended.astype(np.uint8)
            roi_mask[mask_r] = inst["class_id"]
            return True
    return False


def make_augmented_sample(stem, out_stem, images_dir, masks_dir, out_img, out_lbl, instances, class_weights, rng, num_classes, min_area):
    img = cv2.imread(str(find_image(images_dir, stem)), cv2.IMREAD_COLOR)
    mask = cv2.imread(str(masks_dir / f"{stem}.png"), cv2.IMREAD_GRAYSCALE)
    if img is None or mask is None:
        return False

    n_paste = rng.choice([1, 1, 2, 2, 3])
    pasted = 0
    for _ in range(n_paste):
        inst = choose_instance(instances, class_weights, rng)
        if paste_instance(img, mask, inst, rng, max_overlap=0.15):
            pasted += 1

    if pasted == 0:
        return False

    cv2.imwrite(str(out_img / f"{out_stem}.jpg"), img)
    boxes = mask_to_yolo_boxes(mask, num_classes=num_classes, min_area=min_area)
    write_yolo_label(out_lbl / f"{out_stem}.txt", boxes)
    return True


def save_yaml(out_dir: Path, names):
    yaml_text = [
        f"path: {out_dir.as_posix()}",
        "train: images/train",
        "val: images/val",
        "names:",
    ]
    for idx, name in enumerate(names):
        yaml_text.append(f"  {idx}: {name}")
    (out_dir / "railway_defect_copypaste.yaml").write_text("\n".join(yaml_text) + "\n", encoding="utf-8")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default="dataset_seg2")
    ap.add_argument("--out", default="dataset_yolo_mask_copypaste")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--val_ratio", type=float, default=0.2)
    ap.add_argument("--aug_per_train", type=float, default=1.0)
    ap.add_argument("--min_area", type=int, default=20)
    ap.add_argument(
        "--sampling",
        choices=("inverse_sqrt", "uniform_instance", "uniform_class"),
        default="inverse_sqrt",
        help="Donor sampling policy. uniform_instance is the Simple Copy-Paste baseline.",
    )
    args = ap.parse_args()

    rng = random.Random(args.seed)
    src = Path(args.src)
    out = Path(args.out)
    images_dir = src / "images"
    masks_dir = src / "masks"
    _, id_to_name = load_classes(src / "classes.json")
    num_classes = max(id_to_name) + 1
    yolo_names = [id_to_name[i] for i in range(1, num_classes)]

    ids = sorted(p.stem for p in masks_dir.glob("*.png"))
    rng.shuffle(ids)
    val_n = int(round(len(ids) * args.val_ratio))
    val_ids = sorted(ids[:val_n])
    train_ids = sorted(ids[val_n:])

    if out.exists():
        shutil.rmtree(out)
    for split in ("train", "val"):
        (out / "images" / split).mkdir(parents=True, exist_ok=True)
        (out / "labels" / split).mkdir(parents=True, exist_ok=True)

    for split, split_ids in (("train", train_ids), ("val", val_ids)):
        for stem in split_ids:
            ip = find_image(images_dir, stem)
            ext = ip.suffix.lower()
            dst_name = f"{stem}{ext if ext in IMG_EXTS else '.jpg'}"
            copy_image(ip, out / "images" / split / dst_name)
            mask = cv2.imread(str(masks_dir / f"{stem}.png"), cv2.IMREAD_GRAYSCALE)
            boxes = mask_to_yolo_boxes(mask, num_classes=num_classes, min_area=args.min_area)
            write_yolo_label(out / "labels" / split / f"{stem}.txt", boxes)

    instances, class_counts = build_instance_library(train_ids, images_dir, masks_dir, num_classes, args.min_area)
    if args.sampling == "inverse_sqrt":
        class_weights = {cid: 1.0 / math.sqrt(max(count, 1)) for cid, count in class_counts.items()}
    elif args.sampling == "uniform_class":
        # choose_instance operates on instances, therefore 1/n_c yields equal total class mass
        class_weights = {cid: 1.0 / max(count, 1) for cid, count in class_counts.items()}
    else:
        # Every extracted instance receives equal probability: standard Simple Copy-Paste.
        class_weights = {cid: 1.0 for cid in class_counts}

    aug_target = int(round(len(train_ids) * args.aug_per_train))
    made = 0
    attempts = 0
    while made < aug_target and attempts < aug_target * 8:
        attempts += 1
        stem = rng.choice(train_ids)
        out_stem = f"cp_{made:05d}_{stem}"
        ok = make_augmented_sample(
            stem,
            out_stem,
            images_dir,
            masks_dir,
            out / "images" / "train",
            out / "labels" / "train",
            instances,
            class_weights,
            rng,
            num_classes,
            args.min_area,
        )
        if ok:
            made += 1

    save_yaml(out, yolo_names)
    summary = {
        "source": str(src),
        "output": str(out),
        "seed": args.seed,
        "annotated_images": len(ids),
        "train_original": len(train_ids),
        "val_original": len(val_ids),
        "augmented_train": made,
        "sampling": args.sampling,
        "num_classes_yolo": len(yolo_names),
        "names": yolo_names,
        "train_instance_counts": {id_to_name[cid]: count for cid, count in class_counts.items()},
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
