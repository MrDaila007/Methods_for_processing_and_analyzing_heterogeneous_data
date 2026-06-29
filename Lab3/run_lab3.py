"""
Utility script to test YOLO-World/Yolov8 inference for Lab3 without notebook.

Features:
- Resolves paths to Google Drive (`/content/drive/.../Methods_for_processing_and_analyzing_heterogeneous_data/Lab3`)
  or falls back to the local repo path.
- (Optional) downloads weights and demo images if they are missing.
- Runs YOLO-World on all images in `data/demo` with custom text classes and saves predictions.

Usage (Colab):
  python run_lab3.py --download-weights --download-demo --classes "electric scooter,coffee cup with logo,dog,person"

Usage (local):
  python run_lab3.py --root /path/to/repo/Lab3 --no-download-weights --no-download-demo
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import time
import urllib.request
from pathlib import Path


def resolve_root(user_root: str | None) -> Path:
    """Resolve root folder (Lab3) with a Drive-first fallback."""
    if user_root:
        return Path(user_root).expanduser().resolve()
    drive_base = Path("/content/drive/MyDrive/Methods_for_processing_and_analyzing_heterogeneous_data")
    if drive_base.exists():
        return (drive_base / "Lab3").resolve()
    return (Path(__file__).resolve().parent)


def download_file(url: str, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists():
        print(f"[skip] {dst.name} already exists")
        return
    try:
        print(f"[download] {url} -> {dst}")
        urllib.request.urlretrieve(url, dst)
    except Exception as exc:  # noqa: BLE001
        print(f"[warn] failed to download {url}: {exc}")


def ensure_ultralytics(auto_install: bool):
    """Import ultralytics; optionally install if missing (requires internet)."""
    try:
        from ultralytics import YOLO, YOLOWorld
        return YOLO, YOLOWorld
    except Exception as exc:  # noqa: BLE001
        if not auto_install:
            print("[error] ultralytics is not installed; enable --auto-install or install manually:", exc)
            return None, None
        print("[info] ultralytics not found, trying to install (requires internet)...")
        try:
            # Try installing without opencv-python-headless first (may conflict with conda)
            subprocess.check_call(
                [sys.executable, "-m", "pip", "install", "-q", "ultralytics"]
            )
        except Exception as install_exc:  # noqa: BLE001
            print("[error] failed to install ultralytics:", install_exc)
            return None, None
        try:
            from ultralytics import YOLO, YOLOWorld  # type: ignore
            return YOLO, YOLOWorld
        except Exception as exc2:  # noqa: BLE001
            print("[error] ultralytics import failed after install:", exc2)
            return None, None


def prepare_assets(root: Path, download_weights: bool, download_demo: bool) -> None:
    models = root / "models"
    demo = root / "data" / "demo"
    models.mkdir(parents=True, exist_ok=True)
    demo.mkdir(parents=True, exist_ok=True)

    weight_urls = {
        "yolov8m-worldv2.pt": "https://huggingface.co/ultralytics/yolo-world/resolve/main/yolo_world_v2_m.pt",
        "yolov8n.pt": "https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8n.pt",
    }
    demo_urls = {
        "bus.jpg": "https://ultralytics.com/images/bus.jpg",
        "zidane.jpg": "https://ultralytics.com/images/zidane.jpg",
    }

    if download_weights:
        for name, url in weight_urls.items():
            download_file(url, models / name)
    if download_demo:
        for name, url in demo_urls.items():
            download_file(url, demo / name)


def load_models(root: Path, auto_install: bool):
    YOLO, YOLOWorld = ensure_ultralytics(auto_install)
    if YOLO is None or YOLOWorld is None:
        return None, None

    models_dir = root / "models"
    yw_path = models_dir / "yolov8m-worldv2.pt"
    y8_path = models_dir / "yolov8n.pt"

    if not yw_path.exists():
        print(f"[error] missing {yw_path}, run with --download-weights or add manually")
        return None, None
    try:
        yw_model = YOLOWorld(yw_path)
    except Exception as exc:  # noqa: BLE001
        print("[error] failed to load YOLO-World:", exc)
        yw_model = None

    try:
        y8_model = YOLO(y8_path) if y8_path.exists() else None
    except Exception as exc:  # noqa: BLE001
        print("[warn] failed to load YOLOv8:", exc)
        y8_model = None

    return yw_model, y8_model


def run_inference(yw_model, demo_dir: Path, classes: list[str], imgsz: int, conf: float) -> None:
    if yw_model is None:
        print("[error] YOLO-World model is not loaded; aborting inference.")
        return
    images = sorted(demo_dir.glob("*.jpg")) + sorted(demo_dir.glob("*.png"))
    if not images:
        print(f"[warn] no images found in {demo_dir}")
        return

    yw_model.set_classes(classes)
    for img_path in images:
        start = time.time()
        res = yw_model.predict(img_path, imgsz=imgsz, conf=conf, verbose=False)[0]
        latency = time.time() - start
        res.save(filename=str(img_path.with_name(f"pred_{img_path.name}")))
        print(f"{img_path.name}: {len(res.boxes)} boxes, {latency:.3f}s -> pred_{img_path.name}")


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run YOLO-World demo inference for Lab3")
    parser.add_argument("--root", type=str, default=None, help="Root path to Lab3 (defaults to Drive path or script folder)")
    parser.add_argument("--classes", type=str, default="electric scooter,coffee cup with logo,dog,person",
                        help="Comma-separated text classes for YOLO-World")
    parser.add_argument("--imgsz", type=int, default=640, help="Inference image size")
    parser.add_argument("--conf", type=float, default=0.25, help="Confidence threshold")
    parser.add_argument("--download-weights", action="store_true", help="Download weights if missing")
    parser.add_argument("--download-demo", action="store_true", help="Download demo images if missing")
    parser.add_argument("--auto-install", dest="auto_install", action="store_true", default=True,
                        help="Auto-install ultralytics if missing (requires internet). Disable with --no-auto-install.")
    parser.add_argument("--no-auto-install", dest="auto_install", action="store_false")
    return parser.parse_args(argv)


def main(argv: list[str]) -> int:
    args = parse_args(argv)
    root = resolve_root(args.root)
    print(f"[info] using root: {root}")

    prepare_assets(root, download_weights=args.download_weights, download_demo=args.download_demo)
    yw_model, _ = load_models(root, auto_install=args.auto_install)

    demo_dir = root / "data" / "demo"
    classes = [c.strip() for c in args.classes.split(",") if c.strip()]
    run_inference(yw_model, demo_dir, classes, imgsz=args.imgsz, conf=args.conf)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
