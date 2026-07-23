
import argparse
import copy
import glob
import logging
import os

import numpy as np
import torch
from PIL import Image, ImageOps

from retailglue.config import get_config, resolve_path
from retailglue.detector import YOLODetector
from retailglue.embeddings import extract_dino_embeddings, DINO_VARIANTS, BF_TO_DINO_VARIANT
from retailglue.stitching.stitcher import ImageStitcher

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger("batch_stitch")

DINO_MODEL_NAMES = tuple(DINO_VARIANTS.keys()) + tuple(BF_TO_DINO_VARIANT.keys())
IMG_EXTS = ("*.jpg", "*.jpeg", "*.png", "*.JPG", "*.JPEG", "*.PNG")


def load_images(images_dir):
    paths = []
    for ext in IMG_EXTS:
        paths.extend(glob.glob(os.path.join(images_dir, ext)))
    paths = sorted(paths)
    images = [np.array(ImageOps.exif_transpose(Image.open(p).convert("RGB"))) for p in paths]
    return images, paths


def build_detector(config, device):
    weights = resolve_path(getattr(config.detector, "sku_yolo_weights",
                                    "weights/sku_yolo/best_sku110k.pt"))
    if not os.path.exists(weights):
        raise FileNotFoundError(
            f"SKU-YOLO weights not found at {weights}. Download them first, see README "
            f"(git clone https://huggingface.co/arda92/retailglue-model-weights weights)."
        )
    return YOLODetector(weights, device=device)


def build_stitcher(config, model_name, device):
    cfg = copy.copy(config.stitching)
    cfg.model_name = model_name
    cfg.device = device
    cfg.verbose = False
    return ImageStitcher(config=cfg)


def detect_and_embed(images, detector, config, device, model_name):
    detections = detector.detect(images, conf=getattr(config.detector, "confidence", 0.25))
    if model_name in DINO_MODEL_NAMES:
        dino_variant = BF_TO_DINO_VARIANT.get(model_name, model_name)
        dino_weights = None
        dw = getattr(config.embeddings, "dino_weights", None)
        if dw is not None:
            w = getattr(dw, dino_variant, None)
            if w:
                dino_weights = resolve_path(w)
        crop_margin = getattr(config.embeddings, "crop_margin", 10)
        extract_dino_embeddings(images, detections, device, variant=model_name,
                                 weights_path=dino_weights, crop_margin=crop_margin)
    return detections


def stitch_store(store_dir, config, detector, stitcher, device, model_name):
    images_dir = os.path.join(store_dir, "images")
    images, paths = load_images(images_dir)
    if len(images) < 2:
        logger.warning(f"{store_dir}: fewer than 2 images found, skipping")
        return

    detections = detect_and_embed(images, detector, config, device, model_name)
    result = stitcher.stitch_images(images, detections=detections)

    # verbose=False -> result is (panoramas, det_results) when detections were passed
    panoramas, _det_results = result if isinstance(result, tuple) else (result, None)
    if not isinstance(panoramas, list):
        panoramas = [panoramas]

    result_dir = os.path.join(store_dir, "result")
    os.makedirs(result_dir, exist_ok=True)
    for i, pano in enumerate(panoramas):
        out_path = os.path.join(result_dir, f"panorama_{i}.jpg")
        Image.fromarray(pano).save(out_path, quality=95)
        logger.info(f"  saved {out_path}")

    logger.info(f"{os.path.basename(store_dir)}: {len(images)} images -> "
                f"{len(panoramas)} panorama(s)")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root", required=True,
                         help="Path to stitching_assignment_data (contains store_1, store_2, ...)")
    parser.add_argument("--model_name", default="lightglue_dinov3_vits",
                         choices=["lightglue_dinov3_vits", "lightglue_dinov3_vitb",
                                  "lightglue_dinov3_vitl", "lightglue_dinov2_vits"])
    parser.add_argument("--device", default=None, help="cpu | cuda | mps (auto-detect if omitted)")
    args = parser.parse_args()

    device = args.device or (
        "cuda" if torch.cuda.is_available()
        else "mps" if torch.backends.mps.is_available()
        else "cpu"
    )
    logger.info(f"Using device: {device}")

    config = get_config()
    detector = build_detector(config, device)
    stitcher = build_stitcher(config, args.model_name, device)

    store_dirs = sorted(
        d for d in glob.glob(os.path.join(args.data_root, "store_*"))
        if os.path.isdir(d)
    )
    if not store_dirs:
        logger.error(f"No store_* folders found under {args.data_root}")
        return

    for store_dir in store_dirs:
        logger.info(f"Processing {store_dir} ...")
        try:
            stitch_store(store_dir, config, detector, stitcher, device, args.model_name)
        except Exception as e:
            logger.exception(f"Failed on {store_dir}: {e}")


if __name__ == "__main__":
    main()