import cv2
import numpy as np
import os
import glob
import torch
import argparse
from pathlib import Path
from lightglue import LightGlue, SuperPoint
from lightglue.utils import load_image, rbd


# ─────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {DEVICE}")


# ─────────────────────────────────────────────
# MODELS
# ─────────────────────────────────────────────
def load_models():
    """Load SuperPoint extractor + LightGlue matcher."""
    print("Loading SuperPoint + LightGlue models...")
    extractor = SuperPoint(max_num_keypoints=4096).eval().to(DEVICE)
    matcher = LightGlue(features="superpoint").eval().to(DEVICE)
    print("Models loaded.")
    return extractor, matcher


# ─────────────────────────────────────────────
# IMAGE LOADING
# ─────────────────────────────────────────────
def load_images_cv2(image_dir: str):
    """Load images using OpenCV for warping/blending."""
    extensions = ["*.jpg", "*.jpeg", "*.png"]
    paths = []
    for ext in extensions:
        paths.extend(glob.glob(os.path.join(image_dir, ext)))
    paths = sorted(paths)

    images_cv2 = []
    valid_paths = []
    for p in paths:
        img = cv2.imread(p)
        if img is not None:
            images_cv2.append(img)
            valid_paths.append(p)
            print(f"  Loaded: {os.path.basename(p)} | shape: {img.shape}")
    return images_cv2, valid_paths


def cv2_to_lightglue_tensor(img_bgr: np.ndarray) -> torch.Tensor:
    """
    Convert BGR OpenCV image to LightGlue-compatible tensor.
    LightGlue expects: float32 RGB tensor, shape [C, H, W], range [0, 1]
    """
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    tensor = torch.from_numpy(img_rgb).float() / 255.0
    tensor = tensor.permute(2, 0, 1)  # HWC -> CHW
    return tensor.to(DEVICE)


# ─────────────────────────────────────────────
# FEATURE EXTRACTION & MATCHING
# ─────────────────────────────────────────────
def extract_features(extractor, img_tensor: torch.Tensor) -> dict:
    """Extract SuperPoint features from a single image tensor."""
    with torch.no_grad():
        feats = extractor.extract(img_tensor.unsqueeze(0))  # add batch dim
    return feats


def match_features(matcher, feats0: dict, feats1: dict) -> dict:
    """Match features between two images using LightGlue."""
    with torch.no_grad():
        matches = matcher({"image0": feats0, "image1": feats1})
    return matches


def get_matched_keypoints(feats0, feats1, matches_dict):
    """
    Extract matched keypoint coordinates.
    Returns: src_pts, dst_pts as numpy arrays of shape (N, 2)
    """
    # Remove batch dimension
    feats0_r = rbd(feats0)
    feats1_r = rbd(feats1)
    matches_r = rbd(matches_dict)

    kpts0 = feats0_r["keypoints"].cpu().numpy()  # (N, 2)
    kpts1 = feats1_r["keypoints"].cpu().numpy()  # (M, 2)
    matched_indices = matches_r["matches"].cpu().numpy()  # (K, 2)
    scores = matches_r["scores"].cpu().numpy()  # (K,)

    print(f"    Keypoints: img0={len(kpts0)}, img1={len(kpts1)}")
    print(f"    Matches found: {len(matched_indices)} | "
          f"avg score: {scores.mean():.3f}" if len(scores) > 0 else "    No matches.")

    if len(matched_indices) < 10:
        return None, None, 0

    src_pts = kpts0[matched_indices[:, 0]]  # (K, 2)
    dst_pts = kpts1[matched_indices[:, 1]]  # (K, 2)

    return src_pts, dst_pts, len(matched_indices)


# ─────────────────────────────────────────────
# HOMOGRAPHY
# ─────────────────────────────────────────────
def compute_homography_from_pts(src_pts: np.ndarray,
                                 dst_pts: np.ndarray) -> np.ndarray:
    """Compute homography using RANSAC from matched point pairs."""
    src = src_pts.reshape(-1, 1, 2).astype(np.float32)
    dst = dst_pts.reshape(-1, 1, 2).astype(np.float32)

    H, mask = cv2.findHomography(
        src, dst,
        method=cv2.RANSAC,
        ransacReprojThreshold=3.0,
        maxIters=10000,
        confidence=0.9995
    )

    if mask is not None:
        inliers = int(mask.sum())
        print(f"    RANSAC inliers: {inliers}/{len(src_pts)}")
    else:
        print("    Homography estimation failed.")

    return H


# ─────────────────────────────────────────────
# WARPING & BLENDING
# ─────────────────────────────────────────────
def warp_and_blend(img_src: np.ndarray,
                   img_dst: np.ndarray,
                   H: np.ndarray) -> np.ndarray:
    """
    Warp img_src onto img_dst using homography H,
    then blend with linear alpha in overlap region.
    """
    h_src, w_src = img_src.shape[:2]
    h_dst, w_dst = img_dst.shape[:2]

    # Project corners of img_src
    corners_src = np.float32([
        [0, 0], [w_src, 0], [w_src, h_src], [0, h_src]
    ]).reshape(-1, 1, 2)
    corners_dst = np.float32([
        [0, 0], [w_dst, 0], [w_dst, h_dst], [0, h_dst]
    ]).reshape(-1, 1, 2)

    warped_corners = cv2.perspectiveTransform(corners_src, H)
    all_corners = np.concatenate([corners_dst, warped_corners], axis=0)

    [xmin, ymin] = np.int32(all_corners.min(axis=0).ravel() - 0.5)
    [xmax, ymax] = np.int32(all_corners.max(axis=0).ravel() + 0.5)

    canvas_w = xmax - xmin
    canvas_h = ymax - ymin

    # Translation to shift everything to positive coords
    T = np.array([
        [1, 0, -xmin],
        [0, 1, -ymin],
        [0, 0, 1]
    ], dtype=np.float64)

    # Warp src image into canvas space
    warped_src = cv2.warpPerspective(img_src, T @ H, (canvas_w, canvas_h))

    # Place dst image into canvas
    canvas_dst = np.zeros((canvas_h, canvas_w, 3), dtype=np.uint8)
    y_off, x_off = -ymin, -xmin
    canvas_dst[y_off: y_off + h_dst, x_off: x_off + w_dst] = img_dst

    # Build masks
    mask_src = (cv2.cvtColor(warped_src, cv2.COLOR_BGR2GRAY) > 0).astype(np.float32)
    mask_dst = (cv2.cvtColor(canvas_dst, cv2.COLOR_BGR2GRAY) > 0).astype(np.float32)

    overlap = mask_src * mask_dst  # 1 where both images exist

    # ── Alpha blend in overlap, hard copy elsewhere ──
    # Horizontal gradient alpha in overlap zone
    alpha = np.zeros_like(mask_src)
    ys, xs = np.where(overlap > 0)
    if len(xs) > 0:
        x_lo, x_hi = xs.min(), xs.max()
        span = max(x_hi - x_lo, 1)
        for y, x in zip(ys, xs):
            alpha[y, x] = (x - x_lo) / span  # 0=src side, 1=dst side

    alpha3 = np.stack([alpha, alpha, alpha], axis=2)

    result = np.where(
        overlap[:, :, np.newaxis] > 0,
        (warped_src * (1 - alpha3) + canvas_dst * alpha3).astype(np.uint8),
        np.where(
            mask_src[:, :, np.newaxis] > 0,
            warped_src,
            canvas_dst
        )
    ).astype(np.uint8)

    return result


def crop_black_borders(img: np.ndarray) -> np.ndarray:
    """Crop black borders from panorama."""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 1, 255, cv2.THRESH_BINARY)
    coords = cv2.findNonZero(thresh)
    if coords is None:
        return img
    x, y, w, h = cv2.boundingRect(coords)
    return img[y: y + h, x: x + w]


# ─────────────────────────────────────────────
# MAIN STITCHING PIPELINE
# ─────────────────────────────────────────────
def stitch_images_superpoint(images_cv2: list,
                              extractor,
                              matcher) -> tuple[bool, np.ndarray]:
    """
    Sequential stitching using SuperPoint + LightGlue:
    img0 → stitch with img1 → stitch result with img2 → ...
    """
    if len(images_cv2) < 2:
        print("  Need at least 2 images.")
        return False, None

    panorama = images_cv2[0].copy()

    for i in range(1, len(images_cv2)):
        print(f"\n  ── Stitching image {i+1}/{len(images_cv2)} ──")

        img_next = images_cv2[i]

        # Convert both to tensors
        tensor_pano = cv2_to_lightglue_tensor(panorama)
        tensor_next = cv2_to_lightglue_tensor(img_next)

        # Extract features
        print("    Extracting SuperPoint features...")
        feats_pano = extract_features(extractor, tensor_pano)
        feats_next = extract_features(extractor, tensor_next)

        # Match
        print("    Matching with LightGlue...")
        matches_dict = match_features(matcher, feats_pano, feats_next)

        # Get matched point coordinates
        src_pts, dst_pts, n_matches = get_matched_keypoints(
            feats_pano, feats_next, matches_dict
        )

        if src_pts is None or n_matches < 10:
            print(f"  ❌ Not enough matches ({n_matches}). Cannot stitch image {i+1}.")
            return False, None

        # Compute homography
        print("    Computing homography (RANSAC)...")
        H = compute_homography_from_pts(src_pts, dst_pts)

        if H is None:
            print(f"  ❌ Homography failed for image {i+1}.")
            return False, None

        # Warp and blend
        print("    Warping and blending...")
        panorama = warp_and_blend(panorama, img_next, H)
        panorama = crop_black_borders(panorama)
        print(f"    Panorama size: {panorama.shape[1]}x{panorama.shape[0]} px")

    return True, panorama


def stitch_store(store_dir: str, output_dir: str,
                  store_name: str, extractor, matcher):
    """Full pipeline for one store."""
    image_dir = os.path.join(store_dir, "images")
    print(f"\n{'='*60}")
    print(f"Store: {store_name}")
    print(f"{'='*60}")

    images_cv2, paths = load_images_cv2(image_dir)

    if len(images_cv2) < 2:
        print("  Not enough images.")
        return

    success, panorama = stitch_images_superpoint(images_cv2, extractor, matcher)

    if success and panorama is not None:
        os.makedirs(output_dir, exist_ok=True)
        out_path = os.path.join(output_dir, f"{store_name}_superpoint_panorama.jpg")
        cv2.imwrite(out_path, panorama, [cv2.IMWRITE_JPEG_QUALITY, 95])
        print(f"\n✅ Saved: {out_path}")
        print(f"   Size: {panorama.shape[1]}x{panorama.shape[0]} px")
    else:
        print(f"\n❌ Stitching failed for {store_name}")


# ─────────────────────────────────────────────
# ENTRY POINT
# ─────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="SuperPoint Panorama Stitcher")
    parser.add_argument(
        "--base_dir",
        default="/Users/testuser/Desktop/store_panaroma_stiching/stitching_assignment_data",
    )
    parser.add_argument(
        "--output_dir",
        default="/Users/testuser/Desktop/store_panaroma_stiching/output_superpoint",
    )
    args = parser.parse_args()

    # Load models ONCE — reuse across all stores
    extractor, matcher = load_models()

    store_dirs = sorted(glob.glob(os.path.join(args.base_dir, "store_*")))
    if not store_dirs:
        print("No store directories found.")
        return

    print(f"\nFound stores: {[os.path.basename(s) for s in store_dirs]}")

    for store_dir in store_dirs:
        store_name = os.path.basename(store_dir)
        stitch_store(store_dir, args.output_dir,
                     store_name, extractor, matcher)

    print(f"\n{'='*60}")
    print("Done.")


if __name__ == "__main__":
    main()

