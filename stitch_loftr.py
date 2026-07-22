import cv2
import numpy as np
import os
import glob
import torch
import argparse
import kornia
from kornia.feature import LoFTR


# ─────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {DEVICE}")

# LoFTR works best at this resolution
LOFTR_IMAGE_SIZE = (480, 640)  # (H, W)


# ─────────────────────────────────────────────
# MODEL
# ─────────────────────────────────────────────
def load_model(weights: str = "indoor") -> LoFTR:
    """
    Load pretrained LoFTR model.
    weights: 'indoor' or 'outdoor'
    - Use 'indoor'  for store/retail environments
    - Use 'outdoor' for open scenes
    """
    print(f"Loading LoFTR model (weights='{weights}')...")
    model = LoFTR(pretrained=weights).eval().to(DEVICE)
    print("LoFTR model loaded.")
    return model


# ─────────────────────────────────────────────
# IMAGE LOADING
# ─────────────────────────────────────────────
def load_images_cv2(image_dir: str):
    """Load images using OpenCV."""
    extensions = ["*.jpg", "*.jpeg", "*.png"]
    paths = []
    for ext in extensions:
        paths.extend(glob.glob(os.path.join(image_dir, ext)))
    paths = sorted(paths)

    images = []
    valid_paths = []
    for p in paths:
        img = cv2.imread(p)
        if img is not None:
            images.append(img)
            valid_paths.append(p)
            print(f"  Loaded: {os.path.basename(p)} | shape: {img.shape}")
        else:
            print(f"  WARNING: Could not load {p}")

    return images, valid_paths


def cv2_to_loftr_tensor(img_bgr: np.ndarray,
                         size: tuple = LOFTR_IMAGE_SIZE) -> tuple:
    """
    Convert BGR OpenCV image to LoFTR-compatible grayscale tensor.
    LoFTR expects: float32 grayscale tensor [1, 1, H, W], range [0, 1]

    Returns:
        tensor     : torch.Tensor [1, 1, H, W]
        scale      : (scale_x, scale_y) to map keypoints back to original size
    """
    orig_h, orig_w = img_bgr.shape[:2]

    # Resize to LoFTR input size
    img_resized = cv2.resize(img_bgr, (size[1], size[0]))  # (W, H)

    # Convert to grayscale
    gray = cv2.cvtColor(img_resized, cv2.COLOR_BGR2GRAY)

    # To float tensor [1, 1, H, W]
    tensor = torch.from_numpy(gray).float() / 255.0
    tensor = tensor.unsqueeze(0).unsqueeze(0).to(DEVICE)  # [1, 1, H, W]

    # Scale factors to map back to original image coords
    scale_x = orig_w / size[1]
    scale_y = orig_h / size[0]

    return tensor, (scale_x, scale_y)


# ─────────────────────────────────────────────
# FEATURE MATCHING WITH LoFTR
# ─────────────────────────────────────────────
def match_loftr(model: LoFTR,
                img0_bgr: np.ndarray,
                img1_bgr: np.ndarray,
                confidence_threshold: float = 0.5) -> tuple:
    """
    Match two images using LoFTR (dense matcher — no keypoint detection needed).

    LoFTR directly outputs:
        - keypoints in both images
        - confidence scores per match

    Returns:
        src_pts : np.ndarray (N, 2) — matched pts in img0 (original coords)
        dst_pts : np.ndarray (N, 2) — matched pts in img1 (original coords)
        n_matches: int
    """
    # Prepare tensors
    tensor0, scale0 = cv2_to_loftr_tensor(img0_bgr)
    tensor1, scale1 = cv2_to_loftr_tensor(img1_bgr)

    input_dict = {
        "image0": tensor0,  # [1, 1, H, W]
        "image1": tensor1,  # [1, 1, H, W]
    }

    print("    Running LoFTR matching...")
    with torch.no_grad():
        correspondences = model(input_dict)

    # Extract outputs
    kpts0 = correspondences["keypoints0"].cpu().numpy()  # (N, 2) in resized coords
    kpts1 = correspondences["keypoints1"].cpu().numpy()  # (N, 2) in resized coords
    confidence = correspondences["confidence"].cpu().numpy()  # (N,)

    print(f"    Raw matches: {len(kpts0)} | "
          f"avg confidence: {confidence.mean():.3f}" if len(confidence) > 0
          else "    No matches found.")

    # Filter by confidence threshold
    mask = confidence >= confidence_threshold
    kpts0 = kpts0[mask]
    kpts1 = kpts1[mask]
    confidence = confidence[mask]

    print(f"    Matches after confidence filter (>={confidence_threshold}): {len(kpts0)}")

    if len(kpts0) < 10:
        return None, None, 0

    # Scale keypoints back to original image coordinates
    kpts0[:, 0] *= scale0[0]  # x * scale_x
    kpts0[:, 1] *= scale0[1]  # y * scale_y
    kpts1[:, 0] *= scale1[0]
    kpts1[:, 1] *= scale1[1]

    return kpts0, kpts1, len(kpts0)


# ─────────────────────────────────────────────
# HOMOGRAPHY
# ─────────────────────────────────────────────
def compute_homography(src_pts: np.ndarray,
                        dst_pts: np.ndarray) -> np.ndarray:
    """Compute homography using RANSAC."""
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
        print("    ❌ Homography estimation failed.")
        return None

    return H


# ─────────────────────────────────────────────
# WARPING & BLENDING
# ─────────────────────────────────────────────
def warp_and_blend(img_src: np.ndarray,
                   img_dst: np.ndarray,
                   H: np.ndarray) -> np.ndarray:
    """
    Warp img_src using H, blend with img_dst using
    horizontal gradient alpha in overlap region.
    """
    h_src, w_src = img_src.shape[:2]
    h_dst, w_dst = img_dst.shape[:2]

    # Project corners to find canvas size
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

    # Translation matrix
    T = np.array([
        [1, 0, -xmin],
        [0, 1, -ymin],
        [0, 0, 1]
    ], dtype=np.float64)

    # Warp src
    warped_src = cv2.warpPerspective(img_src, T @ H, (canvas_w, canvas_h))

    # Place dst on canvas
    canvas_dst = np.zeros((canvas_h, canvas_w, 3), dtype=np.uint8)
    y_off, x_off = -ymin, -xmin
    canvas_dst[y_off: y_off + h_dst, x_off: x_off + w_dst] = img_dst

    # Build masks
    mask_src = (cv2.cvtColor(warped_src, cv2.COLOR_BGR2GRAY) > 0).astype(np.float32)
    mask_dst = (cv2.cvtColor(canvas_dst, cv2.COLOR_BGR2GRAY) > 0).astype(np.float32)
    overlap = mask_src * mask_dst

    # Horizontal gradient alpha blend in overlap
    alpha = np.zeros_like(mask_src)
    ys, xs = np.where(overlap > 0)
    if len(xs) > 0:
        x_lo, x_hi = xs.min(), xs.max()
        span = max(x_hi - x_lo, 1)
        for y, x in zip(ys, xs):
            alpha[y, x] = (x - x_lo) / span  # 0=src, 1=dst

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
def stitch_images_loftr(images_cv2: list,
                         model: LoFTR,
                         confidence_threshold: float = 0.5) -> tuple:
    """
    Sequential LoFTR stitching pipeline:
    panorama = img0
    panorama = stitch(panorama, img1)
    panorama = stitch(panorama, img2) ...
    """
    if len(images_cv2) < 2:
        print("  Need at least 2 images.")
        return False, None

    panorama = images_cv2[0].copy()

    for i in range(1, len(images_cv2)):
        print(f"\n  ── Stitching image {i+1}/{len(images_cv2)} ──")

        img_next = images_cv2[i]

        # Match with LoFTR
        src_pts, dst_pts, n_matches = match_loftr(
            model, panorama, img_next,
            confidence_threshold=confidence_threshold
        )

        if src_pts is None or n_matches < 10:
            print(f"  ❌ Not enough matches ({n_matches}).")
            return False, None

        # Compute homography
        print("    Computing homography (RANSAC)...")
        H = compute_homography(src_pts, dst_pts)

        if H is None:
            print(f"  ❌ Homography failed for image {i+1}.")
            return False, None

        # Warp and blend
        print("    Warping and blending...")
        panorama = warp_and_blend(panorama, img_next, H)
        panorama = crop_black_borders(panorama)
        print(f"    Panorama size: {panorama.shape[1]}x{panorama.shape[0]} px")

    return True, panorama


# ─────────────────────────────────────────────
# STORE PIPELINE
# ─────────────────────────────────────────────
def stitch_store(store_dir: str,
                  output_dir: str,
                  store_name: str,
                  model: LoFTR,
                  confidence_threshold: float = 0.5):
    """Full pipeline for one store."""
    image_dir = os.path.join(store_dir, "images")
    print(f"\n{'='*60}")
    print(f"Store: {store_name}")
    print(f"{'='*60}")

    images_cv2, paths = load_images_cv2(image_dir)

    if len(images_cv2) < 2:
        print("  Not enough images.")
        return

    success, panorama = stitch_images_loftr(
        images_cv2, model,
        confidence_threshold=confidence_threshold
    )

    if success and panorama is not None:
        os.makedirs(output_dir, exist_ok=True)
        out_path = os.path.join(output_dir, f"{store_name}_loftr_panorama.jpg")
        cv2.imwrite(out_path, panorama, [cv2.IMWRITE_JPEG_QUALITY, 95])
        print(f"\n✅ Saved: {out_path}")
        print(f"   Size: {panorama.shape[1]}x{panorama.shape[0]} px")
    else:
        print(f"\n❌ Stitching failed for {store_name}")


# ─────────────────────────────────────────────
# ENTRY POINT
# ─────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="LoFTR Panorama Stitcher")
    parser.add_argument(
        "--base_dir",
        default="/Users/testuser/Desktop/store_panaroma_stiching/stitching_assignment_data",
        help="Base directory containing store_1, store_2, ..."
    )
    parser.add_argument(
        "--output_dir",
        default="/Users/testuser/Desktop/store_panaroma_stiching/output_loftr",
        help="Output directory for panoramas"
    )
    parser.add_argument(
        "--weights",
        default="indoor",
        choices=["indoor", "outdoor"],
        help="LoFTR pretrained weights (indoor recommended for retail stores)"
    )
    parser.add_argument(
        "--confidence",
        type=float,
        default=0.5,
        help="LoFTR match confidence threshold (0.0 - 1.0)"
    )
    parser.add_argument(
        "--loftr_height",
        type=int,
        default=480,
        help="LoFTR input image height"
    )
    parser.add_argument(
        "--loftr_width",
        type=int,
        default=640,
        help="LoFTR input image width"
    )
    args = parser.parse_args()

    # Override global LoFTR size if provided
    global LOFTR_IMAGE_SIZE
    LOFTR_IMAGE_SIZE = (args.loftr_height, args.loftr_width)
    print(f"LoFTR input size: {LOFTR_IMAGE_SIZE}")

    # Load model ONCE — reuse across all stores
    model = load_model(weights=args.weights)

    # Auto-discover store directories
    store_dirs = sorted(glob.glob(os.path.join(args.base_dir, "store_*")))

    if not store_dirs:
        print("❌ No store directories found.")
        return

    print(f"\nFound {len(store_dirs)} store(s): "
          f"{[os.path.basename(s) for s in store_dirs]}")

    # Process each store
    for store_dir in store_dirs:
        store_name = os.path.basename(store_dir)
        stitch_store(
            store_dir=store_dir,
            output_dir=args.output_dir,
            store_name=store_name,
            model=model,
            confidence_threshold=args.confidence
        )

    print(f"\n{'='*60}")
    print("✅ All stores processed.")
    print(f"   Outputs saved to: {args.output_dir}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()


