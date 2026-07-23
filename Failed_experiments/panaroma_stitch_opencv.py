import cv2
import numpy as np
import os
import glob
import argparse
from pathlib import Path


def load_images(image_dir: str) -> tuple[list, list]:
    extensions = ["*.jpg", "*.jpeg", "*.png"]
    image_paths = []
    for ext in extensions:
        image_paths.extend(glob.glob(os.path.join(image_dir, ext)))
    
    image_paths = sorted(image_paths)  
    images = []
    valid_paths = []
    
    for path in image_paths:
        img = cv2.imread(path)
        if img is not None:
            images.append(img)
            valid_paths.append(path)
            print(f"  Loaded: {os.path.basename(path)} | shape: {img.shape}")
        else:
            print(f"  WARNING: Could not load {path}")
    
    return images, valid_paths


def stitch_opencv(images: list) -> tuple[bool, np.ndarray]:
    """
    Method 1: OpenCV built-in Stitcher (best for standard panoramas).
    Uses internal feature detection, matching, bundle adjustment, blending.
    """
    print("\n[Method 1] Trying cv2.Stitcher...")
    stitcher = cv2.Stitcher.create(cv2.Stitcher_PANORAMA)
    
    # Try with original images
    status, panorama = stitcher.stitch(images)
    
    if status == cv2.Stitcher_OK:
        print("  cv2.Stitcher succeeded.")
        return True, panorama
    
    # Retry with resized images 
    print(f"  Stitcher failed (status={status}). Retrying with resized images...")
    resized = [cv2.resize(img, (img.shape[1] // 2, img.shape[0] // 2)) 
               for img in images]
    status, panorama = stitcher.stitch(resized)
    
    if status == cv2.Stitcher_OK:
        print("  cv2.Stitcher succeeded with resized images.")
        return True, panorama
    
    status_map = {
        cv2.Stitcher_ERR_NEED_MORE_IMGS: "Need more images",
        cv2.Stitcher_ERR_HOMOGRAPHY_EST_FAIL: "Homography estimation failed",
        cv2.Stitcher_ERR_CAMERA_PARAMS_ADJUST_FAIL: "Camera params adjustment failed",
    }
    reason = status_map.get(status, f"Unknown error code {status}")
    print(f"  cv2.Stitcher failed: {reason}")
    return False, None


def detect_and_match_features(img1: np.ndarray, img2: np.ndarray):
    """
    Detect SIFT features and match using FLANN-based matcher.
    Falls back to BFMatcher if FLANN fails.
    """
    # Convert to grayscale
    gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

    # SIFT detector
    sift = cv2.SIFT_create(nfeatures=5000)
    kp1, des1 = sift.detectAndCompute(gray1, None)
    kp2, des2 = sift.detectAndCompute(gray2, None)
    
    print(f"    Features detected: img1={len(kp1)}, img2={len(kp2)}")

    if des1 is None or des2 is None or len(kp1) < 4 or len(kp2) < 4:
        return None, None, None, None

    # FLANN matcher
    try:
        FLANN_INDEX_KDTREE = 1
        index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
        search_params = dict(checks=100)
        flann = cv2.FlannBasedMatcher(index_params, search_params)
        matches = flann.knnMatch(des1, des2, k=2)
    except Exception:
        # Fallback to BFMatcher
        bf = cv2.BFMatcher(cv2.NORM_L2)
        matches = bf.knnMatch(des1, des2, k=2)

    # Lowe's ratio test
    good_matches = []
    for m_n in matches:
        if len(m_n) == 2:
            m, n = m_n
            if m.distance < 0.75 * n.distance:
                good_matches.append(m)

    print(f"    Good matches (Lowe's ratio test): {len(good_matches)}")
    return kp1, kp2, good_matches, None


def compute_homography(kp1, kp2, good_matches):
    """Compute homography using RANSAC."""
    if len(good_matches) < 10:
        print(f"    Not enough matches: {len(good_matches)} (need ≥ 10)")
        return None

    src_pts = np.float32(
        [kp1[m.queryIdx].pt for m in good_matches]
    ).reshape(-1, 1, 2)
    dst_pts = np.float32(
        [kp2[m.trainIdx].pt for m in good_matches]
    ).reshape(-1, 1, 2)

    H, mask = cv2.findHomography(
        src_pts, dst_pts,
        cv2.RANSAC,
        ransacReprojThreshold=5.0,
        maxIters=5000,
        confidence=0.995
    )
    
    inliers = int(mask.sum()) if mask is not None else 0
    print(f"    Homography inliers: {inliers}/{len(good_matches)}")
    return H


def warp_and_blend(img1: np.ndarray, img2: np.ndarray, H: np.ndarray) -> np.ndarray:
    """
    Warp img1 onto img2's plane and blend using multi-band or simple alpha blend.
    """
    h1, w1 = img1.shape[:2]
    h2, w2 = img2.shape[:2]

    # Compute canvas size by projecting corners of img1
    corners_img1 = np.float32([[0, 0], [w1, 0],
                                 [w1, h1], [0, h1]]).reshape(-1, 1, 2)
    corners_img2 = np.float32([[0, 0], [w2, 0],
                                 [w2, h2], [0, h2]]).reshape(-1, 1, 2)

    warped_corners = cv2.perspectiveTransform(corners_img1, H)
    all_corners = np.concatenate([corners_img2, warped_corners], axis=0)

    [xmin, ymin] = np.int32(all_corners.min(axis=0).ravel() - 0.5)
    [xmax, ymax] = np.int32(all_corners.max(axis=0).ravel() + 0.5)

    # Translation matrix to shift to positive coordinates
    translation = np.array([[1, 0, -xmin],
                             [0, 1, -ymin],
                             [0, 0, 1]], dtype=np.float64)

    canvas_w = xmax - xmin
    canvas_h = ymax - ymin

    # Warp img1 into canvas
    warped_img1 = cv2.warpPerspective(
        img1, translation @ H, (canvas_w, canvas_h)
    )

    # Place img2 into canvas
    canvas = warped_img1.copy()
    roi = canvas[-ymin: -ymin + h2, -xmin: -xmin + w2]
    
    # Simple blending: where warped_img1 is black, use img2
    mask_img2 = np.zeros_like(canvas[:, :, 0])
    mask_img2[-ymin: -ymin + h2, -xmin: -xmin + w2] = 255

    gray_warped = cv2.cvtColor(warped_img1, cv2.COLOR_BGR2GRAY)
    _, warped_mask = cv2.threshold(gray_warped, 1, 255, cv2.THRESH_BINARY)

    # Overlap region: blend 50/50
    overlap = cv2.bitwise_and(mask_img2, warped_mask)
    only_img2 = cv2.bitwise_and(mask_img2, cv2.bitwise_not(warped_mask))

    # Fill non-overlapping img2 region
    canvas[-ymin: -ymin + h2, -xmin: -xmin + w2] = np.where(
        only_img2[-ymin: -ymin + h2, -xmin: -xmin + w2, np.newaxis] > 0,
        img2,
        roi
    )

    # Blend overlap region
    overlap_roi = overlap[-ymin: -ymin + h2, -xmin: -xmin + w2]
    canvas[-ymin: -ymin + h2, -xmin: -xmin + w2] = np.where(
        overlap_roi[:, :, np.newaxis] > 0,
        cv2.addWeighted(
            roi, 0.5,
            img2, 0.5, 0
        ),
        canvas[-ymin: -ymin + h2, -xmin: -xmin + w2]
    )

    return canvas


def crop_black_borders(img: np.ndarray) -> np.ndarray:
    """Crop black borders from stitched panorama."""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 1, 255, cv2.THRESH_BINARY)
    coords = cv2.findNonZero(thresh)
    if coords is None:
        return img
    x, y, w, h = cv2.boundingRect(coords)
    return img[y:y+h, x:x+w]


def stitch_manual_sift(images: list) -> tuple[bool, np.ndarray]:
    """
    Method 2: Sequential SIFT + RANSAC homography stitching.
    Stitches images one by one from left to right.
    """
    
    if len(images) < 2:
        return False, None

    panorama = images[0].copy()

    for i in range(1, len(images)):
        print(f"  Stitching image {i+1}/{len(images)}...")
        kp1, kp2, good_matches, _ = detect_and_match_features(panorama, images[i])
        
        if good_matches is None:
            print(f"  Feature detection failed for image {i+1}")
            return False, None

        H = compute_homography(kp1, kp2, good_matches)
        if H is None:
            print(f"  Homography failed for image {i+1}")
            return False, None

        panorama = warp_and_blend(panorama, images[i], H)
        panorama = crop_black_borders(panorama)
        print(f"  Panorama size after step {i}: {panorama.shape}")

    return True, panorama


def stitch_store(store_dir: str, output_dir: str, store_name: str):
    image_dir = os.path.join(store_dir, "images")
    print(f"\n{'='*60}")
    print(f"Processing: {store_name}")
    print(f"Image directory: {image_dir}")
    print(f"{'='*60}")

    images, paths = load_images(image_dir)

    if len(images) < 2:
        print(f"  ERROR: Need at least 2 images, found {len(images)}")
        return

    output_path = os.path.join(output_dir, f"{store_name}_panorama.jpg")

    # --- Method 1: cv2.Stitcher ---
    success, panorama = stitch_opencv(images)

    # --- Method 2: Manual SIFT fallback ---
    if not success:
        success, panorama = stitch_manual_sift(images)

    if success and panorama is not None:
        panorama = crop_black_borders(panorama)
        os.makedirs(output_dir, exist_ok=True)
        cv2.imwrite(output_path, panorama, [cv2.IMWRITE_JPEG_QUALITY, 95])
        print(f"\n Panorama saved: {output_path}")
        print(f"   Final size: {panorama.shape[1]}x{panorama.shape[0]} px")
    else:
        print(f"\n All stitching methods failed for {store_name}")


def main():
    parser = argparse.ArgumentParser(description="Panorama Stitcher")
    parser.add_argument(
        "--base_dir",
        default="/Users/testuser/Desktop/store_panaroma_stiching/stitching_assignment_data",
        help="Base directory containing store_1, store_2, ..."
    )
    parser.add_argument(
        "--output_dir",
        default="/Users/testuser/Desktop/store_panaroma_stiching/output",
        help="Output directory for panoramas"
    )
    args = parser.parse_args()


    store_dirs = sorted(glob.glob(os.path.join(args.base_dir, "store_*")))

    if not store_dirs:
        print("No store directories found!")
        return

    print(f"Found {len(store_dirs)} store(s): {[os.path.basename(s) for s in store_dirs]}")

    for store_dir in store_dirs:
        store_name = os.path.basename(store_dir)
        stitch_store(store_dir, args.output_dir, store_name)

    print(f"\n{'='*60}")
    print("All stores processed.")


if __name__ == "__main__":
    main()

