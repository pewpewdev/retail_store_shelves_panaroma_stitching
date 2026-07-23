# progressive_ransac.py
# ---------------------------------------------------------------
#Author : Prerana Bora
# Progressive RANSAC for product-level retail stitching.
# Adapted from Jia et al. 2024, "Semantic Aware Stitching for
# Panorama", Sensors 24(11):3512, Section 2.1.2.
#
# ADAPTATION according to RetailGlue:
#   - operates on PRODUCT CENTROIDS (YOLO bbox centers)
#   - adds DINOv3 embedding-similarity + LightGlue-confidence filters
#   - groups planes by REPROJECTION ERROR to the estimated similarity
#     (not by centroid proximity)
#   - resolution-independent thresholds via coordinate normalization
# ---------------------------------------------------------------

from dataclasses import dataclass, field
from typing import Optional, List, Dict
import numpy as np
import cv2


MIN_POINTS      = 3        # similarity needs >=2
MAX_ITERS       = 5000
CONFIDENCE      = 0.999
REPROJ_INLIER   = 0.02     # normalized-coord reproj threshold 


# Data type for a product-level correspondence
@dataclass
class ProductMatch:
    centroid_A: np.ndarray            # (x, y) YOLO bbox center, image A
    centroid_B: np.ndarray            # (x, y) YOLO bbox center, image B
    embedding_sim: float              # DINOv3 cosine similarity (0..1)
    lightglue_score: float = 1.0      # LightGlue match confidence (0..1)
    bbox_A: Optional[np.ndarray] = None   
    bbox_B: Optional[np.ndarray] = None   



def _pts(matches):
    A = np.array([np.asarray(m.centroid_A, np.float32) for m in matches],
                 dtype=np.float32)
    B = np.array([np.asarray(m.centroid_B, np.float32) for m in matches],
                 dtype=np.float32)
    return A, B


def _normalize(pts, w, h):
    """Change #1: resolution-independent normalization to [0,1]."""
    out = pts.copy().astype(np.float32)
    out[:, 0] /= max(float(w), 1.0)
    out[:, 1] /= max(float(h), 1.0)
    return out


def mean_centroid(matches):
    A, _ = _pts(matches)
    return A.mean(axis=0)


def inlier_ratio(remaining, total):
    return len(remaining) / max(total, 1)


def _apply_similarity(S, pts):
    """Apply a 2x3 affine/similarity transform to Nx2 points."""
    ones = np.ones((len(pts), 1), np.float32)
    hom = np.hstack([pts.astype(np.float32), ones])   # Nx3
    return (S @ hom.T).T                              # Nx2


def _reproj_errors(S, A_norm, B_norm):
    """Change #2: per-match reprojection error under similarity S."""
    pred = _apply_similarity(S, A_norm)
    return np.linalg.norm(pred - B_norm, axis=1)      # N,


def ransac_similarity_norm(matches, w, h, threshold=0.02):
    """
    Fit a SIMILARITY (Jia Eq. 9) with RANSAC on NORMALIZED coords.
    Returns (S, inlier_mask). threshold is in normalized units.
    """
    if len(matches) < 2:
        return None, None
    A, B = _pts(matches)
    An, Bn = _normalize(A, w, h), _normalize(B, w, h)
    S, mask = cv2.estimateAffinePartial2D(
        An, Bn,
        method=cv2.RANSAC,
        ransacReprojThreshold=threshold,
        maxIters=MAX_ITERS,
        confidence=CONFIDENCE,
    )
    if S is None or mask is None:
        return None, None
    return S, mask.ravel().astype(bool)



# Progressive RANSAC (Jia et al. 2024, Section 2.1.2)

def progressive_ransac(matches: List[ProductMatch],
                       image_size,                    # (width, height)
                       lenient_thresh: float = 0.05,  # Jia Step 1 (normalized)
                       strict_thresh: float = 0.02,   # Jia Step 2 (normalized)
                       min_inlier_ratio: float = 0.30,# Jia Step 4 (stop)
                       emb_sim_thresh: Optional[float] = None,   
                       lg_score_thresh: Optional[float] = None   
                       ) -> List[Dict]:
    """
    Groups product correspondences by shelf plane, returning a similarity
    transform + quality metrics per plane.

    Args:
        matches: list[ProductMatch].
        image_size: (width, height) for coordinate normalization.
        lenient_thresh / strict_thresh: normalized reproj thresholds.
        min_inlier_ratio: stop when remaining/total drops below this.
        emb_sim_thresh: if set, drop matches with DINOv3 sim below it.
        lg_score_thresh: if set, drop matches with LightGlue score below it.

    Returns:
        list of dicts (sorted by score, best first):
          {inliers, center, S, score, mean_error, n_inliers}
    """
    w, h = image_size
    total = len(matches)
    if total < MIN_POINTS:
        return []


    if emb_sim_thresh is not None:
        matches = [m for m in matches if m.embedding_sim >= emb_sim_thresh]
    if lg_score_thresh is not None:
        matches = [m for m in matches if m.lightglue_score >= lg_score_thresh]
    if len(matches) < MIN_POINTS:
        return []

    # Step 1: lenient RANSAC removes gross outliers 
    S0, mask0 = ransac_similarity_norm(matches, w, h, lenient_thresh)
    if S0 is not None and mask0 is not None:
        matches = [m for m, keep in zip(matches, mask0) if keep]
    if len(matches) < MIN_POINTS:
        return []

    plane_groups = []
    remaining = matches

    # Step 4: iterate until inlier ratio < min_inlier_ratio 
    while inlier_ratio(remaining, total) >= min_inlier_ratio:

        # Step 2: strict RANSAC 
        S, mask = ransac_similarity_norm(remaining, w, h, strict_thresh)
        if S is None or mask is None or mask.sum() < MIN_POINTS:
            break

        inliers = [m for m, keep in zip(remaining, mask) if keep]

        A, B = _pts(inliers)
        An, Bn = _normalize(A, w, h), _normalize(B, w, h)
        errs = _reproj_errors(S, An, Bn)
        mean_error = float(errs.mean())

        # plane quality score 
        mean_emb = float(np.mean([m.embedding_sim for m in inliers]))
        score = (len(inliers) * mean_emb) / (mean_error + 1e-6)

        center = mean_centroid(inliers)

        plane_groups.append({
            "inliers": inliers,
            "center": center,
            "S": S,                     
            "score": score,             
            "mean_error": mean_error,   
            "n_inliers": len(inliers),
        })

        # remove points by REPROJECTION ERROR, not distance
        Ar, Br = _pts(remaining)
        Arn, Brn = _normalize(Ar, w, h), _normalize(Br, w, h)
        rem_errs = _reproj_errors(S, Arn, Brn)
        remaining = [m for m, e in zip(remaining, rem_errs)
                     if e > REPROJ_INLIER]   

        if len(remaining) < MIN_POINTS:
            break

    # return sorted by quality, best plane first
    plane_groups.sort(key=lambda g: g["score"], reverse=True)
    return plane_groups


