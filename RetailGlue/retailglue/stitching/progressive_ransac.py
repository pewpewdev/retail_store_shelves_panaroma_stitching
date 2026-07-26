# # progressive_ransac.py
# # ---------------------------------------------------------------
# #Author : Prerana Bora
# # Progressive RANSAC for product-level retail stitching.
# # Adapted from Jia et al. 2024, "Semantic Aware Stitching for
# # Panorama", Sensors 24(11):3512, Section 2.1.2.
# #
# # ADAPTATION according to RetailGlue:
# #   - operates on PRODUCT CENTROIDS (YOLO bbox centers)
# #   - adds DINOv3 embedding-similarity + LightGlue-confidence filters
# #   - groups planes by REPROJECTION ERROR to the estimated similarity
# #     (not by centroid proximity)
# #   - resolution-independent thresholds via coordinate normalization
# # ---------------------------------------------------------------

# from dataclasses import dataclass, field
# from typing import Optional, List, Dict
# import numpy as np
# import cv2


# MIN_POINTS      = 3        # similarity needs >=2
# MAX_ITERS       = 5000
# CONFIDENCE      = 0.999
# REPROJ_INLIER   = 0.02     # normalized-coord reproj threshold 


# # Data type for a product-level correspondence
# @dataclass
# class ProductMatch:
#     centroid_A: np.ndarray            # (x, y) YOLO bbox center, image A
#     centroid_B: np.ndarray            # (x, y) YOLO bbox center, image B
#     embedding_sim: float              # DINOv3 cosine similarity (0..1)
#     lightglue_score: float = 1.0      # LightGlue match confidence (0..1)
#     bbox_A: Optional[np.ndarray] = None   
#     bbox_B: Optional[np.ndarray] = None   



# def _pts(matches):
#     A = np.array([np.asarray(m.centroid_A, np.float32) for m in matches],
#                  dtype=np.float32)
#     B = np.array([np.asarray(m.centroid_B, np.float32) for m in matches],
#                  dtype=np.float32)
#     return A, B


# def _normalize(pts, w, h):
#     """Change #1: resolution-independent normalization to [0,1]."""
#     out = pts.copy().astype(np.float32)
#     out[:, 0] /= max(float(w), 1.0)
#     out[:, 1] /= max(float(h), 1.0)
#     return out


# def mean_centroid(matches):
#     A, _ = _pts(matches)
#     return A.mean(axis=0)


# def inlier_ratio(remaining, total):
#     return len(remaining) / max(total, 1)


# def _apply_similarity(S, pts):
#     """Apply a 2x3 affine/similarity transform to Nx2 points."""
#     ones = np.ones((len(pts), 1), np.float32)
#     hom = np.hstack([pts.astype(np.float32), ones])   # Nx3
#     return (S @ hom.T).T                              # Nx2


# def _reproj_errors(S, A_norm, B_norm):
#     """Change #2: per-match reprojection error under similarity S."""
#     pred = _apply_similarity(S, A_norm)
#     return np.linalg.norm(pred - B_norm, axis=1)      # N,


# def ransac_similarity_norm(matches, w, h, threshold=0.02):
#     """
#     Fit a SIMILARITY (Jia Eq. 9) with RANSAC on NORMALIZED coords.
#     Returns (S, inlier_mask). threshold is in normalized units.
#     """
#     if len(matches) < 2:
#         return None, None
#     A, B = _pts(matches)
#     An, Bn = _normalize(A, w, h), _normalize(B, w, h)
#     S, mask = cv2.estimateAffinePartial2D(
#         An, Bn,
#         method=cv2.RANSAC,
#         ransacReprojThreshold=threshold,
#         maxIters=MAX_ITERS,
#         confidence=CONFIDENCE,
#     )
#     if S is None or mask is None:
#         return None, None
#     return S, mask.ravel().astype(bool)



# # Progressive RANSAC (Jia et al. 2024, Section 2.1.2)

# def progressive_ransac(matches: List[ProductMatch],
#                        image_size,                    # (width, height)
#                        lenient_thresh: float = 0.05,  # Jia Step 1 (normalized)
#                        strict_thresh: float = 0.02,   # Jia Step 2 (normalized)
#                        min_inlier_ratio: float = 0.30,# Jia Step 4 (stop)
#                        emb_sim_thresh: Optional[float] = None,   
#                        lg_score_thresh: Optional[float] = None   
#                        ) -> List[Dict]:
#     """
#     Groups product correspondences by shelf plane, returning a similarity
#     transform + quality metrics per plane.

#     Args:
#         matches: list[ProductMatch].
#         image_size: (width, height) for coordinate normalization.
#         lenient_thresh / strict_thresh: normalized reproj thresholds.
#         min_inlier_ratio: stop when remaining/total drops below this.
#         emb_sim_thresh: if set, drop matches with DINOv3 sim below it.
#         lg_score_thresh: if set, drop matches with LightGlue score below it.

#     Returns:
#         list of dicts (sorted by score, best first):
#           {inliers, center, S, score, mean_error, n_inliers}
#     """
#     w, h = image_size
#     total = len(matches)
#     if total < MIN_POINTS:
#         return []


#     if emb_sim_thresh is not None:
#         matches = [m for m in matches if m.embedding_sim >= emb_sim_thresh]
#     if lg_score_thresh is not None:
#         matches = [m for m in matches if m.lightglue_score >= lg_score_thresh]
#     if len(matches) < MIN_POINTS:
#         return []

#     # Step 1: lenient RANSAC removes gross outliers 
#     S0, mask0 = ransac_similarity_norm(matches, w, h, lenient_thresh)
#     if S0 is not None and mask0 is not None:
#         matches = [m for m, keep in zip(matches, mask0) if keep]
#     if len(matches) < MIN_POINTS:
#         return []

#     plane_groups = []
#     remaining = matches

#     # Step 4: iterate until inlier ratio < min_inlier_ratio 
#     while inlier_ratio(remaining, total) >= min_inlier_ratio:

#         # Step 2: strict RANSAC 
#         S, mask = ransac_similarity_norm(remaining, w, h, strict_thresh)
#         if S is None or mask is None or mask.sum() < MIN_POINTS:
#             break

#         inliers = [m for m, keep in zip(remaining, mask) if keep]

#         A, B = _pts(inliers)
#         An, Bn = _normalize(A, w, h), _normalize(B, w, h)
#         errs = _reproj_errors(S, An, Bn)
#         mean_error = float(errs.mean())

#         # plane quality score 
#         mean_emb = float(np.mean([m.embedding_sim for m in inliers]))
#         score = (len(inliers) * mean_emb) / (mean_error + 1e-6)

#         center = mean_centroid(inliers)

#         plane_groups.append({
#             "inliers": inliers,
#             "center": center,
#             "S": S,                     
#             "score": score,             
#             "mean_error": mean_error,   
#             "n_inliers": len(inliers),
#         })

#         # remove points by REPROJECTION ERROR, not distance
#         Ar, Br = _pts(remaining)
#         Arn, Brn = _normalize(Ar, w, h), _normalize(Br, w, h)
#         rem_errs = _reproj_errors(S, Arn, Brn)
#         remaining = [m for m, e in zip(remaining, rem_errs)
#                      if e > REPROJ_INLIER]   

#         if len(remaining) < MIN_POINTS:
#             break

#     # return sorted by quality, best plane first
#     plane_groups.sort(key=lambda g: g["score"], reverse=True)
#     return plane_groups


# progressive_ransac.py
# ---------------------------------------------------------------
# Author : Prerana Bora
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
#
# ROBUSTNESS PASS (this version):
#   - [FIX 1] score now DOF-adjusted (overdetermination, not raw n_inliers)
#             to stop small/spurious point sets from outscoring the true
#             dominant shelf plane via artificially low mean_error
#   - [FIX 2] single source of truth for the "inlier" distance threshold,
#             so RANSAC fitting and post-hoc point removal never diverge
#   - [FIX 3] degeneracy / spread check before trusting a fitted plane
#             (rejects near-collinear centroid sets, common along a
#             single shelf row)
#   - [FIX 4] margin test between top-2 planes; flags result as
#             "ambiguous" so the caller (stitcher.py) can route to the
#             SIFT + bundle-adjustment fallback instead of silently
#             trusting a marginal winner
# ---------------------------------------------------------------

from dataclasses import dataclass
from typing import Optional, List, Dict
import numpy as np
import cv2


MIN_POINTS           = 3        # absolute floor: similarity needs >=2, keep 3 for safety
MIN_POINTS_FOR_PLANE = 8        # [FIX 1] a plane must be overdetermined to be *scored/trusted*
SIMILARITY_DOF        = 4        # scale, rotation, tx, ty
MAX_ITERS            = 5000
CONFIDENCE           = 0.999
MIN_SPREAD           = 0.05     # [FIX 3] min std-dev (normalized coords) along weakest axis
AMBIGUITY_RATIO      = 1.5      # [FIX 4] top score must beat 2nd place by this multiple


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
    """Resolution-independent normalization to [0,1]."""
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
    """Per-match reprojection error under similarity S (normalized coords)."""
    pred = _apply_similarity(S, A_norm)
    return np.linalg.norm(pred - B_norm, axis=1)      # N,


def _is_well_conditioned(pts_norm, min_spread=MIN_SPREAD):
    """
    [FIX 3] Reject near-degenerate point configurations before trusting
    the fitted transform. YOLO centroids along a single shelf row are
    often close to collinear, which lets estimateAffinePartial2D return
    a "consistent" but poorly-constrained similarity transform.

    Uses the smallest eigenvalue of the point covariance as a proxy for
    spread along the weakest axis; a near-collinear set has one axis
    with ~0 spread.
    """
    if len(pts_norm) < 3:
        return False
    cov = np.cov(pts_norm.T)
    if cov.shape != (2, 2):
        return False
    eigvals = np.linalg.eigvalsh(cov)
    return eigvals.min() > (min_spread ** 2)


def ransac_similarity_norm(matches, w, h, threshold):
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


# ---------------------------------------------------------------
# Progressive RANSAC (Jia et al. 2024, Section 2.1.2)
# ---------------------------------------------------------------

def progressive_ransac(matches: List[ProductMatch],
                       image_size,                    # (width, height)
                       lenient_thresh: float = 0.05,  # Jia Step 1 (normalized)
                       strict_thresh: float = 0.02,   # Jia Step 2 (normalized)
                       min_inlier_ratio: float = 0.30,# Jia Step 4 (stop)
                       emb_sim_thresh: Optional[float] = None,
                       lg_score_thresh: Optional[float] = None,
                       min_points_for_plane: int = MIN_POINTS_FOR_PLANE,
                       ambiguity_ratio: float = AMBIGUITY_RATIO,
                       min_spread: float = MIN_SPREAD,
                       ) -> List[Dict]:
    """
    Groups product correspondences by shelf plane, returning a similarity
    transform + quality metrics per plane.

    Args:
        matches: list[ProductMatch].
        image_size: (width, height) for coordinate normalization.
        lenient_thresh / strict_thresh: normalized reproj thresholds.
            NOTE: strict_thresh is now the SINGLE source of truth for
            both the RANSAC fit AND the post-hoc point-removal step
            ([FIX 2]) -- previously a separate module constant
            (REPROJ_INLIER) could silently drift out of sync with
            strict_thresh if only one was tuned.
        min_inlier_ratio: stop when remaining/total drops below this.
        emb_sim_thresh: if set, drop matches with DINOv3 sim below it.
        lg_score_thresh: if set, drop matches with LightGlue score below it.
        min_points_for_plane: [FIX 1] a candidate plane needs at least
            this many inliers to be scored/accepted at all. Below this,
            the fit is too close to the minimal 3-point solution and
            mean_error collapses toward zero regardless of whether the
            plane is real -- which previously let spurious small planes
            outscore the true dominant shelf plane.
        ambiguity_ratio: [FIX 4] if the best plane's score doesn't beat
            the second-best by at least this multiple, the result list
            carries a top-level "ambiguous" flag so the caller can
            route to a fallback (e.g. SIFT + bundle adjustment) instead
            of silently trusting a marginal winner.
        min_spread: [FIX 3] minimum spread (normalized coords, weakest
            axis) required to accept a plane's fit as well-conditioned;
            guards against near-collinear centroid sets (e.g. all
            products along one shelf row).

    Returns:
        list of dicts (sorted by score, best first), each:
          {inliers, center, S, score, mean_error, n_inliers,
           well_conditioned, ambiguous}
        "ambiguous" is only set (True) on the first (best) plane's dict,
        and only when there IS a second plane to compare against.
        An empty list is returned if no valid plane could be extracted.
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

        # [FIX 3] degeneracy check -- skip planes fit on a near-collinear
        # or too-tightly-clustered point set, since the transform is
        # poorly constrained even though RANSAC reports it as consistent.
        well_conditioned = _is_well_conditioned(An, min_spread=min_spread)

        errs = _reproj_errors(S, An, Bn)
        mean_error = float(errs.mean())
        mean_emb = float(np.mean([m.embedding_sim for m in inliers]))
        n_inliers = len(inliers)

        # [FIX 1] DOF-adjusted score. Raw n_inliers rewards small,
        # easily-overfit point sets: with few points near the minimal
        # 3-point solution, mean_error collapses toward zero simply
        # because there's little surplus data to disagree with the fit.
        # Using "overdetermination" (points beyond the DOF needed to fit
        # a similarity transform) makes the score reflect how much the
        # fit is actually being *tested*, not just how tightly it agrees
        # with a handful of points.
        overdetermination = max(n_inliers - SIMILARITY_DOF, 1)

        if n_inliers >= min_points_for_plane and well_conditioned:
            score = (overdetermination * mean_emb) / (mean_error + 1e-6)
        else:
            # Still record the plane (useful for diagnostics/visualization)
            # but score it at zero so it can never be selected as the
            # dominant plane while under-supported or poorly conditioned.
            score = 0.0

        center = mean_centroid(inliers)

        plane_groups.append({
            "inliers": inliers,
            "center": center,
            "S": S,
            "score": score,
            "mean_error": mean_error,
            "n_inliers": n_inliers,
            "well_conditioned": well_conditioned,
        })

        # [FIX 2] remove points using the SAME threshold used to fit this
        # plane (strict_thresh), instead of a separately-tuned constant
        # that could drift out of sync if only one of the two was ever
        # adjusted per-store.
        Ar, Br = _pts(remaining)
        Arn, Brn = _normalize(Ar, w, h), _normalize(Br, w, h)
        rem_errs = _reproj_errors(S, Arn, Brn)
        remaining = [m for m, e in zip(remaining, rem_errs)
                     if e > strict_thresh]

        if len(remaining) < MIN_POINTS:
            break

    # return sorted by quality, best plane first
    plane_groups.sort(key=lambda g: g["score"], reverse=True)

    # [FIX 4] margin/ambiguity test between the top two candidates.
    # A close call here is exactly the failure mode observed on Store 3
    # (dominant-plane selection not matching the true shelf plane) --
    # flag it instead of silently trusting the top score.
    if len(plane_groups) >= 2:
        top_score = plane_groups[0]["score"]
        second_score = plane_groups[1]["score"]
        is_ambiguous = top_score < second_score * ambiguity_ratio
        plane_groups[0]["ambiguous"] = is_ambiguous
    elif len(plane_groups) == 1:
        plane_groups[0]["ambiguous"] = False

    return plane_groups