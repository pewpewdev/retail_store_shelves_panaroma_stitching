# Store Panorama Stitching

Stitches multiple overlapping retail store shelf images into a single seamless panorama.

📄 Detailed Report: `Build_panorama_for_store_by_PreranaB.pdf`

---

## Project Structure

```
store_panaroma_stiching/
│
├── Build_panorama_for_store_by_PreranaB.pdf # Detailed report
├── Readme.md # This file
├── compare_reference_vs_output.jpg # Visual comparison
│
├── reference_image_provided/ # Reference panoramas provided
│
├── retailglue_output/ # RetailGlue base results
│ ├── store1.jpg
│ ├── store2_without_modification.jpg 
│ ├── store2_with_modification.jpg #logic to handle >=2 images
│ ├── store3.jpg
│ ├── store4.jpg
│ ├── store5.jpg
│ └── all_stores_grid.jpg
│
├── retailglue_progressive_ransac_output/ # RetailGlue + Progressive RANSAC results
│ ├── store1.jpg
│ ├── store2_without_modification.jpg
│ ├── store2_with_modification.jpg #logic to handle >=2 images
│ ├── store3.jpg
│ ├── store4.jpg
│ ├── store5.jpg
│ └── all_store_grid.jpg
│
├── Failed_experiments/ # Earlier approaches (not used in final)
│ ├── panaroma_stitch_opencv.py # Approach 1: OpenCV + SIFT
│ ├── panaroma_stitch_superpoint.py # Approach 2: SuperPoint + LightGlue
│ ├── panaroma_stitching_loftr.py # Approach 3: LoFTR
│ ├── output_opencv/ # Output: Approach 1
│ ├── output_superpoint/ # Output: Approach 2
│ └── output_loftr/ # Output: Approach 3
│
└── RetailGlue/ # Final Approach (cloned + modified)
 ├── batch_stitch.py # End to end Inference script
 └── retailglue/
 ├── entities.py # Core data types
 ├── config.py # YAML configuration loader
 ├── io.py # Image I/O utilities
 ├── detector.py # SKU YOLO product detector
 ├── embeddings.py # DINOv3 embedding extraction
 ├── visualization.py # Drawing and visualization
 ├── matchers/
 │ ├── __init__.py # Matcher factory
 │ ├── lightglue.py # Product-level LightGlue (core contribution)
 │ ├── lightgluestick.py # LightGlueStick baseline
 │ ├── gluestick.py # GlueStick baseline
 │ ├── roma.py # RoMa v2 baseline
 │ └── hf_model.py # HuggingFace models
 ├── stitching/
 │ ├── stitcher.py # Core stitching engine — modified (2-image tilt fix + prograssive_ransac logic)
 │ ├── progressive_ransac.py # New Additional concept to RetailGlue (optional)
 │ ├── blender.py # Adaptive distance-transform blending
 │ └── transforms.py # Homography-based detection transformation
 ├── training/
 │ ├── __init__.py
 │ ├── dataset.py # Product pairs dataset
 │ ├── losses.py # NLL loss with focal weighting
 │ ├── metrics.py # Matching recall, precision, AP
 │ └── trainer.py # Training loop
 └── benchmark/
 ├── runner.py # Benchmark orchestrator
 ├── evaluation.py # IOU matching, Hungarian assignment
 ├── stats.py # Precision, Recall, F1
 └── drawer.py # Result visualization
```

---

## Approaches

| Approach | Script | Status |
|---|---|---|
| OpenCV + SIFT | `Failed_experiments/panaroma_stitch_opencv.py` |  Failed |
| SuperPoint + LightGlue | `Failed_experiments/panaroma_stitch_superpoint.py` |  Failed |
| LoFTR | `Failed_experiments/panaroma_stitching_loftr.py` |  Failed |
| RetailGlue (base) | `RetailGlue/batch_stitch.py` |  Used |
| RetailGlue + Progressive RANSAC(optional) + 2-image fix | `RetailGlue/batch_stitch.py` |  Final |

---

## Personal Contributions

- **`stitching/stitcher.py`** — Added 2-image tilt fix:
 - Hybrid stitching path (RetailGlue matches + `cv2.detail` blending)
 - Homography decomposition with least-tilt rotation selection
 - `waveCorrect(HORIZ)` for panorama straightening
 - `cv2. Stitcher` fallback for low-inlier cases
 - Black border cropping post-warp

- **`stitching/progressive_ransac.py`** — Adapted from Jia et al. 2024

---

## Run Inference

```bash
uv run python batch_stitch.py \\
 --data_root /Users/testuser/Desktop/store_panaroma_stiching/stitching_assignment_data \\
 --model_name lightglue_dinov3_vits \\
 --device mps
```
## Comparative Study

Compared with the provided reference panoramas:

- The improved RetailGlue pipeline generates **higher-resolution panorama images** than the provided reference panoramas.
- The generated panoramas are **more zoomed-in**, resulting in better visibility of product-level details than the reference panoramas.

## References

1. **Jia, et al.**  
   *Semantic-Aware Stitching for Panorama*.  
   (Progressive RANSAC implementation adapted from this work.)

2. **Arda Oztuner, Ayberk Celik, Ibrahim Samil Yalciner, Server Calap.**  
   *RetailGlue: Semantic Product-Level Image Stitching for Retail Shelf Panoramas*.  
   (Base implementation used and modified in this project.)
   
---
---

## Author

**Prerana Bora** 
