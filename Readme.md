*Store Panorama Stitching*
Stitches multiple overlapping retail store shelf images into a single seamless panorama.
Detailed report: Build_panorama_for_store_by_PreranaB.pdf

Project Structure
store_panaroma_stiching/
│
├── stitch_panaroma.py                  # Approach 1: OpenCV + SIFT
├── stitch_superpoint.py                # Approach 2: SuperPoint + LightGlue
├── stitch_loftr.py                     # Approach 3: LoFTR
│
├── reference_provided/                 # Reference panoramas provided
├── retailglue_base/                    # RetailGlue base results
├── retailglue_adapted/                 # RetailGlue + Progressive RANSAC results
│
├── output_opencv/                      # Output: Approach 1
├── output_superpoint/                  # Output: Approach 2
├── output_loftr/                       # Output: Approach 3
│
└── RetailGlue/                         # Final Approach (cloned)
    ├── batch_stitch.py                 # Inference script
    └── retailglue/
        └── stitching/
            ├── stitcher.py             # Main stitcher
            └── progressive_ransac.py  # Personal contribution (adapted from Jia et al. 2024)


Run Inference
uv run python batch_stitch.py \
    --data_root /Users/testuser/Desktop/store_panaroma_stiching/stitching_assignment_data \
    --model_name lightglue_dinov3_vits \
    --device mps


Author
Prerana Bora
