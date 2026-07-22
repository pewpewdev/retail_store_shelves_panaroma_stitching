Snap2Insight — CV Engineer Take-Home Assignment: data
=====================================================

Directory structure
-------------------
stitching_assignment_data/
├── README.txt                    this file
├── store_1/
│   ├── images/                   input: individual photos captured in the store
│   └── reference_preview.jpg     preview of our pipeline's output — your quality bar
├── store_2/
├── store_3/
├── store_4/
└── store_5/                      (same layout for every store)

Notes
-----
- reference_preview.jpg is downscaled to keep this zip small. The full-
  resolution reference for each store is on the assignment page (Data tab,
  "Reference full resolution").
- Image filenames are the original capture UUIDs; they carry no ordering
  information. Discovering the arrangement of the images is part of the task.
- Every image in a store overlaps with at least one other image — they all
  belong in the panorama.
- Your deliverable: one stitched image per store, named
  store_<n>_stitched.jpg.

Confidential: these are real in-store photos. Use them only for this
assignment — do not publish them or include them in a public repository.
