# NeuroFusion – sMRI Preprocessing Pipeline

## Overview

This project processes structural MRI (sMRI) data from the COBRE dataset for downstream multimodal neuroimaging analysis.

The preprocessing pipeline used for the first dataset (`scan_data001`) was completed successfully and should be followed consistently for the remaining datasets.

The main goal is to convert the raw MRI data into a standardized feature matrix containing AAL-based regional sMRI features.

---

# Dataset Organization

There are a total of:

- 22 ZIP datasets
- 1 dataset has already been completely processed
- 21 ZIP datasets remain

The first processed dataset was:

`scan_data001`

The remaining ZIP files should be processed using the same pipeline described below.

---

# IMPORTANT RULE

For every new ZIP dataset:

1. Extract the ZIP temporarily.
2. Inspect the extracted data.
3. Identify the valid structural MRI / MPRAGE scan.
4. Convert DICOM to NIfTI if required.
5. Reorient the structural MRI.
6. Apply N4 bias-field correction.
7. Perform HD-BET brain extraction.
8. Register the brain image to MNI152 2 mm space.
9. Perform registration quality control.
10. Use the verified AAL atlas.
11. Resample the AAL atlas to the exact MNI grid if required.
12. Extract 116 AAL ROI features.
13. Create/append the combined feature matrix.
14. Verify the resulting CSV.
15. Only after all required outputs are safely stored should the temporary extracted raw folder be deleted.

Do NOT change the preprocessing method between datasets unless a specific dataset requires troubleshooting.

---

# Current Project Pipeline

```text
Raw ZIP Dataset
      |
      v
Temporary Extraction
      |
      v
DICOM MRI Data
      |
      v
DICOM -> NIfTI
      |
      v
MRI Inspection / MPRAGE Identification
      |
      v
Reorientation
      |
      v
N4 Bias Field Correction
      |
      v
HD-BET Brain Extraction
      |
      v
MNI152 2 mm Affine Registration
      |
      v
Registration QC
      |
      v
Verified AAL SPM12 Atlas
      |
      v
AAL -> Exact MNI Grid
      |
      v
AAL ROI Feature Extraction
      |
      v
116 sMRI ROI Features / Subject
      |
      v
Combined sMRI Feature Matrix