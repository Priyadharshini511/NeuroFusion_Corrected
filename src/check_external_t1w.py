import os
import nibabel as nib
import numpy as np


ROOT = os.path.join(
    "processed_data",
    "smri",
    "external_test",
    "ds004302",
    "pilot",
)

SUBJECTS = [
    ("sub-01", "CONTROL"),
    ("sub-26", "SCHIZOPHRENIA"),
    ("sub-55", "SCHIZOPHRENIA"),
]


print("=" * 80)
print("OPENNEURO EXTERNAL T1W GEOMETRY CHECK")
print("=" * 80)


for subject, label in SUBJECTS:

    path = os.path.join(
        ROOT,
        subject,
        "anat",
        f"{subject}_T1w.nii.gz",
    )

    print("\n" + "-" * 80)
    print(f"SUBJECT : {subject}")
    print(f"LABEL   : {label}")
    print(f"FILE    : {path}")
    print("-" * 80)

    if not os.path.isfile(path):
        print("STATUS: FILE NOT FOUND")
        continue

    try:

        img = nib.load(path)

        data = img.get_fdata(
            dtype=np.float32
        )

        print("Shape:", img.shape)

        print(
            "Voxel size:",
            img.header.get_zooms()[:3]
        )

        print(
            "Orientation:",
            nib.aff2axcodes(img.affine)
        )

        print(
            "Affine:"
        )

        print(img.affine)

        print(
            "Data type:",
            img.get_data_dtype()
        )

        print(
            "Intensity min:",
            float(np.min(data))
        )

        print(
            "Intensity max:",
            float(np.max(data))
        )

        print(
            "Intensity mean:",
            float(np.mean(data))
        )

        print(
            "Intensity std:",
            float(np.std(data))
        )

        print(
            "Finite:",
            bool(np.all(np.isfinite(data)))
        )

        print(
            "Non-zero voxels:",
            int(np.count_nonzero(data))
        )

        print(
            "STATUS: READ SUCCESSFULLY"
        )

    except Exception as e:

        print(
            "STATUS: FAILED"
        )

        print(
            "ERROR:",
            e
        )


print("\n" + "=" * 80)
print("CHECK COMPLETE")
print("=" * 80)