"""
NeuroFusion AI
Experiment 02 - Mild 3D sMRI Spatial Augmentation

Training-only augmentation.

Validation and test volumes must never use this transform.
"""

import numpy as np
from scipy import ndimage


def random_flip(
    volume: np.ndarray,
    probability: float = 0.5,
) -> np.ndarray:

    if np.random.rand() < probability:

        volume = np.flip(
            volume,
            axis=2,
        ).copy()

    return volume


def random_rotation(
    volume: np.ndarray,
    max_degrees: float = 5.0,
) -> np.ndarray:

    angle = np.random.uniform(
        -max_degrees,
        max_degrees,
    )

    volume = ndimage.rotate(
        volume,
        angle=angle,
        axes=(1, 2),
        reshape=False,
        order=1,
        mode="constant",
        cval=0.0,
        prefilter=False,
    )

    return volume.astype(
        np.float32,
        copy=False,
    )


def random_translation(
    volume: np.ndarray,
    max_voxels: int = 3,
) -> np.ndarray:

    shifts = np.random.randint(
        -max_voxels,
        max_voxels + 1,
        size=3,
    )

    volume = ndimage.shift(
        volume,
        shift=tuple(
            int(value)
            for value in shifts
        ),
        order=1,
        mode="constant",
        cval=0.0,
        prefilter=False,
    )

    return volume.astype(
        np.float32,
        copy=False,
    )


def apply_training_augmentation(
    image: np.ndarray,
) -> np.ndarray:

    """
    Apply mild random 3D spatial augmentation.

    Input:
        (1, D, H, W)

    Output:
        (1, D, H, W)

    The channel dimension is preserved.
    """

    if image.ndim != 4:

        raise ValueError(
            f"Expected 4D tensor "
            f"(C,D,H,W), got {image.shape}"
        )

    if image.shape[0] != 1:

        raise ValueError(
            f"Expected one channel, "
            f"got shape {image.shape}"
        )

    volume = image[0]

    volume = random_flip(
        volume,
        probability=0.5,
    )

    volume = random_rotation(
        volume,
        max_degrees=5.0,
    )

    volume = random_translation(
        volume,
        max_voxels=3,
    )

    volume = np.clip(
        volume,
        0.0,
        1.0,
    )

    return volume[
        np.newaxis,
        ...
    ].astype(
        np.float32,
        copy=False,
    )